import argparse
import json
import os
from types import MethodType
import torch
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from utils.utils import get_test_questions

# ---------- SAE Loading and Inference ----------
class JumpReLUInferenceSAE(torch.nn.Module):
    """SAE with JumpReLU activation using sae_lens weight format"""
    def __init__(self, W_dec, W_enc, b_dec, b_enc, jumprelu_init_threshold=0.001, jumprelu_bandwidth=0.001):
        super().__init__()
        self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.jumprelu_init_threshold = jumprelu_init_threshold
        self.jumprelu_bandwidth = jumprelu_bandwidth
    
    @torch.no_grad()
    def encode(self, x):
        """Encode using JumpReLU activation function"""
        # Linear transformation
        # x: [batch, d_in], W_enc: [d_in, d_sae], b_enc: [d_sae]
        z = torch.matmul(x, self.W_enc) + self.b_enc
        # JumpReLU: f(z) = z if z > threshold else 0
        f = torch.where(z > self.jumprelu_init_threshold, z, torch.zeros_like(z))
        return f
    
    @torch.no_grad()
    def decode(self, f):
        """Decode sparse features back to activation space"""
        # f: [batch, d_sae], W_dec: [d_sae, d_in], b_dec: [d_in]
        return torch.matmul(f, self.W_dec) + self.b_dec

def load_sae_from_safetensors(repo_id, file_path, device="cuda", dtype=torch.bfloat16):
    """Load SAE from safetensors format (sae_lens format)
    
    Args:
        repo_id: HuggingFace repository ID
        file_path: Path to safetensors file within repo
        device: Device to load to
        dtype: Data type to convert SAE weights to
    """
    print(f"Downloading SAE from {repo_id}/{file_path}...")
    hf_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        revision="main",
    )
    print(f"Loading SAE from {hf_path}...")
    state = load_file(hf_path)
    
    # sae_lens uses these weight names
    W_dec = state["W_dec"].to(device).to(dtype)
    W_enc = state["W_enc"].to(device).to(dtype)
    b_dec = state["b_dec"].to(device).to(dtype)
    b_enc = state["b_enc"].to(device).to(dtype)
    
    # Use sae_lens default JumpReLU parameters
    jumprelu_init_threshold = 0.001
    jumprelu_bandwidth = 0.001
    
    sae = JumpReLUInferenceSAE(
        W_dec, W_enc, b_dec, b_enc,
        jumprelu_init_threshold=jumprelu_init_threshold,
        jumprelu_bandwidth=jumprelu_bandwidth
    ).eval()
    
    return sae

def factory_layernorm_sae(sae_idx, sae, steering_diff_sae, steering_strength):
    """Hook to apply SAE steering in LayerNorm after residual addition
    
    This hooks into the LayerNorm's forward method (forward_native or equivalent).
    The LayerNorm receives hidden_states and residual, combines them, and normalizes.
    We intercept after the combination but before normalization to apply SAE steering.
    
    Process:
    1. Reconstruct combined stream (hidden_states + residual)
    2. Encode to SAE sparse activations
    3. Apply steering in SAE latent space
    4. Decode back to residual stream
    5. Apply LayerNorm normalization to steered stream
    """
    def layernorm_forward_hook(self, x, residual=None):
        # Reconstruct combined stream (this is what LayerNorm normally adds)
        x_combined = x.to(torch.float32)
        if residual is not None:
            x_combined = x_combined + residual
        
        # Convert to SAE dtype (bfloat16 to match SAE weights)
        x_combined_sae = x_combined.to(torch.bfloat16)
        
        # Prepare for SAE processing
        original_shape = x_combined_sae.shape
        if x_combined_sae.dim() == 3:
            flat_combined = x_combined_sae.reshape(-1, x_combined_sae.shape[-1])
        else:
            flat_combined = x_combined_sae
        
        # 1. Encode to SAE sparse space
        sparse_acts = sae.encode(flat_combined)
        
        # 2. Apply steering in SAE latent space
        steering_vector_sae = steering_diff_sae.to(flat_combined.device).to(flat_combined.dtype)
        steering_vector_sae = steering_vector_sae / (steering_vector_sae.norm(p=2) + 1e-8)
        sparse_acts_steered = sparse_acts + steering_strength * steering_vector_sae.unsqueeze(0)
        
        # 3. Decode back to residual stream
        x_combined_steered = sae.decode(sparse_acts_steered)

        # --- NEW: reconstruction-error correction ---
        # Compute residual from original reconstruction
        recon_error = x_combined - sae.decode(sparse_acts)
        x_combined_steered = x_combined_steered + recon_error
        # --------------------------------------------
        
        if original_shape[0] > 1 and len(original_shape) == 3:
            x_combined_steered = x_combined_steered.reshape(original_shape)
        
        # Convert back to float32 for LayerNorm computation
        x_combined_steered = x_combined_steered.to(torch.float32)
        
        # 4. Manually apply RMSNorm to the steered stream
        # This mimics what the original LayerNorm does
        x_var = x_combined_steered
        if hasattr(self, 'variance_size_override') and self.variance_size_override is not None:
            if x_combined_steered.shape[-1] >= self.variance_size_override:
                x_var = x_combined_steered[:, :, :self.variance_size_override]
        
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x_combined_steered * torch.rsqrt(variance + self.variance_epsilon)
        
        # Cast to original dtype and apply weight if it exists
        orig_dtype = x.dtype  # Get original dtype from input
        x_normalized = x_normalized.to(orig_dtype)
        
        if hasattr(self, 'has_weight') and self.has_weight and self.weight is not None:
            x_normalized = x_normalized * self.weight
        
        # Return normalized output with residual if it was provided
        if residual is None:
            return x_normalized
        else:
            return x_normalized, residual
    
    return layernorm_forward_hook

def get_model_layers(model):
    """Get model layers with compatibility for different model architectures"""
    try:
        # Try Llama-style models
        return model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    except AttributeError:
        try:
            # Try Gemma-3 style models
            return model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers
        except AttributeError:
            raise RuntimeError("Could not access model layers. Model architecture may not be supported.")

def load_sae_steering_vectors(steering_dir, languages, sae_repo, sae_layers=None, model_variant="llama", dtype=torch.bfloat16):
    """Load SAE steering vectors for specified languages and layers
    
    Args:
        steering_dir: Directory containing steering vector files
        languages: List of 2-letter language codes
        sae_repo: Repository containing SAEs
        sae_layers: List of layer indices to load SAEs for (default: [4])
        model_variant: Model variant name (default: "llama")
        dtype: Data type to convert SAE weights to (default: bfloat16)
    
    Returns:
        Dictionary mapping language codes to SAE steering vectors
        Dictionary mapping sae_idx to (SAE model, actual layer index)
        List of actual SAE layer indices
    """
    steering_data_sae = {}
    saes = {}
    sae_dict_size = None
    
    # Default to layer 4 if not specified
    if sae_layers is None:
        sae_layers = [4]
    
    # Load steering vectors from .sae files
    for lang in languages:
        filepath = os.path.join(steering_dir, f'sae-fast.{lang}.{model_variant}')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SAE steering vector file not found for '{lang}': {filepath}")
        data = torch.load(filepath, map_location='cpu')

        steering_data_sae[lang] = data['steering_vectors']  # Shape: [num_sae_layers, sae_dict_size]
        
        if sae_dict_size is None:
            sae_dict_size = data.get('sae_dict_size')
        
        print(f"Loaded SAE steering vectors for {lang}: {data['steering_vectors'].shape}")
    
    # Load SAE models for specified layers
    print(f"\nLoading SAEs for layers: {sae_layers}...")
    
    for sae_idx, layer_idx in enumerate(sae_layers):
        sae_path = f"FAST/blocks_{layer_idx}_hook_resid_post_8X_2048_jumprelu/sae_weights.safetensors"
        try:
            sae = load_sae_from_safetensors(
                sae_repo,
                sae_path,
                device="cuda",
                dtype=dtype
            )
            saes[sae_idx] = (sae, layer_idx)
            print(f"Loaded SAE for layer {layer_idx}")
        except Exception as e:
            print(f"Error loading SAE for layer {layer_idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(saes)} SAE(s)\n")
    
    return steering_data_sae, saes, sae_layers

def apply_sae_steering_to_layernorms(model, steering_data_sae, saes, sae_layer_indices, 
                                      source_lang, target_lang, steering_strength, layer_range=None):
    """Apply SAE steering to the NEXT layer's input_layernorm
    
    If SAE is for layer N, hook into layer N+1's input_layernorm
    because that's where the residual stream from layer N gets processed.
    
    Args:
        model: vLLM model
        steering_data_sae: Steering vectors keyed by language
        saes: Dict mapping sae_idx to (sae, layer_idx)
        sae_layer_indices: List of actual model layer indices with SAEs
        source_lang: Source language code
        target_lang: Target language code
        steering_strength: Scaling factor for steering
        layer_range: Tuple (start, end) to limit which layers to steer, or None for all
    """
    layers = get_model_layers(model)
    
    # Compute steering difference in SAE space
    steering_diff_sae = (steering_data_sae[target_lang] - steering_data_sae[source_lang])
    # Shape: [num_sae_layers, sae_dict_size]
    
    original_forwards = []
    
    # Filter SAEs based on layer range
    saes_to_apply = {}
    for sae_idx, (sae, layer_idx) in saes.items():
        if layer_range is not None:
            start, end = layer_range
            if not (start <= layer_idx < end):
                continue
        saes_to_apply[sae_idx] = (sae, layer_idx)
    
    # Apply hooks to NEXT layer's input_layernorm
    for sae_idx, (sae, sae_layer_idx) in saes_to_apply.items():
        # Hook into next layer's input_layernorm
        next_layer_idx = sae_layer_idx + 1
        
        if next_layer_idx >= len(layers):
            print(f"Warning: Next layer {next_layer_idx} exceeds model depth {len(layers)}")
            continue
        
        next_layer = layers[next_layer_idx]
        layernorm = next_layer.input_layernorm
        
        # Store original forward method
        original_forwards.append((next_layer_idx, 'input_layernorm', layernorm.forward))
        layernorm._original_forward = layernorm.forward
        
        # Replace with hooked version using MethodType
        layernorm.forward = MethodType(
            factory_layernorm_sae(sae_idx, sae, steering_diff_sae[sae_idx], steering_strength),
            layernorm
        )
        
        print(f"Hooked SAE {sae_idx} (layer {sae_layer_idx}) into layer {next_layer_idx}'s input_layernorm")
    
    print(f"Applied SAE steering to {len(original_forwards)} LayerNorm layers")
    if layer_range:
        print(f"Layer range: {layer_range[0]} to {layer_range[1]-1}")
    return original_forwards

def restore_layernorm_forwards(model, original_forwards):
    """Restore original LayerNorm forward methods"""
    layers = get_model_layers(model)
    
    for layer_idx, norm_name, original_forward in original_forwards:
        layer = layers[layer_idx]
        layernorm = getattr(layer, norm_name)
        layernorm.forward = original_forward
        if hasattr(layernorm, '_original_forward'):
            delattr(layernorm, '_original_forward')
    
    # Force GPU memory cleanup after restoration
    torch.cuda.empty_cache()

def verify_model_reset(model):
    """Check if any layers still have steering hooks attached"""
    layers = get_model_layers(model)
    contaminated_layers = []
    
    for idx, layer in enumerate(layers):
        if hasattr(layer.input_layernorm, '_original_forward'):
            contaminated_layers.append(idx)
    
    if contaminated_layers:
        print(f"WARNING: Layers {contaminated_layers} still have steering hooks!")
        return False
    return True

def create_chat_messages(prompts):
    """Create simple chat messages for steering experiments"""
    return [[{"role": "user", "content": prompt}] for prompt in prompts]

def run_sae_steering_experiment(model, sampling_params, test_questions, source_lang, target_lang, 
                                steering_data_sae, saes, sae_layer_indices, steering_strength, 
                                layer_range=None):
    """Run a single SAE steering experiment"""
    
    print(f"\n{'='*60}")
    print(f"SAE Steering Experiment: {source_lang} → {target_lang}")
    print(f"Steering strength: {steering_strength}")
    if layer_range:
        print(f"Layer range: {layer_range[0]} to {layer_range[1]-1}")
    print(f"{'='*60}")

    assert source_lang in steering_data_sae, f"Source language {source_lang} not in steering_data!"
    assert target_lang in steering_data_sae, f"Target language {target_lang} not in steering_data!"
    
    # Apply SAE steering
    original_forwards = apply_sae_steering_to_layernorms(model, steering_data_sae, saes, sae_layer_indices,
                                                          source_lang, target_lang, steering_strength, layer_range)
    
    if source_lang not in test_questions:
        print(f"Warning: No test questions found for {source_lang}")
        print(f"Available languages in test_questions: {list(test_questions.keys())}")
        restore_layernorm_forwards(model, original_forwards)
        return None
    
    test_prompts = test_questions[source_lang]
    print(f"Using test questions for: {source_lang}")
    
    # Create batch of chat messages
    messages_batch = create_chat_messages(test_prompts)
    
    print(f"Running batched inference on {len(messages_batch)} questions...")
    
    # Generate all responses in one batch using chat format
    outputs = model.chat(messages_batch, sampling_params=sampling_params)
    
    # Process results
    all_results = []
    for i, (output, original_prompt) in enumerate(zip(outputs, test_prompts)):
        response = output.outputs[0].text.strip()
        
        all_results.append({
            "question_idx": i,
            "input": original_prompt,
            "output": response
        })
        
        if i < 3:  # Print first 3 for verification
            print(f"Q{i+1}: {original_prompt[:50]}...")
            print(f"R{i+1}: {response[:100]}...\n")
    
    # Restore original model
    restore_layernorm_forwards(model, original_forwards)
    
    layer_range_str = f"{layer_range[0]}-{layer_range[1]-1}" if layer_range else "all"
    
    results = {
        "source_language": source_lang,
        "target_language": target_lang,
        "steering_strength": steering_strength,
        "steering_method": "sae_diffmean_layernorm",
        "num_sae_layers": len(saes),
        "sae_layer_indices": sae_layer_indices,
        "layer_range": layer_range_str,
        "model": args.model,
        "results": all_results
    }
    
    return results

def check_experiment_exists(source_lang, target_lang, output_dir, steering_strength, layer_range=None):
    """Check if experiment results already exist"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    layer_range_str = f"layers_{layer_range[0]}_{layer_range[1]-1}" if layer_range else "all_layers"
    output_subdir = os.path.join(output_dir, strength_str, layer_range_str)
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    return os.path.exists(output_file)

def save_results(results, output_dir, steering_strength, layer_range=None):
    """Save experiment results to JSON file"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    layer_range_str = f"layers_{layer_range[0]}_{layer_range[1]-1}" if layer_range else "all_layers"
    output_subdir = os.path.join(output_dir, strength_str, layer_range_str)
    os.makedirs(output_subdir, exist_ok=True)
    
    source_lang = results["source_language"]
    target_lang = results["target_language"]
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Apply SAE steering vectors to language models via LayerNorm hooks")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--steering_dir", type=str, default="data_llama_3-1", 
                       help="Directory containing SAE steering vector files")
    parser.add_argument("--sae_repo", type=str, default="Geaming/Llama-3.1-8B-Instruct_SAEs",
                       help="HuggingFace repo containing SAE models")
    parser.add_argument("--sae_layers", type=int, nargs='+', default=[4, 12, 18, 20, 25],
                       help="SAE layer indices to load (e.g., --sae_layers 4 12 18 20 25)")
    parser.add_argument("--output", type=str, default="steering_results_sae_layernorm")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language (2-letter code)")
    parser.add_argument("--target_lang", type=str, default="fr", help="Target language (2-letter code)")
    parser.add_argument("--languages", nargs='+', 
                       default=["en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk", "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"],
                       help="Languages to test (2-letter codes)")
    parser.add_argument("--steering_strength", type=float, default=1.0, 
                       help="Scaling factor for steering vector (alpha)")
    parser.add_argument("--layer_start", type=int, default=None, 
                       help="First layer to apply steering (default: all layers with SAEs)")
    parser.add_argument("--layer_end", type=int, default=None, 
                       help="Last layer to apply steering (default: all layers with SAEs)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--skip_existing", action='store_true', 
                       help="Skip experiments that already have results")

    global args
    args = parser.parse_args()

    print("="*70)
    print("SAE Steering Experiment")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"SAE Layers: {args.sae_layers}")
    print(f"Languages: {args.languages}")
    print(f"Steering Strength: {args.steering_strength}")
    print("="*70 + "\n")

    print("Loading model and SAE steering vectors...")
    
    # Load SAE steering vectors and SAE models
    languages_to_load = args.languages
    print(f"Loading SAE steering vectors for: {languages_to_load}")
    
    steering_data_sae, saes, sae_layer_indices = load_sae_steering_vectors(
        args.steering_dir, languages_to_load, args.sae_repo, 
        sae_layers=args.sae_layers, dtype=torch.bfloat16
    )
    
    print(f"Available SAE layers: {sae_layer_indices}")
    
    # Determine layer range
    layer_range = None
    if args.layer_start is not None or args.layer_end is not None:
        start = args.layer_start if args.layer_start is not None else min(sae_layer_indices)
        end = args.layer_end if args.layer_end is not None else max(sae_layer_indices) + 1
        layer_range = (start, end)
        print(f"Applying steering to layers {start} to {end-1}")
    else:
        print(f"Applying steering to all available SAE layers: {sae_layer_indices}")
    
    # Load test questions (assuming they are keyed by 2-letter codes)
    test_questions = get_test_questions(k=70)
    
    # Initialize model
    print("\nInitializing LLM...")
    model = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )
    
    # Get model-specific stop tokens
    tokenizer = model.get_tokenizer()
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    print(f"Using stop tokens: {stop_token_ids}")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        repetition_penalty=1.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        skip_special_tokens=True
    )
    
    if args.batch_mode:
        languages = args.languages
        total_combinations = len(languages) * len(languages)
        counter = 0
        skipped = 0

        print(f"\nLanguage processing order: {languages}")
        
        print(f"\nStarting systematic SAE steering analysis...")
        print(f"Total combinations: {total_combinations}")
        print(f"Steering strength: {args.steering_strength}")
        print(f"Number of SAE layers: {len(saes)}")
        print(f"SAE layer indices: {sae_layer_indices}\n")
        
        for source_lang in languages:
            for target_lang in languages:
                counter += 1

                if not verify_model_reset(model):
                    print(f"ERROR: Model contaminated before {source_lang} → {target_lang}")
                    # Force cleanup
                    try:
                        layers = get_model_layers(model)
                        for layer in layers:
                            if hasattr(layer.input_layernorm, '_original_forward'):
                                layer.input_layernorm.forward = layer.input_layernorm._original_forward
                                delattr(layer.input_layernorm, '_original_forward')
                    except Exception as e:
                        print(f"Cleanup error: {e}")
                
                if args.skip_existing and check_experiment_exists(
                    source_lang, target_lang, args.output, args.steering_strength, layer_range
                ):
                    skipped += 1
                    print(f"[{counter}/{total_combinations}] Skipping existing: {source_lang} → {target_lang}")
                    continue
                
                print(f"\n[{counter}/{total_combinations}] Processing... (Skipped: {skipped})")
                
                try:
                    results = run_sae_steering_experiment(
                        model, sampling_params, test_questions,
                        source_lang, target_lang,
                        steering_data_sae, saes, sae_layer_indices, args.steering_strength,
                        layer_range
                    )
                    if results is not None:
                        save_results(results, args.output, args.steering_strength, layer_range)
                    
                except Exception as e:
                    print(f"Error in experiment {source_lang} → {target_lang}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n{'='*70}")
        print(f"Batch processing complete! Skipped {skipped} existing experiments.")
        print(f"{'='*70}")
    
    else:
        # Single experiment mode
        print(f"\nRunning single experiment mode...")
        print(f"Source: {args.source_lang} → Target: {args.target_lang}\n")
        
        results = run_sae_steering_experiment(
            model, sampling_params, test_questions,
            args.source_lang, args.target_lang,
            steering_data_sae, saes, sae_layer_indices, args.steering_strength,
            layer_range
        )
        if results is not None:
            save_results(results, args.output, args.steering_strength, layer_range)
            print("\nSingle experiment complete!")
        else:
            print("\nExperiment failed - no results to save")

if __name__ == "__main__":
    main()