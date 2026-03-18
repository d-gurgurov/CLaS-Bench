import argparse
import json
import os
from types import MethodType
import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from utils.utils import get_test_questions

def factory(layer_idx, probe, intervention_strength):
    """Hook to add probe-based intervention to residual stream"""
    def forward_hook(self, positions, hidden_states, residual):
        # Call original forward to get output
        output_hidden_states, output_residual = self._original_forward(positions, hidden_states, residual)
        
        # Get the final output (what gets passed to next layer)
        final_output = output_hidden_states
        
        if final_output.dim() == 3:
            # Shape: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = final_output.shape
            # Reshape for probe: [batch_size * seq_len, hidden_size]
            flat_output = final_output.reshape(-1, hidden_size)
        elif final_output.dim() == 2:
            # Shape: [batch_size * seq_len, hidden_size]
            flat_output = final_output
        else:
            return output_hidden_states, output_residual
        
        # Get probe weight (linear layer weight is [1, hidden_size])
        probe_weight = probe.weight.squeeze(0)  # [hidden_size]
        probe_weight = probe_weight.to(flat_output.device).to(flat_output.dtype)
        
        # Normalize probe weight to have unit norm
        probe_weight_normalized = probe_weight / (probe_weight.norm(p=2) + 1e-8)
        
        # Compute intervention direction (normalized probe weight)
        intervention_vector = probe_weight_normalized
        
        # Apply intervention scaled by strength
        if final_output.dim() == 3:
            # Add intervention to each position
            final_output = final_output + intervention_strength * intervention_vector.unsqueeze(0).unsqueeze(0)
        elif final_output.dim() == 2:
            # Add intervention
            final_output = final_output + intervention_strength * intervention_vector.unsqueeze(0)
        
        # Update the appropriate output
        output_hidden_states = final_output
        
        return output_hidden_states, output_residual
    
    return forward_hook

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

def load_probes(probe_dir, languages, save_name):
    """Load probe vectors for specified languages
    
    Args:
        probe_dir: Directory containing probe files
        languages: List of 2-letter language codes
        save_name: Save name (e.g., 'llama_3-1')
    
    Returns:
        Dictionary mapping language codes to probe state dicts
    """
    probe_data = {}
    
    for lang in languages:
        # Use the 2-letter language code
        filepath = os.path.join(probe_dir, f"probe.{lang}.{save_name}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Probe file not found for '{lang}': looked for {filepath}")
        
        data = torch.load(filepath, map_location='cpu')
        probe_data[lang] = data
        print(f"Loaded probes for {lang}: {len(data['probes'])} layers")
    
    return probe_data

def create_probe_models(probe_data, device='cpu'):
    """Convert probe state dicts to nn.Module instances
    
    Args:
        probe_data: Dictionary mapping language to probe data
        device: Device to place probes on
    
    Returns:
        Dictionary mapping language to dictionary of layer probes
    """
    probe_models = {}
    
    for lang, data in probe_data.items():
        hidden_size = data['hidden_size']
        probes_by_layer = {}
        
        for layer_idx, state_dict in data['probes'].items():
            # Create linear probe model
            probe = nn.Linear(hidden_size, 1, bias=True)
            probe.load_state_dict(state_dict)
            probe = probe.to(device)
            probe.eval()
            
            probes_by_layer[layer_idx] = probe
        
        probe_models[lang] = probes_by_layer
    
    return probe_models

def apply_probe_intervention(model, probe, intervention_strength, layer_range=None):
    """Apply probe-based intervention to model layers"""
    layers = get_model_layers(model)
    num_layers = len(layers)
    
    if layer_range is None:
        layer_range = range(num_layers)
    elif isinstance(layer_range, tuple):
        layer_range = range(layer_range[0], min(layer_range[1], num_layers))
    
    original_forwards = []
    
    for layer_idx in layer_range:
        layer = layers[layer_idx]
        
        # Store original forward method
        original_forwards.append((layer_idx, layer.forward))
        layer._original_forward = layer.forward
        
        # Apply probe intervention hook
        if layer_idx in probe:
            layer.forward = MethodType(
                factory(layer_idx, probe[layer_idx], intervention_strength), 
                layer
            )
        else:
            print(f"Warning: No probe found for layer {layer_idx}")
    
    print(f"Applied probe interventions to layers {min(layer_range)} to {max(layer_range)}")
    return original_forwards

def restore_original_forwards(model, original_forwards):
    """Restore original forward methods"""
    layers = get_model_layers(model)
    
    for layer_idx, original_forward in original_forwards:
        layer = layers[layer_idx]
        layer.forward = original_forward
        if hasattr(layer, '_original_forward'):
            delattr(layer, '_original_forward')

def create_chat_messages(prompts):
    """Create simple chat messages for intervention experiments"""
    return [[{"role": "user", "content": prompt}] for prompt in prompts]

def run_probe_intervention_experiment(model, sampling_params, test_questions, source_lang, target_lang, 
                                      probe_models, intervention_strength, layer_range=None):
    """Run a single probe-based intervention experiment"""
    
    print(f"\n{'='*60}")
    print(f"Probe Intervention Experiment: {source_lang} → {target_lang}")
    print(f"Intervention strength: {intervention_strength}")
    print(f"{'='*60}")
    
    # Get target language probes (trying to steer activations towards target language)
    if target_lang not in probe_models:
        print(f"Error: No probes found for target language {target_lang}")
        return None
    
    target_probes = probe_models[target_lang]
    print(f"Using {len(target_probes)} probes for target language: {target_lang}")
    
    # Apply probe intervention
    original_forwards = apply_probe_intervention(model, target_probes, intervention_strength, layer_range)
    
    # Get test prompts for source language
    source_lang_short = source_lang
    
    if source_lang_short not in test_questions:
        print(f"Warning: No test questions found for {source_lang_short}")
        print(f"Available languages in test_questions: {list(test_questions.keys())}")
        restore_original_forwards(model, original_forwards)
        return None
    
    test_prompts = test_questions[source_lang_short]
    print(f"Using test questions for: {source_lang_short}")
    
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
    restore_original_forwards(model, original_forwards)
    
    results = {
        "source_language": source_lang,
        "source_language_short": source_lang_short,
        "target_language": target_lang,
        "intervention_strength": intervention_strength,
        "layer_range": f"{min(layer_range) if layer_range else 0}-{max(layer_range) if layer_range else 'all'}",
        "model": args.model,
        "results": all_results
    }
    
    return results

def check_experiment_exists(source_lang, target_lang, output_dir, intervention_strength):
    """Check if experiment results already exist"""
    strength_str = f"strength_{intervention_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    return os.path.exists(output_file)

def save_results(results, output_dir, intervention_strength):
    """Save experiment results to JSON file"""
    strength_str = f"strength_{intervention_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    os.makedirs(output_subdir, exist_ok=True)
    
    source_lang = results["source_language"]
    target_lang = results["target_language"]
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Apply probe-based interventions to language models")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe_dir", type=str, default="data_llama_3-1", 
                       help="Directory containing probe files")
    parser.add_argument("-s", "--save_name", type=str, default="llama",
                       help="Save name for probe files (e.g., 'llama_3-1')")
    parser.add_argument("--output", type=str, default="probe_intervention_results")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--source_lang", type=str, default="eng_Latn", help="Source language (FLORES code)")
    parser.add_argument("--target_lang", type=str, default="fra_Latn", help="Target language (FLORES code)")
    parser.add_argument("--languages", nargs='+', 
                       default=["en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk", "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"],
                       help="Languages to test (FLORES codes)")
    parser.add_argument("--intervention_strength", type=float, default=1.0, 
                       help="Scaling factor for probe intervention (alpha)")
    parser.add_argument("--layer_start", type=int, default=None, 
                       help="First layer to apply intervention (default: all layers)")
    parser.add_argument("--layer_end", type=int, default=None, 
                       help="Last layer to apply intervention (default: all layers)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--skip_existing", action='store_true', 
                       help="Skip experiments that already have results")

    global args
    args = parser.parse_args()
    
    print("Loading model and probes...")
    
    # Load probes using FLORES codes
    languages_to_load = args.languages
    print(f"Loading probes for: {languages_to_load}")
    
    probe_data = load_probes(args.probe_dir, languages_to_load, args.save_name)
    
    # Create probe models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_models = create_probe_models(probe_data, device=device)
    
    # Load test questions
    test_questions = get_test_questions(k=70)
    
    # Initialize model
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
    
    # Determine layer range
    layer_range = None
    if args.layer_start is not None or args.layer_end is not None:
        start = args.layer_start if args.layer_start is not None else 0
        # Get number of layers from first probe data
        first_lang = list(probe_data.keys())[0]
        num_layers = probe_data[first_lang]['num_layers']
        end = args.layer_end if args.layer_end is not None else num_layers
        layer_range = (start, end)
        print(f"Applying interventions to layers {start} to {end-1}")
    else:
        print("Applying interventions to all layers")
    
    if args.batch_mode:
        languages = args.languages
        total_combinations = len(languages) * len(languages)
        counter = 0
        skipped = 0
        
        print(f"\nStarting systematic probe intervention analysis...")
        print(f"Total combinations: {total_combinations}")
        print(f"Intervention strength: {args.intervention_strength}")
        
        for source_lang in languages:
            for target_lang in languages:
                counter += 1
                
                if args.skip_existing and check_experiment_exists(
                    source_lang, target_lang, args.output, args.intervention_strength
                ):
                    skipped += 1
                    print(f"[{counter}/{total_combinations}] Skipping existing: {source_lang} → {target_lang}")
                    continue
                
                print(f"\n[{counter}/{total_combinations}] Processing... (Skipped: {skipped})")
                
                try:
                    results = run_probe_intervention_experiment(
                        model, sampling_params, test_questions,
                        source_lang, target_lang,
                        probe_models, args.intervention_strength,
                        layer_range
                    )
                    if results is not None:
                        save_results(results, args.output, args.intervention_strength)
                    
                except Exception as e:
                    print(f"Error in experiment {source_lang} → {target_lang}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\nBatch processing complete! Skipped {skipped} existing experiments.")
    
    else:
        # Single experiment mode
        results = run_probe_intervention_experiment(
            model, sampling_params, test_questions,
            args.source_lang, args.target_lang,
            probe_models, args.intervention_strength,
            layer_range
        )
        if results is not None:
            save_results(results, args.output, args.intervention_strength)
            print("Single experiment complete!")
        else:
            print("Experiment failed - no results to save")

if __name__ == "__main__":
    main()