import argparse
import json
import os
from types import MethodType
import torch
from vllm import LLM, SamplingParams
from utils.utils import get_test_questions

def factory(layer_idx, steering_diff, steering_strength):
    """Hook to add LDA steering vector difference to residual stream"""
    def forward_hook(self, positions, hidden_states, residual):
        # Call original forward to get output
        output_hidden_states, output_residual = self._original_forward(positions, hidden_states, residual)
        
        # Get the final output (what gets passed to next layer)
        final_output = output_hidden_states
        
        # Add LDA steering vector difference scaled by strength
        steering_vector = steering_diff.to(final_output.device).to(final_output.dtype)
        
        steering_vector = steering_vector / (steering_vector.norm(p=2) + 1e-8)

        if final_output.dim() == 2:
            # Shape: [batch_size * seq_len, hidden_size]
            final_output = final_output + steering_strength * steering_vector.unsqueeze(0)
        elif final_output.dim() == 3:
            # Shape: [batch_size, seq_len, hidden_size]
            final_output = final_output + steering_strength * steering_vector.unsqueeze(0).unsqueeze(0)
        
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

def load_pca_vectors(pca_dir, languages, save_name):
    """Load LDA vectors for specified languages
    
    Args:
        pca_dir: Directory containing LDA vector files
        languages: List of language codes (FLORES or 2-letter)
        save_name: Save name (e.g., 'llama_3-1')
    
    Returns:
        Dictionary mapping language codes to LDA steering vectors
    """
    pca_data = {}
    
    for lang in languages:
        # Use the language code
        filepath = os.path.join(pca_dir, f"lda.{lang}.{save_name}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LDA vector file not found for '{lang}': looked for {filepath}")
        
        data = torch.load(filepath, map_location='cpu')
        pca_data[lang] = data['steering_vectors']  # Shape: [num_layers, hidden_size]
        print(f"Loaded LDA vectors for {lang}: {data['steering_vectors'].shape}")
    
    return pca_data

def apply_pca_steering(model, steering_diff, steering_strength, layer_range=None):
    """Apply LDA steering vectors to model layers"""
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
        
        # Apply LDA steering hook
        layer.forward = MethodType(
            factory(layer_idx, steering_diff[layer_idx], steering_strength), 
            layer
        )
    
    print(f"Applied LDA steering to layers {min(layer_range)} to {max(layer_range)}")
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
    """Create simple chat messages for steering experiments"""
    return [[{"role": "user", "content": prompt}] for prompt in prompts]

def run_pca_steering_experiment(model, sampling_params, test_questions, source_lang, target_lang, 
                                pca_data, steering_strength, layer_range=None):
    """Run a single LDA steering experiment"""
    
    print(f"\n{'='*60}")
    print(f"LDA Steering Experiment: {source_lang} → {target_lang}")
    print(f"Steering strength: {steering_strength}")
    print(f"{'='*60}")
    
    # Use the LDA vector directly for target language
    steering_vector = pca_data[target_lang]
    print(f"Steering vector shape: {steering_vector.shape}")
    
    # Apply LDA steering
    original_forwards = apply_pca_steering(model, steering_vector, steering_strength, layer_range)
    
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
        "steering_strength": steering_strength,
        "layer_range": f"{min(layer_range) if layer_range else 0}-{max(layer_range) if layer_range else 'all'}",
        "model": args.model,
        "results": all_results
    }
    
    return results

def check_experiment_exists(source_lang, target_lang, output_dir, steering_strength):
    """Check if experiment results already exist"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    return os.path.exists(output_file)

def save_results(results, output_dir, steering_strength):
    """Save experiment results to JSON file"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    os.makedirs(output_subdir, exist_ok=True)
    
    source_lang = results["source_language"]
    target_lang = results["target_language"]
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Apply LDA steering vectors to language models")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lda_dir", type=str, default="data_llama_3-1", 
                       help="Directory containing LDA vector files")
    parser.add_argument("-s", "--save_name", type=str, default="llama",
                       help="Save name for LDA files (e.g., 'llama_3-1')")
    parser.add_argument("--output", type=str, default="pca_steering_results")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--source_lang", type=str, default="eng_Latn", help="Source language (language code)")
    parser.add_argument("--target_lang", type=str, default="fra_Latn", help="Target language (language code)")
    parser.add_argument("--languages", nargs='+', 
                       default=["en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk", "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"],
                       help="Languages to test (language codes)")
    parser.add_argument("--steering_strength", type=float, default=1.0, 
                       help="Scaling factor for LDA steering vector (alpha)")
    parser.add_argument("--layer_start", type=int, default=None, 
                       help="First layer to apply steering (default: all layers)")
    parser.add_argument("--layer_end", type=int, default=None, 
                       help="Last layer to apply steering (default: all layers)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--skip_existing", action='store_true', 
                       help="Skip experiments that already have results")

    global args
    args = parser.parse_args()
    
    print("Loading model and LDA vectors...")
    
    # Load LDA vectors
    languages_to_load = args.languages
    print(f"Loading LDA vectors for: {languages_to_load}")
    
    pca_data = load_pca_vectors(args.lda_dir, languages_to_load, args.save_name)
    
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
        # Get number of layers from first LDA data
        first_lang = list(pca_data.keys())[0]
        num_layers = pca_data[first_lang].shape[0]
        end = args.layer_end if args.layer_end is not None else num_layers
        layer_range = (start, end)
        print(f"Applying steering to layers {start} to {end-1}")
    else:
        print("Applying steering to all layers")
    
    if args.batch_mode:
        languages = args.languages
        total_combinations = len(languages) * len(languages)
        counter = 0
        skipped = 0
        
        print(f"\nStarting systematic LDA steering analysis...")
        print(f"Total combinations: {total_combinations}")
        print(f"Steering strength: {args.steering_strength}")
        
        for source_lang in languages:
            for target_lang in languages:
                counter += 1
                
                if args.skip_existing and check_experiment_exists(
                    source_lang, target_lang, args.output, args.steering_strength
                ):
                    skipped += 1
                    print(f"[{counter}/{total_combinations}] Skipping existing: {source_lang} → {target_lang}")
                    continue
                
                print(f"\n[{counter}/{total_combinations}] Processing... (Skipped: {skipped})")
                
                try:
                    results = run_pca_steering_experiment(
                        model, sampling_params, test_questions,
                        source_lang, target_lang,
                        pca_data, args.steering_strength,
                        layer_range
                    )
                    if results is not None:
                        save_results(results, args.output, args.steering_strength)
                    
                except Exception as e:
                    print(f"Error in experiment {source_lang} → {target_lang}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\nBatch processing complete! Skipped {skipped} existing experiments.")
    
    else:
        # Single experiment mode
        results = run_pca_steering_experiment(
            model, sampling_params, test_questions,
            args.source_lang, args.target_lang,
            pca_data, args.steering_strength,
            layer_range
        )
        if results is not None:
            save_results(results, args.output, args.steering_strength)
            print("Single experiment complete!")
        else:
            print("Experiment failed - no results to save")

if __name__ == "__main__":
    main()