import argparse
import json
import os
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from utils.utils import get_test_questions

def compute_average_activations(activations_path):
    """Compute average activation values for each language"""
    n, average_activations = [], []
    lang_names = [
        "en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt",
        "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk",
        "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"
    ]
    for lang in lang_names:
        data = torch.load(f'identification/data_{activations_path[0]}/activation.{lang}.{activations_path[1]}')
        n.append(data['n'])
        average_activations.append(data['average_activations'])

    n = torch.tensor(n)
    average_activations = torch.stack(average_activations, dim=-1)  # layer x inter x lang_num
    
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return average_activations, lang_to_idx, lang_names

def compute_diffmean_values(avg_activations, layer_idx, activate_indices_cpu, activate_idx):
    """Compute difference between target language mean and mean of all other languages"""
    if len(activate_indices_cpu) == 0:
        return torch.tensor([]).to('cuda')
    
    target_activations = avg_activations[layer_idx, activate_indices_cpu, activate_idx]
    
    other_langs_activations = avg_activations[layer_idx, activate_indices_cpu, :]
    other_langs_activations = torch.cat([
        other_langs_activations[:, :activate_idx], 
        other_langs_activations[:, activate_idx+1:]
    ], dim=1)
    
    other_langs_mean = other_langs_activations.mean(dim=1)
    diffmean_values = target_activations - other_langs_mean
    
    return diffmean_values.to('cuda')


def factory_llama(layer_idx, deactivate_indices=None, activate_indices=None, boost_values=None, activation_method="additive", deactivation_strength=0.0, activation_strength=1.0):
    """Factory for Llama-style models using SiLU activation"""
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            activated = F.silu(gate_up[:, :, : i // 2])
            
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                activated.index_fill_(2, deactivate_indices, deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(activated.dtype).unsqueeze(0).unsqueeze(0)
                
                if activation_method == "additive":
                    activated[:, :, activate_indices] += activation_strength * boost_tensor
                elif activation_method == "replacement":
                    activated[:, :, activate_indices] = activation_strength * boost_tensor
                elif activation_method == "diffmean":
                    activated[:, :, activate_indices] += activation_strength * boost_tensor
                else:
                    raise ValueError(f"Unknown activation method: {activation_method}")

            x = activated * gate_up[:, :, i // 2:]

        elif gate_up.dim() == 2:
            activated = F.silu(gate_up[:, : i // 2])

            if deactivate_indices is not None and len(deactivate_indices) > 0:
                activated.index_fill_(1, deactivate_indices, deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(activated.dtype).unsqueeze(0)
                
                if activation_method == "additive":
                    activated[:, activate_indices] += activation_strength * boost_tensor
                elif activation_method == "replacement":
                    activated[:, activate_indices] = activation_strength * boost_tensor
                elif activation_method == "diffmean":
                    activated[:, activate_indices] += activation_strength * boost_tensor
                else:
                    raise ValueError(f"Unknown activation method: {activation_method}")

            x = activated * gate_up[:, i // 2:]

        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        x, _ = self.down_proj(x)
        return x

    return llama_forward


def factory_gemma2(layer_idx, deactivate_indices=None, activate_indices=None, boost_values=None, activation_method="additive", deactivation_strength=0.0, activation_strength=1.0):
    """Factory for Gemma-2 models using GELU activation (tanh approximation)"""
    def gemma2_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            # Gemma-2 uses GELU with tanh approximation
            activated = F.gelu(gate_up[:, :, : i // 2], approximate='tanh')
            
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                activated.index_fill_(2, deactivate_indices, deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(activated.dtype).unsqueeze(0).unsqueeze(0)
                
                if activation_method == "additive":
                    activated[:, :, activate_indices] += activation_strength * boost_tensor
                elif activation_method == "replacement":
                    activated[:, :, activate_indices] = activation_strength * boost_tensor
                elif activation_method == "diffmean":
                    activated[:, :, activate_indices] += activation_strength * boost_tensor
                else:
                    raise ValueError(f"Unknown activation method: {activation_method}")

            x = activated * gate_up[:, :, i // 2:]

        elif gate_up.dim() == 2:
            activated = F.gelu(gate_up[:, : i // 2], approximate='tanh')

            if deactivate_indices is not None and len(deactivate_indices) > 0:
                activated.index_fill_(1, deactivate_indices, deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(activated.dtype).unsqueeze(0)
                
                if activation_method == "additive":
                    activated[:, activate_indices] += activation_strength * boost_tensor
                elif activation_method == "replacement":
                    activated[:, activate_indices] = activation_strength * boost_tensor
                elif activation_method == "diffmean":
                    activated[:, activate_indices] += activation_strength * boost_tensor
                else:
                    raise ValueError(f"Unknown activation method: {activation_method}")

            x = activated * gate_up[:, i // 2:]

        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        x, _ = self.down_proj(x)
        return x

    return gemma2_forward


def detect_model_type(model_name):
    """Detect model architecture type from model name"""
    model_name_lower = model_name.lower()
    
    if "gemma-2" in model_name_lower or "gemma2" in model_name_lower:
        return "gemma2"
    elif "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
        return "gemma3"
    elif "gemma" in model_name_lower:
        # Gemma-1 also uses GELU
        return "gemma"
    elif "llama" in model_name_lower:
        return "llama"
    elif "mistral" in model_name_lower:
        return "llama"  # Mistral uses same architecture as Llama
    elif "aya" in model_name_lower:
        return "llama"  # Aya uses Llama-style architecture
    else:
        # Default to Llama-style
        return "llama"


def get_factory_for_model(model_type):
    """Get the appropriate factory function for the model type"""
    if model_type in ["gemma2", "gemma3", "gemma"]:
        return factory_gemma2
    else:
        return factory_llama


def get_activation_name(model_type):
    """Get human-readable activation function name"""
    if model_type in ["gemma2", "gemma3", "gemma"]:
        return "GELU (tanh)"
    else:
        return "SiLU"


def get_model_layers(model, model_type="llama"):
    """Get model layers with compatibility for different vLLM versions and model types"""
    try:
        if model_type == "gemma3":
            # Gemma-3 has different structure
            return model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers
        else:
            # Try newer vLLM structure
            return model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    except AttributeError:
        try:
            # Try alternative structure
            return model.llm_engine.driver_worker.model_runner.model.model.layers
        except AttributeError:
            try:
                # Try direct access
                return model.model.model.layers
            except AttributeError:
                raise RuntimeError("Could not access model layers. vLLM API may have changed.")


def apply_intervention(model, activation_masks, avg_activations, lang_to_idx, deactivate_lang, activate_lang, activation_method="additive", model_type="llama", deactivation_strength=0.0, activation_strength=1.0):
    """Apply language intervention to the model"""
    
    if deactivate_lang is not None and deactivate_lang.lower() != "none":
        if deactivate_lang not in lang_to_idx:
            raise ValueError(f"Deactivate language '{deactivate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
        deactivate_idx = lang_to_idx[deactivate_lang]
        deactivate_mask = activation_masks[deactivate_idx]
        print(f"Deactivating {deactivate_lang} neurons (index {deactivate_idx})")
    else:
        deactivate_mask = None
        print("Deactivation disabled")

    if activate_lang not in lang_to_idx:
        raise ValueError(f"Activate language '{activate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
    activate_idx = lang_to_idx[activate_lang]
    activate_mask = activation_masks[activate_idx]
    print(f"Activating {activate_lang} neurons (index {activate_idx}) using method: {activation_method}")

    # Get the appropriate factory function
    factory_fn = get_factory_for_model(model_type)
    activation_name = get_activation_name(model_type)
    print(f"Using {activation_name} activation ({model_type} architecture)")

    # Get model layers with compatibility check
    layers = get_model_layers(model, model_type)
    
    original_forwards = []
    
    for layer_idx in range(len(activate_mask)):
        if deactivate_mask is not None:
            deactivate_indices_cpu = deactivate_mask[layer_idx]
            deactivate_indices = deactivate_indices_cpu.to('cuda')
        else:
            deactivate_indices = None
        
        activate_indices_cpu = activate_mask[layer_idx]
        activate_indices = activate_indices_cpu.to('cuda')
        
        if len(activate_indices_cpu) > 0:
            if activation_method == "diffmean":
                boost_values = compute_diffmean_values(avg_activations, layer_idx, activate_indices_cpu, activate_idx)
            else:
                boost_values = avg_activations[layer_idx, activate_indices_cpu, activate_idx].to('cuda')
        else:
            boost_values = torch.tensor([]).to('cuda')
        
        obj = layers[layer_idx].mlp
        
        original_forwards.append(obj.forward)
        
        # Use the appropriate factory with all parameters
        obj.forward = MethodType(
            factory_fn(
                layer_idx, 
                deactivate_indices, 
                activate_indices, 
                boost_values, 
                activation_method,
                deactivation_strength,
                activation_strength
            ), 
            obj
        )
    
    return original_forwards


def restore_original_forwards(model, original_forwards, model_type="llama"):
    """Restore original forward methods"""
    layers = get_model_layers(model, model_type)
    
    for layer_idx, original_forward in enumerate(original_forwards):
        obj = layers[layer_idx].mlp
        obj.forward = original_forward


def create_chat_messages(prompts):
    """Create simple chat messages for steering experiments"""
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def run_single_experiment(model, sampling_params, test_questions, deactivate_lang, activate_lang, activation_masks, avg_activations, lang_to_idx, deactivation=True, activation_method="additive", model_type="llama", deactivation_strength=0.0, activation_strength=1.0, output_dir="results", no_deactivation=False, model_name=""):
    """Run a single language intervention experiment with batched inference"""
    
    print(f"\n{'='*60}")
    if not deactivation:
        print(f"Experiment: Activation only → {activate_lang} (method: {activation_method})")
    else:
        print(f"Experiment: {deactivate_lang} → {activate_lang} (method: {activation_method})")
    print(f"{'='*60}")
    
    if not deactivation:
        original_forwards = apply_intervention(
            model, activation_masks, avg_activations, lang_to_idx, 
            "none", activate_lang, activation_method, model_type,
            deactivation_strength, activation_strength
        )
    else:
        original_forwards = apply_intervention(
            model, activation_masks, avg_activations, lang_to_idx, 
            deactivate_lang, activate_lang, activation_method, model_type,
            deactivation_strength, activation_strength
        )

    test_prompts = test_questions[deactivate_lang]
    
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
    
    restore_original_forwards(model, original_forwards, model_type)
    
    results = {
        "deactivate_language": deactivate_lang if deactivate_lang and deactivate_lang.lower() != "none" else None,
        "activate_language": activate_lang,
        "activation_method": activation_method,
        "model": model_name,
        "model_type": model_type,
        "activation_function": get_activation_name(model_type),
        "deactivation_strength": deactivation_strength,
        "activation_strength": activation_strength,
        "results": all_results
    }
    
    os.makedirs(output_dir, exist_ok=True)

    if no_deactivation:
        output_subdir = os.path.join(output_dir, f"activate_{activation_method}")
    else:
        output_subdir = os.path.join(output_dir, f"deactivate_activate_{activation_method}")
    
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"{deactivate_lang}_to_{activate_lang}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to: {output_file}")
    
    return results


def check_experiment_exists(deactivate_lang, activate_lang, output_dir, activation_method, no_deactivation):
    """Check if experiment results already exist"""
    if no_deactivation:
        output_subdir = os.path.join(output_dir, f"activate_{activation_method}")
    else:
        output_subdir = os.path.join(output_dir, f"deactivate_activate_{activation_method}")
    
    output_file = os.path.join(output_subdir, f"{deactivate_lang}_to_{activate_lang}.json")
    return os.path.exists(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("-a", "--activation_mask", type=str, default="identification/activation_mask/llama-3")
    parser.add_argument("--activations_path", type=str, default="identification/data_llama-3")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--deactivate_lang", type=str, default="de", help="Language to deactivate (single mode)")
    parser.add_argument("--activate_lang", type=str, default="ru", help="Language to activate (single mode)")
    parser.add_argument("--no_deactivation", action='store_true', help="Skip deactivation, only do activation")
    parser.add_argument("--languages", nargs='+', default=["en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk", "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"])
    parser.add_argument("--deactivation_strength", type=float, default=0.0)
    parser.add_argument("--activation_strength", type=float, default=1.0)
    parser.add_argument("--activation_method", type=str, default="additive", choices=["additive", "replacement", "diffmean"])
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)

    args = parser.parse_args()

    activations_path = args.activations_path.split(" ")
    
    # Detect model type
    model_type = detect_model_type(args.model)
    
    print("="*60)
    print("Loading model and data...")
    print(f"Model: {args.model}")
    print(f"Detected architecture: {model_type}")
    print(f"Activation function: {get_activation_name(model_type)}")
    print(f"Using activation method: {args.activation_method}")
    print("="*60)

    test_questions = get_test_questions(k=70)
    
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
    
    # Print model info for verification
    try:
        if model_type == "gemma3":
            num_layers = model.llm_engine.model_config.hf_config.text_config.num_hidden_layers
            intermediate_size = model.llm_engine.model_config.hf_config.text_config.intermediate_size
        else:
            num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
            intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
        print(f"Model layers: {num_layers}")
        print(f"Intermediate size: {intermediate_size}")
    except Exception as e:
        print(f"Could not retrieve model config: {e}")
    
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
    
    activation_masks = torch.load(args.activation_mask)
    avg_activations, lang_to_idx, lang_n = compute_average_activations(activations_path)
    
    # Verify mask dimensions match model
    print(f"Activation mask languages: {len(activation_masks)}")
    print(f"Activation mask layers: {len(activation_masks[0])}")
    
    if args.batch_mode:
        languages = args.languages
        total_combinations = len(languages) * len(languages)
        counter = 0
        skipped = 0
        
        print(f"Starting systematic analysis...")
        print(f"Total combinations: {total_combinations}")
        
        all_experiment_results = []
        
        for deactivate_lang in languages:
            for activate_lang in languages:
                counter += 1
                
                if check_experiment_exists(deactivate_lang, activate_lang, args.output, args.activation_method, args.no_deactivation):
                    skipped += 1
                    print(f"[{counter}/{total_combinations}] Skipping existing: {deactivate_lang} → {activate_lang}")
                    continue
                
                print(f"\n[{counter}/{total_combinations}] Processing... (Skipped: {skipped})")
                
                try:
                    results = run_single_experiment(
                        model, sampling_params, test_questions, 
                        deactivate_lang, activate_lang, 
                        activation_masks, avg_activations, lang_to_idx, 
                        deactivation=not args.no_deactivation,
                        activation_method=args.activation_method,
                        model_type=model_type,
                        deactivation_strength=args.deactivation_strength,
                        activation_strength=args.activation_strength,
                        output_dir=args.output,
                        no_deactivation=args.no_deactivation,
                        model_name=args.model
                    )
                    all_experiment_results.append(results)
                    
                except Exception as e:
                    print(f"Error in experiment {deactivate_lang} → {activate_lang}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"\nBatch processing complete! Skipped {skipped} existing experiments.")

    else:
        results = run_single_experiment(
            model, sampling_params, test_questions, 
            args.deactivate_lang, args.activate_lang, 
            activation_masks, avg_activations, lang_to_idx,
            deactivation=not args.no_deactivation,
            activation_method=args.activation_method,
            model_type=model_type,
            deactivation_strength=args.deactivation_strength,
            activation_strength=args.activation_strength,
            output_dir=args.output,
            no_deactivation=args.no_deactivation,
            model_name=args.model
        )
        print("Single experiment complete!")


if __name__ == "__main__":
    main()