import argparse
from types import MethodType

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="en")
parser.add_argument("-s", "--save", type=str, default="llama")
args = parser.parse_args()

import os
save = args.save.split(" ")
lang = args.lang

output_dir = f"data_{save[0]}"
output_file = f"{output_dir}/activation.{lang}.{save[1]}"

# Check if the file already exists
if os.path.exists(output_file):
    print(f"Activation file already exists for language '{lang}', skipping...")
    exit(0)
else:
    os.makedirs(output_dir, exist_ok=True)

# Detect model type
model_name = args.model.lower()
is_gemma2 = "gemma-2" in model_name or "gemma2" in model_name
is_gemma3 = "gemma-3" in model_name or "gemma3" in model_name
is_gemma = "gemma" in model_name
is_llama = "llama" in model_name
is_aya = "aya" in model_name

print(f"Model: {args.model}")
print(f"Detected architecture: {'Gemma-3' if is_gemma3 else 'Gemma-2' if is_gemma2 else 'Gemma' if is_gemma else 'Llama' if is_llama else 'Other'}")

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

max_length = model.llm_engine.model_config.max_model_len
if is_gemma3:
    num_layers = model.llm_engine.model_config.hf_config.text_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.text_config.intermediate_size
else:
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size

# For models with fused gate_up_proj, we track the full intermediate size
# The activation (gate) part is the first half
activation_size = intermediate_size

print(f"Number of layers: {num_layers}")
print(f"Intermediate size: {intermediate_size}")
print(f"Activation size (gate): {activation_size}")

over_zero = torch.zeros(num_layers, activation_size, dtype=torch.int32).to('cuda')
activation_sums = torch.zeros(num_layers, activation_size, dtype=torch.float32).to('cuda')
activation_counts = torch.zeros(num_layers, activation_size, dtype=torch.int32).to('cuda')


def factory_llama(idx):
    """Factory for Llama-style models using SiLU activation"""
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            activated = F.silu(gate_up[:, :, : i // 2])
            activation = activated.float()
            
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            activation_sums[idx, :] += activation.sum(dim=(0, 1))
            activation_counts[idx, :] += activation.size(0) * activation.size(1)
            
            x = activated * gate_up[:, :, i // 2:]
            
        elif gate_up.dim() == 2:
            activated = F.silu(gate_up[:, : i // 2])
            activation = activated.float()
            
            over_zero[idx, :] += (activation > 0).sum(dim=0)
            activation_sums[idx, :] += activation.sum(dim=0)
            activation_counts[idx, :] += activation.size(0)
            
            x = activated * gate_up[:, i // 2:]
        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")
        
        x, _ = self.down_proj(x)
        return x

    return llama_forward


def factory_gemma2(idx):
    """Factory for Gemma-2 models using GELU activation (tanh approximation)"""
    def gemma2_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            # Gemma-2 uses GELU with tanh approximation
            activated = F.gelu(gate_up[:, :, : i // 2], approximate='tanh')
            activation = activated.float()
            
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            activation_sums[idx, :] += activation.sum(dim=(0, 1))
            activation_counts[idx, :] += activation.size(0) * activation.size(1)
            
            x = activated * gate_up[:, :, i // 2:]
            
        elif gate_up.dim() == 2:
            activated = F.gelu(gate_up[:, : i // 2], approximate='tanh')
            activation = activated.float()
            
            over_zero[idx, :] += (activation > 0).sum(dim=0)
            activation_sums[idx, :] += activation.sum(dim=0)
            activation_counts[idx, :] += activation.size(0)
            
            x = activated * gate_up[:, i // 2:]
        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")
        
        x, _ = self.down_proj(x)
        return x

    return gemma2_forward


# Select the appropriate factory based on model type
if is_gemma2 or is_gemma:  # Gemma-2 and Gemma-1 both use GELU
    factory_fn = factory_gemma2
    print("Using GELU activation (Gemma-style)")
elif is_gemma3:
    # Gemma-3 might have different architecture - check documentation
    factory_fn = factory_gemma2  # Likely also GELU, but verify
    print("Using GELU activation (Gemma-3 style)")
else:
    factory_fn = factory_llama
    print("Using SiLU activation (Llama-style)")


# Hook into MLP layers
for i in range(num_layers):
    if is_gemma3:
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers[i].mlp
    else:
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    
    obj.forward = MethodType(factory_fn(i), obj)


# Load pre-tokenized data
ids = torch.load(f'data_{save[0]}/culturax_{lang}.pt')

l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

actual_tokens = l

print(f"Processing {input_ids.size(0)} sequences of length {max_length}...")
sampling_params = SamplingParams(max_tokens=1, temperature=0)

# Generate with prompt_token_ids
output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)

print("Generation complete, calculating averages...")

average_activations = activation_sums / activation_counts.float()

output = dict(
    n=actual_tokens, 
    over_zero=over_zero.to('cpu'),
    average_activations=average_activations.to('cpu'),
    activation_counts=activation_counts.to('cpu'),
    model_type="gemma2" if (is_gemma2 or is_gemma) else "gemma3" if is_gemma3 else "llama"  # Save model type for reference
)

torch.save(output, output_file)
print(f"Saved to {output_file}")