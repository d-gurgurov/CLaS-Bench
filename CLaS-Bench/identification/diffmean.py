import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="en")
parser.add_argument("-s", "--save", type=str, default="llama_3-1")
args = parser.parse_args()

save = args.save.split(" ")

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
hidden_size = model.llm_engine.model_config.hf_config.hidden_size
max_length = model.llm_engine.model_config.max_model_len

# Store accumulated activations and token counts per layer
activation_sums = torch.zeros(num_layers, hidden_size, dtype=torch.float32).to('cuda')
token_counts = torch.zeros(num_layers, dtype=torch.int64).to('cuda')

def load_tokenized_data(filepath, target_tokens=None):
    """Load pre-tokenized data from torch file."""
    print(f"Loading tokenized data from {filepath}...")
    token_ids = torch.load(filepath)
    if target_tokens and len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
    print(f"Loaded {len(token_ids):,} tokens")
    return token_ids, len(token_ids)

def factory(idx):
    """Hook to capture residual stream activations after all layer processing."""
    def forward_hook(self, positions, hidden_states, residual):
        output_hidden_states, output_residual = self._original_forward(
            positions, hidden_states, residual
        )
        
        # Choose what to capture based on model architecture
        if "llama" in args.model.lower():
            # Llama: capture the true residual stream, which is the sum of output_hidden_states and output_residual 
            # the forward originally returns them separately and they are summed on the next layer in the normalization component for efficiency
            final_output = (output_hidden_states + output_residual).float()
            # print("mlp")
            # print(output_hidden_states)
            # print("res")
            # print(output_residual)
            # print("final")
            # print(final_output)

        elif "aya" in args.model.lower():
            # Aya: capture the final output, which is already the sum of output_hidden_states and output_residual
            final_output = output_hidden_states
        else:
            # Default to residual stream (can change as needed)
            final_output = (output_hidden_states + output_residual).float()
        
        # Accumulate activations
        if final_output.dim() == 3:
            # Shape: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, _ = final_output.shape
            activation_sums[idx, :] += final_output.float().sum(dim=(0, 1))
            token_counts[idx] += batch_size * seq_len
        elif final_output.dim() == 2:
            # Shape: [seq_len, hidden_size]
            activation_sums[idx, :] += final_output.float().sum(dim=0)
            token_counts[idx] += final_output.size(0)
        
        return output_hidden_states, output_residual
    
    return forward_hook



# Hook into decoder layers
for i in range(num_layers):
    if "gemma-3" in str(args.model).lower():
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers[i]
    else:
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i]
    
    # Store original forward method
    layer._original_forward = layer.forward
    layer.forward = MethodType(factory(i), layer)

# Load and prepare data using raw token IDs
target_tokens = 10_000_000 # Why: Just averaging activations is robust. Even with limited data, the mean is stable
token_ids, actual_tokens = load_tokenized_data(
    f"data_{save[0]}/culturax_{args.lang}.pt",
    target_tokens=target_tokens
)

# Reshape token IDs directly without decoding (matching second script approach)
l = len(token_ids)
l = min(l, 99999744) // max_length * max_length
input_ids = token_ids[:l].reshape(-1, max_length)

print(f"Processing {input_ids.size(0)} sequences of length {max_length}...")
sampling_params = SamplingParams(max_tokens=1, temperature=0)

# Generate with prompt_token_ids (matching second script)
output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)

print("Computing steering vectors...")

# Compute average steering vector for each layer
steering_vectors = activation_sums / token_counts.unsqueeze(1).float()

output = dict(
    steering_vectors=steering_vectors.to('cpu'),  # [num_layers, hidden_size]
    token_counts=token_counts.to('cpu'),
    num_tokens=l
)

os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/vector.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved steering vectors to {output_path}")
print(f"Shape: {steering_vectors.shape}")