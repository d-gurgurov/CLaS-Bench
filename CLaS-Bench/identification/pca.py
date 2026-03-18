import argparse
from types import MethodType

import torch
import numpy as np
from sklearn.decomposition import PCA
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="eng_Latn")
parser.add_argument("-s", "--save", type=str, default="llama llama")
args = parser.parse_args()

save = args.save.split(" ")

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
hidden_size = model.llm_engine.model_config.hf_config.hidden_size
max_length = model.llm_engine.model_config.max_model_len

# Store accumulated activations per layer for PCA
layer_activations = [[] for _ in range(num_layers)]

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
        elif "aya" in args.model.lower():
            # Aya: capture the final output, which is already the sum of output_hidden_states and output_residual
            final_output = output_hidden_states
        else:
            # Default to residual stream (can change as needed)
            final_output = (output_hidden_states + output_residual).float()
        
        # Store activations for PCA
        if final_output.dim() == 3:
            # Shape: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, _ = final_output.shape
            # Reshape to [batch_size * seq_len, hidden_size]
            activations = final_output.float().reshape(-1, hidden_size).detach().cpu().numpy()
            layer_activations[idx].append(activations)
        elif final_output.dim() == 2:
            # Shape: [seq_len, hidden_size]
            activations = final_output.float().detach().cpu().numpy()
            layer_activations[idx].append(activations)
        
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
target_tokens = 500_000
token_ids, actual_tokens = load_tokenized_data(
    f"data_{save[0]}/culturax_{args.lang}.pt",
    target_tokens=target_tokens
)

# Reshape token IDs directly without decoding (matching second script approach)
l = len(token_ids)
seq_length = min(max_length, 32768)
l = min(l, 99999744) // seq_length * seq_length
input_ids = token_ids[:l].reshape(-1, seq_length)

print(f"Processing {input_ids.size(0)} sequences of length {seq_length}...")
sampling_params = SamplingParams(max_tokens=1, temperature=0)

# Generate with prompt_token_ids (matching second script)
output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)

print("Computing steering vectors via PCA...")

# Compute PCA for each layer - save top 20 components
n_components = 512
pca_components = torch.zeros(num_layers, n_components, hidden_size, dtype=torch.float32)
variance_explained = torch.zeros(num_layers, n_components, dtype=torch.float32)

for layer_idx in range(num_layers):
    print(f"Computing PCA for layer {layer_idx}/{num_layers}...")
    
    # Concatenate all activations for this layer
    if layer_activations[layer_idx]:
        all_acts = np.concatenate(layer_activations[layer_idx], axis=0)
        
        # Mean center the activations
        mean = all_acts.mean(axis=0, keepdims=True)
        centered_acts = all_acts - mean
        
        # Fit PCA with top 20 components
        # n_components is limited by min of: requested components, samples, and features
        n_actual = min(n_components, centered_acts.shape[0], centered_acts.shape[1])
        pca = PCA(n_components=n_actual)
        pca.fit(centered_acts)
        
        # Extract principal components and variance explained
        pca_components[layer_idx, :n_actual] = torch.tensor(pca.components_, dtype=torch.float32)
        variance_explained[layer_idx, :n_actual] = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32)
        
        cumulative_var = variance_explained[layer_idx, :n_actual].sum().item()
        print(f"Layer {layer_idx}: {n_actual} components, cumulative variance explained = {cumulative_var:.5%}")
    else:
        print(f"Layer {layer_idx}: No activations captured")

output = dict(
    pca_components=pca_components.to('cpu'),  # [num_layers, n_components, hidden_size]
    variance_explained=variance_explained.to('cpu'),  # [num_layers, n_components]
    n_components=n_components,
    num_tokens=l
)

import os

os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/pca.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved PCA components to {output_path}")
print(f"PCA components shape: {pca_components.shape}")
print(f"Variance explained shape: {variance_explained.shape}")
print(f"Cumulative variance explained (mean across layers): {variance_explained.sum(dim=1).mean():.5%}")