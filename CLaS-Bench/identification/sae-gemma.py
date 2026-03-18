import argparse
from types import MethodType

import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="google/gemma-2-9b-it")
parser.add_argument("-l", "--lang", type=str, default="en")
parser.add_argument("-s", "--save", type=str, default="gemma_2-9b")
parser.add_argument("--sae-repo", type=str, default="google/gemma-scope-9b-it-res")
parser.add_argument("--sae-width", type=str, default="16k", choices=["16k", "131k"])
args = parser.parse_args()

save = args.save.split(" ")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
hidden_size = model.llm_engine.model_config.hf_config.hidden_size
max_length = model.llm_engine.model_config.max_model_len

# SAE configuration - Gemma Scope 9B IT only has SAEs for layers 9, 20, 31
# Each layer has different L0 values available
# Format: {layer_idx: {width: [available_l0_values]}}
# We pick the highest L0 (least sparse) for each layer by default
SAE_L0_CONFIG = {
    "google/gemma-scope-9b-it-res": {
        9: {"131k": ["22"], "16k": ["186"]},
        20: {"131k": ["43", "81"], "16k": ["189"]},
        31: {"131k": ["37", "63"], "16k": ["142"]},
    },
}

def get_l0_for_layer(repo_id, layer_idx, width):
    """Get the appropriate L0 value for a given layer and width.
    Returns the highest L0 (closest to canonical ~100) available."""
    if repo_id not in SAE_L0_CONFIG:
        raise ValueError(f"Unknown SAE repo: {repo_id}. Available: {list(SAE_L0_CONFIG.keys())}")
    
    layer_config = SAE_L0_CONFIG[repo_id].get(layer_idx)
    if layer_config is None:
        raise ValueError(f"Layer {layer_idx} not available for {repo_id}")
    
    width_config = layer_config.get(width)
    if width_config is None:
        raise ValueError(f"Width {width} not available for layer {layer_idx} in {repo_id}")
    
    # Return the highest L0 value (last in the sorted list)
    return max(width_config, key=lambda x: int(x))

sae_layer_indices = list(SAE_L0_CONFIG.get(args.sae_repo, SAE_L0_CONFIG["google/gemma-scope-9b-it-res"]).keys())

# Placeholder for activation arrays (will be initialized after loading SAE)
activation_sums = None
token_counts = None
sae_dict_size = None

# ---------- SAE Loading and Inference ----------
class JumpReLUSAE(torch.nn.Module):
    """JumpReLU SAE for Gemma Scope (uses learned threshold per feature)"""
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        # Gemma Scope format: W_enc is [d_model, d_sae], W_dec is [d_sae, d_model]
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)
    
    @torch.no_grad()
    def encode(self, x):
        """Encode using JumpReLU activation function with learned thresholds"""
        # x: [batch, d_model], W_enc: [d_model, d_sae], b_enc: [d_sae]
        pre_acts = x @ self.W_enc + self.b_enc
        # JumpReLU: f(z) = z * (z > threshold), using learned per-feature thresholds
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    
    @torch.no_grad()
    def decode(self, acts):
        """Decode sparse features back to activation space"""
        # acts: [batch, d_sae], W_dec: [d_sae, d_model], b_dec: [d_model]
        return acts @ self.W_dec + self.b_dec

def load_gemma_scope_sae(repo_id, layer_idx, width, device="cuda"):
    """Load Gemma Scope SAE from HuggingFace (npz format)"""
    l0 = get_l0_for_layer(repo_id, layer_idx, width)
    filename = f"layer_{layer_idx}/width_{width}/average_l0_{l0}/params.npz"
    print(f"Downloading SAE from {repo_id}/{filename}...")
    
    hf_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision="main",
    )
    print(f"Loading SAE from {hf_path}...")
    
    # Load numpy arrays
    params = np.load(hf_path)
    
    # Convert to torch tensors
    W_enc = torch.from_numpy(params['W_enc']).to(device)
    W_dec = torch.from_numpy(params['W_dec']).to(device)
    b_enc = torch.from_numpy(params['b_enc']).to(device)
    b_dec = torch.from_numpy(params['b_dec']).to(device)
    threshold = torch.from_numpy(params['threshold']).to(device)
    
    sae = JumpReLUSAE(W_enc, W_dec, b_enc, b_dec, threshold).eval()
    
    return sae, l0

# Load SAEs for all layers
saes = {}
sae_l0_used = {}
print(f"Loading Gemma Scope SAEs for layers {sae_layer_indices}...")
for layer_idx in sae_layer_indices:
    try:
        sae, l0 = load_gemma_scope_sae(
            args.sae_repo, 
            layer_idx, 
            args.sae_width, 
            device=device
        )
        saes[layer_idx] = sae
        sae_l0_used[layer_idx] = l0
        
        # Infer sae_dict_size from first loaded SAE (W_dec shape: [d_sae, d_model])
        if sae_dict_size is None:
            sae_dict_size = sae.W_dec.shape[0]
            print(f"Inferred SAE dict size: {sae_dict_size}")
        
        print(f"Loaded SAE for layer {layer_idx} (L0={l0})")
    except Exception as e:
        print(f"Error loading SAE for layer {layer_idx}: {e}")
        continue

# Initialize activation arrays after learning dict size
num_sae_layers = len(saes)
activation_sums = torch.zeros(num_sae_layers, sae_dict_size, dtype=torch.float32).to(device)
token_counts = torch.zeros(num_sae_layers, dtype=torch.int64).to(device)
print(f"Loaded {len(saes)} SAE(s)")
print(f"L0 values used: {sae_l0_used}")

def load_tokenized_data(filepath, target_tokens=None):
    """Load pre-tokenized data from torch file."""
    print(f"Loading tokenized data from {filepath}...")
    token_ids = torch.load(filepath)
    if target_tokens and len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
    print(f"Loaded {len(token_ids):,} tokens")
    return token_ids, len(token_ids)

def factory(idx):
    """Hook to capture SAE activations after all layer processing."""
    def forward_hook(self, positions, hidden_states, residual):
        output_hidden_states, output_residual = self._original_forward(
            positions, hidden_states, residual
        )
        
        # For Gemma Scope SAEs trained on residual post, use the full residual stream
        final_output = (output_hidden_states + output_residual).float()
        
        # Pass through SAE encoder to get sparse activations
        sae = saes[idx]
        if final_output.dim() == 3:
            batch_size, seq_len, _ = final_output.shape
            # Reshape for SAE: [batch_size * seq_len, hidden_size]
            flat_output = final_output.reshape(-1, final_output.shape[-1])
            sparse_acts = sae.encode(flat_output)  # [batch_size * seq_len, sae_dict_size]
            
            # Find the position in our activation arrays based on layer index
            sae_idx = list(saes.keys()).index(idx)
            activation_sums[sae_idx, :] += sparse_acts.float().sum(dim=0)
            token_counts[sae_idx] += flat_output.size(0)
        elif final_output.dim() == 2:
            sparse_acts = sae.encode(final_output)
            sae_idx = list(saes.keys()).index(idx)
            activation_sums[sae_idx, :] += sparse_acts.float().sum(dim=0)
            token_counts[sae_idx] += final_output.size(0)
        
        return output_hidden_states, output_residual
    
    return forward_hook

# Hook into decoder layers that have SAEs
print("Hooking into decoder layers...")
for layer_idx in sae_layer_indices:
    if layer_idx not in saes:
        print(f"Skipping layer {layer_idx} (SAE not loaded)")
        continue
    
    # Gemma 2 model structure in vLLM
    layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]
    
    layer._original_forward = layer.forward
    layer.forward = MethodType(factory(layer_idx), layer)
    print(f"Hooked layer {layer_idx}")

# Load and prepare data using raw token IDs
target_tokens = 10_000_000
token_ids, actual_tokens = load_tokenized_data(
    f"data_{save[0]}/culturax_{args.lang}.pt",
    target_tokens=target_tokens
)

# Reshape token IDs directly without decoding
l = len(token_ids)
l = min(l, 99999744) // max_length * max_length
input_ids = token_ids[:l].reshape(-1, max_length)

print(f"Processing {input_ids.size(0)} sequences of length {max_length}...")
print("Collecting SAE activations...")
sampling_params = SamplingParams(max_tokens=1, temperature=0)

# Generate with prompt_token_ids
output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)

print("Computing steering vectors...")

# Compute average steering vector for each layer
steering_vectors = activation_sums / token_counts.unsqueeze(1).float()

output = dict(
    steering_vectors=steering_vectors.to('cpu'),
    token_counts=token_counts.to('cpu'),
    num_tokens=l,
    sae_dict_size=sae_dict_size,
    sae_layer_indices=list(saes.keys()),
    sae_width=args.sae_width,
    sae_l0_per_layer=sae_l0_used,
)

os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/sae-gemma-scope.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved SAE steering vectors to {output_path}")
print(f"Shape: {steering_vectors.shape}")
print(f"SAE dict size: {sae_dict_size}")
print(f"Layers extracted: {list(saes.keys())}")
print(f"L0 per layer: {sae_l0_used}")