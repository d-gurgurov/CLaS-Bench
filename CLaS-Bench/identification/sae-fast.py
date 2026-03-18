import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="en")
parser.add_argument("-s", "--save", type=str, default="llama_3-1")
parser.add_argument("--sae-repo", type=str, default="Geaming/Llama-3.1-8B-Instruct_SAEs")
parser.add_argument("--sae-base-path", type=str, default="FAST/blocks_{layer}_hook_resid_post_8X_2048_jumprelu/sae_weights.safetensors")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
hidden_size = model.llm_engine.model_config.hf_config.hidden_size
max_length = model.llm_engine.model_config.max_model_len

# SAE configuration - extract activations for these layers
sae_layer_indices = [4, 12, 18, 20, 25]

# Placeholder for activation arrays (will be initialized after loading SAE)
activation_sums = None
token_counts = None
sae_dict_size = None

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

def load_sae_from_safetensors(repo_id, file_path, device="cuda"):
    """Load SAE from safetensors format (sae_lens format)"""
    print(f"Downloading SAE from {repo_id}/{file_path}...")
    hf_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        revision="main",
    )
    print(f"Loading SAE from {hf_path}...")
    state = load_file(hf_path)
    
    # sae_lens uses these weight names
    W_dec = state["W_dec"].to(device)
    W_enc = state["W_enc"].to(device)
    b_dec = state["b_dec"].to(device)
    b_enc = state["b_enc"].to(device)
    
    # Use sae_lens default JumpReLU parameters
    jumprelu_init_threshold = 0.001
    jumprelu_bandwidth = 0.001
    
    sae = JumpReLUInferenceSAE(
        W_dec, W_enc, b_dec, b_enc,
        jumprelu_init_threshold=jumprelu_init_threshold,
        jumprelu_bandwidth=jumprelu_bandwidth
    ).eval()
    
    return sae

# Load SAEs for all layers
saes = {}
print("Loading SAEs for layers...")
for layer_idx in sae_layer_indices:
    sae_path = args.sae_base_path.format(layer=layer_idx)
    try:
        sae = load_sae_from_safetensors(args.sae_repo, sae_path, device=device)
        saes[layer_idx] = sae
        
        # Infer sae_dict_size from first loaded SAE (W_dec shape: [dict_size, d_in])
        if sae_dict_size is None:
            sae_dict_size = sae.W_dec.shape[0]
            print(f"Inferred SAE dict size: {sae_dict_size}")
        
        print(f"Loaded SAE for layer {layer_idx}")
    except Exception as e:
        print(f"Error loading SAE for layer {layer_idx}: {e}")
        continue

# Initialize activation arrays after learning dict size
num_sae_layers = len(saes)
activation_sums = torch.zeros(num_sae_layers, sae_dict_size, dtype=torch.float32).to(device)
token_counts = torch.zeros(num_sae_layers, dtype=torch.int64).to(device)
print(f"Loaded {len(saes)} SAE(s)")

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
        
        # For sae_lens SAEs trained on residual post, use the full residual stream
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
    
    if "gemma-3" in str(args.model).lower():
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers[layer_idx]
    else:
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]
    
    layer._original_forward = layer.forward
    layer.forward = MethodType(factory(layer_idx), layer)
    print(f"Hooked layer {layer_idx}")

# Load and prepare data using raw token IDs
target_tokens = 10_000_000
token_ids, actual_tokens = load_tokenized_data(
    f"data_llama/culturax_{args.lang}.pt",
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
)

save = args.save.split(" ")
os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/sae-fast.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved SAE steering vectors to {output_path}")
print(f"Shape: {steering_vectors.shape}")
print(f"SAE dict size: {sae_dict_size}")
print(f"Layers extracted: {list(saes.keys())}")