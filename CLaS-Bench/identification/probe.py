import argparse
from types import MethodType
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="eng_Latn", help="Target language (positive class)")
parser.add_argument("-n", "--negative_langs", type=str, default="ru,de,zh", help="Comma-separated negative languages")
parser.add_argument("-s", "--save", type=str, default="llama-3")
parser.add_argument("--probe_batch_size", type=int, default=64, help="Batch size for probe training")
parser.add_argument("--probe_epochs", type=int, default=1, help="Number of epochs for probe training")
parser.add_argument("--probe_lr", type=float, default=1e-4, help="Learning rate for probe training")
parser.add_argument("--target_tokens", type=int, default=100_000, help="Target tokens for positive language")
parser.add_argument("--val_samples_per_class", type=int, default=1000, help="Validation samples per class")
parser.add_argument("--substitute_lang", type=str, default="fr", help="Substitute language if one of the negative languages matches positive class")
args = parser.parse_args()

save = args.save.split(" ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
hidden_size = model.llm_engine.model_config.hf_config.hidden_size
max_length = model.llm_engine.model_config.max_model_len

# Store activations per language per layer
layer_activations_by_lang = defaultdict(lambda: [[] for _ in range(num_layers)])

def load_tokenized_data(filepath, target_tokens=None):
    """Load pre-tokenized data from torch file."""
    print(f"Loading tokenized data from {filepath}...")
    token_ids = torch.load(filepath)
    if target_tokens and len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
    print(f"Loaded {len(token_ids):,} tokens")
    return token_ids, len(token_ids)

def make_forward_hook(layer_idx, current_lang):
    """Factory function to create hooks that capture activations for each language."""
    def forward_hook(self, positions, hidden_states, residual):
        output_hidden_states, output_residual = self._original_forward(
            positions, hidden_states, residual
        )
        
        # Choose what to capture based on model architecture
        if "llama" in args.model.lower():
            final_output = (output_hidden_states + output_residual).float()
        elif "aya" in args.model.lower():
            final_output = output_hidden_states
        else:
            final_output = (output_hidden_states + output_residual).float()
        
        # Store activations
        if final_output.dim() == 3:
            batch_size, seq_len, _ = final_output.shape
            activations = final_output.float().reshape(-1, hidden_size).detach().cpu().numpy()
            layer_activations_by_lang[current_lang][layer_idx].append(activations)
        elif final_output.dim() == 2:
            activations = final_output.float().detach().cpu().numpy()
            layer_activations_by_lang[current_lang][layer_idx].append(activations)
        
        return output_hidden_states, output_residual
    
    return forward_hook

def process_language(lang_code, filepath, target_tokens=None):
    """Process a single language and collect activations."""
    print(f"\n{'='*60}")
    print(f"Processing language: {lang_code}")
    print(f"{'='*60}")
    
    # Load data using raw token IDs
    token_ids, actual_tokens = load_tokenized_data(filepath, target_tokens=target_tokens)
    
    # Reshape token IDs directly without decoding (matching second script approach)
    l = len(token_ids)
    seq_length = min(max_length, 32768) # Cap at .. if tokens are limited
    l = min(l, 99999744) // seq_length * seq_length
    input_ids = token_ids[:l].reshape(-1, seq_length)
    
    print(f"Processing {input_ids.size(0)} sequences of length {seq_length}...")
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    
    # Generate with prompt_token_ids (matching second script approach)
    output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)
    
    return l

# Hook into all layers for all languages
def install_hooks(current_lang):
    """Install hooks for capturing activations."""
    for i in range(num_layers):
        if "gemma-3" in str(args.model).lower():
            layer = model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers[i]
        else:
            layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i]
        
        # Store original forward if not already stored
        if not hasattr(layer, '_original_forward'):
            layer._original_forward = layer.forward
        
        # Install new hook
        layer.forward = MethodType(make_forward_hook(i, current_lang), layer)

# Parse negative languages
negative_langs = [lang.strip() for lang in args.negative_langs.split(",")]

# Handle positive language in negative list
original_num_negative_langs = len(negative_langs)
if args.lang in negative_langs:
    print(f"Warning: Positive language {args.lang} found in negative languages")
    if args.substitute_lang:
        print(f"Substituting {args.lang} with {args.substitute_lang}")
        negative_langs[negative_langs.index(args.lang)] = args.substitute_lang
        print(f"Negative languages: {negative_langs}")
    else:
        print(f"Removing {args.lang} from negative languages")
        negative_langs.remove(args.lang)
        print(f"Negative languages reduced from {original_num_negative_langs} to {len(negative_langs)}")
        print(f"Negative languages: {negative_langs}")
else:
    print(f"Negative languages count: {original_num_negative_langs}")
    print(f"Negative languages: {negative_langs}")

# Calculate optimized token counts
# Positive language gets full target_tokens, negative languages split equally
num_negative_langs = len(negative_langs)
tokens_per_negative_lang = args.target_tokens // num_negative_langs

print(f"\n{'='*60}")
print("Token Allocation:")
print(f"{'='*60}")
print(f"Positive language ({args.lang}): {args.target_tokens:,} tokens")
print(f"Negative languages ({num_negative_langs} languages): {tokens_per_negative_lang:,} tokens each")

# ============================================================================
# ACTIVATION COLLECTION
# ============================================================================

print(f"\n{'='*60}")
print("Collecting activations...")
print(f"{'='*60}")

# Process target (positive) language
print(f"Target language (positive class): {args.lang}")
install_hooks(args.lang)
target_tokens = process_language(args.lang, f"data_{save[0]}/culturax_{args.lang}.pt", 
                                  target_tokens=args.target_tokens)

# Process negative languages
for neg_lang in negative_langs:
    install_hooks(neg_lang)
    process_language(neg_lang, f"data_{save[0]}/culturax_{neg_lang}.pt", 
                    target_tokens=tokens_per_negative_lang)

print(f"\n{'='*60}")
print("Training probes for each layer...")
print(f"{'='*60}")

# Train probes for each layer
probes = {}
probe_results = {}

for layer_idx in range(num_layers):
    print(f"\nTraining probe for layer {layer_idx}/{num_layers}...")
    
    # ========== TRAINING DATA ==========
    
    # Collect activations: positive class
    positive_acts = []
    for acts_batch in layer_activations_by_lang[args.lang][layer_idx]:
        positive_acts.append(acts_batch)
    
    if not positive_acts:
        print(f"Layer {layer_idx}: No positive activations found, skipping")
        continue
    
    positive_acts = np.concatenate(positive_acts, axis=0)
    num_positive = len(positive_acts)
    
    # Collect activations: negative classes (equal samples from each language)
    negative_acts_by_lang = {}
    
    for neg_lang in negative_langs:
        neg_lang = neg_lang.strip()
        lang_acts = []
        for acts_batch in layer_activations_by_lang[neg_lang][layer_idx]:
            lang_acts.append(acts_batch)
        
        if lang_acts:
            negative_acts_by_lang[neg_lang] = np.concatenate(lang_acts, axis=0)
    
    if not negative_acts_by_lang:
        print(f"Layer {layer_idx}: No negative activations found, skipping")
        continue
    
    # Number of actual languages with data available
    num_neg_langs = len(negative_acts_by_lang)
    
    # Sample equal number of examples from each available negative language
    samples_per_lang = num_positive // num_neg_langs
    
    negative_acts = []
    for neg_lang, acts in negative_acts_by_lang.items():
        if len(acts) > samples_per_lang:
            # Downsample
            indices = np.random.choice(len(acts), samples_per_lang, replace=False)
            negative_acts.append(acts[indices])
        else:
            # Upsample with replacement
            indices = np.random.choice(len(acts), samples_per_lang, replace=True)
            negative_acts.append(acts[indices])
    
    negative_acts = np.concatenate(negative_acts, axis=0)
    
    print(f"Layer {layer_idx}: {num_positive} positive, {len(negative_acts)} negative ({num_neg_langs} languages, {samples_per_lang} samples each)")
    
    # Create balanced training dataset
    X = np.concatenate([positive_acts, negative_acts], axis=0)
    y = np.concatenate([np.ones(num_positive), np.zeros(len(negative_acts))], axis=0)
    
    # Shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # Convert to tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    
    # ========== VALIDATION DATA ==========
    # Take 1000 pos and 1000 neg from training activations
    
    val_num_positive = min(1000, len(positive_acts))
    val_num_negative_per_lang = val_num_positive // num_neg_langs
    
    # Sample from positive
    val_indices_pos = np.random.choice(len(positive_acts), val_num_positive, replace=False)
    val_positive_acts = positive_acts[val_indices_pos]
    
    # Sample from negatives
    val_negative_acts = []
    for neg_lang, acts in negative_acts_by_lang.items():
        if len(acts) > val_num_negative_per_lang:
            indices = np.random.choice(len(acts), val_num_negative_per_lang, replace=False)
            val_negative_acts.append(acts[indices])
        else:
            indices = np.random.choice(len(acts), val_num_negative_per_lang, replace=True)
            val_negative_acts.append(acts[indices])
    
    val_negative_acts = np.concatenate(val_negative_acts, axis=0)
    
    X_val = np.concatenate([val_positive_acts, val_negative_acts], axis=0)
    y_val = np.concatenate([np.ones(val_num_positive), np.zeros(len(val_negative_acts))], axis=0)
    
    # Shuffle validation data
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    
    # Convert to tensors
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    
    # ========== CREATE AND TRAIN PROBE ==========
    
    # Create probe model
    probe = nn.Linear(hidden_size, 1, bias=True).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=args.probe_lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train probe
    probe.train()
    best_loss = float('inf')
    
    for epoch in range(args.probe_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X), args.probe_batch_size):
            batch_X = X[i:i + args.probe_batch_size].to(device)
            batch_y = y[i:i + args.probe_batch_size].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        print(f"  Epoch {epoch + 1}/{args.probe_epochs}: Loss = {avg_loss:.4f}")
    
    # ========== EVALUATE ON TRAINING SET ==========
    
    probe.eval()
    with torch.no_grad():
        logits = probe(X.to(device))
        predictions = (torch.sigmoid(logits) > 0.5).float().squeeze()
        accuracy = (predictions == y.to(device)).float().mean().item()
        
        # Compute per-class accuracy
        pos_acc = (predictions[:num_positive] == 1).float().mean().item()
        neg_acc = (predictions[num_positive:] == 0).float().mean().item()
    
    print(f"  Training - Overall Acc: {accuracy:.4f} | Pos Acc: {pos_acc:.4f} | Neg Acc: {neg_acc:.4f}")
    
    # ========== EVALUATE ON VALIDATION SET ==========
    
    with torch.no_grad():
        logits_val = probe(X_val.to(device))
        predictions_val = (torch.sigmoid(logits_val) > 0.5).float().squeeze()
        accuracy_val = (predictions_val == y_val.to(device)).float().mean().item()
        
        # Compute validation loss
        y_val_unsqueezed = y_val.to(device).unsqueeze(1)
        val_loss = criterion(logits_val, y_val_unsqueezed).item()
        
        # Compute per-class accuracy for validation
        pos_acc_val = (predictions_val[:val_num_positive] == 1).float().mean().item()
        neg_acc_val = (predictions_val[val_num_positive:] == 0).float().mean().item()
    
    print(f"  Validation - Overall Acc: {accuracy_val:.4f} | Pos Acc: {pos_acc_val:.4f} | Neg Acc: {neg_acc_val:.4f} | Loss: {val_loss:.4f}")
    
    probes[layer_idx] = probe.cpu()
    probe_results[layer_idx] = {
        'train_accuracy': accuracy,
        'train_pos_accuracy': pos_acc,
        'train_neg_accuracy': neg_acc,
        'train_loss': best_loss,
        'train_num_positive': num_positive,
        'train_num_negative': len(negative_acts),
        'val_accuracy': accuracy_val,
        'val_pos_accuracy': pos_acc_val,
        'val_neg_accuracy': neg_acc_val,
        'val_loss': val_loss,
        'val_num_positive': val_num_positive,
        'val_num_negative': len(val_negative_acts),
    }

print(f"\n{'='*60}")
print("Saving probes...")
print(f"{'='*60}")

# Save probes and results
output = {
    'probes': {layer_idx: probe.state_dict() for layer_idx, probe in probes.items()},
    'probe_results': probe_results,
    'target_language': args.lang,
    'negative_languages': negative_langs,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
}

import os

os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/probe.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved probes to {output_path}")

# Print summary statistics
print(f"\n{'='*60}")
print("Summary Statistics")
print(f"{'='*60}")

train_accs = [v['train_accuracy'] for v in probe_results.values()]
print(f"Training Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"  Min: {np.min(train_accs):.4f} | Max: {np.max(train_accs):.4f}")

val_accs = [v['val_accuracy'] for v in probe_results.values()]
print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
print(f"  Min: {np.min(val_accs):.4f} | Max: {np.max(val_accs):.4f}")

print(f"Number of probes trained: {len(probes)}/{num_layers}")