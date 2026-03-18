import argparse
from types import MethodType
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-l", "--lang", type=str, default="eng_Latn", help="Target language (positive class)")
parser.add_argument("-n", "--negative_langs", type=str, default="ru,de,zh", help="Comma-separated negative languages")
parser.add_argument("-s", "--save", type=str, default="llama-3")
parser.add_argument("--target_tokens", type=int, default=100_000, help="Target tokens for positive language")
parser.add_argument("--val_samples_per_class", type=int, default=1000, help="Validation samples per class")
parser.add_argument("--substitute_lang", type=str, default="fr", help="Substitute language if one of the negative languages matches positive class")
args = parser.parse_args()

save = args.save.split(" ")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
    # Reshape token IDs directly without decoding
    l = len(token_ids)
    seq_length = min(max_length, 32768)
    l = min(l, 99999744) // seq_length * seq_length
    input_ids = token_ids[:l].reshape(-1, seq_length)
    
    print(f"Processing {input_ids.size(0)} sequences of length {seq_length}...")
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    
    # Generate with prompt_token_ids
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
print("Computing LDA steering vectors for each layer (GPU-accelerated)...")
print(f"{'='*60}")

# Compute LDA for each layer using PyTorch
lda_steering_vectors = torch.zeros(num_layers, hidden_size, dtype=torch.float32)
lda_results = {}

def compute_lda_torch(X, y, X_val=None, y_val=None):
    """
    GPU-accelerated LDA computation using PyTorch with language knowledge metrics.
    
    For binary classification, computes the discriminant direction:
    w = Σ_w^{-1} (μ_1 - μ_0)
    
    where Σ_w is the within-class covariance matrix.
    
    Also computes additional metrics to quantify language-specific knowledge.
    """
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    # Separate by class
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    
    # Compute means
    mu_pos = X_pos.mean(dim=0)
    mu_neg = X_neg.mean(dim=0)
    
    # Compute within-class covariance
    X_pos_centered = X_pos - mu_pos
    X_neg_centered = X_neg - mu_neg
    
    # Compute scatter matrices
    S_pos = X_pos_centered.T @ X_pos_centered
    S_neg = X_neg_centered.T @ X_neg_centered
    S_w = (S_pos + S_neg) / len(X)
    
    # Add regularization to avoid singular matrix
    reg = 1e-6 * torch.eye(S_w.shape[0], device=device)
    S_w_reg = S_w + reg
    
    # Compute LDA direction: w = Σ_w^{-1} (μ_1 - μ_0)
    mean_diff = mu_pos - mu_neg
    try:
        w = torch.linalg.solve(S_w_reg, mean_diff)
    except:
        # If solve fails, use pseudoinverse
        w = torch.linalg.pinv(S_w_reg) @ mean_diff
    
    # Compute accuracy on training set
    scores = X @ w
    predictions = (scores > 0).long()
    accuracy = (predictions == y).float().mean()
    
    pos_acc = (predictions[y == 1] == 1).float().mean()
    neg_acc = (predictions[y == 0] == 0).float().mean()
    
    # --- Language Knowledge Quantification Metrics ---
    
    # 1. Fisher's Ratio (class separability strength)
    fisher_ratio = (mean_diff @ torch.linalg.pinv(S_w) @ mean_diff).item()
    
    # 2. Between-class vs Within-class scatter ratio
    S_b = len(X_pos) * torch.outer(mean_diff, mean_diff)
    scatter_ratio = (torch.trace(S_b) / torch.trace(S_w)).item()
    
    # 3. Mean activation magnitude per class
    pos_magnitude = torch.norm(mu_pos).item()
    neg_magnitude = torch.norm(mu_neg).item()
    magnitude_diff = pos_magnitude - neg_magnitude
    
    # 4. Per-class variance (dimensionality of representation)
    var_pos = (torch.trace(X_pos_centered.T @ X_pos_centered) / len(X_pos)).item()
    var_neg = (torch.trace(X_neg_centered.T @ X_neg_centered) / len(X_neg)).item()
    
    # 5. Euclidean distance between class means in original space
    euclidean_distance = torch.norm(mu_pos - mu_neg).item()
    
    metrics = {
        'train_accuracy': accuracy.item(),
        'train_pos_accuracy': pos_acc.item(),
        'train_neg_accuracy': neg_acc.item(),
        'fisher_ratio': fisher_ratio,
        'scatter_ratio': scatter_ratio,
        'pos_magnitude': pos_magnitude,
        'neg_magnitude': neg_magnitude,
        'magnitude_diff': magnitude_diff,
        'var_pos': var_pos,
        'var_neg': var_neg,
        'euclidean_distance': euclidean_distance,
    }
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        
        scores_val = X_val @ w
        predictions_val = (scores_val > 0).long()
        accuracy_val = (predictions_val == y_val).float().mean()
        
        pos_acc_val = (predictions_val[y_val == 1] == 1).float().mean()
        neg_acc_val = (predictions_val[y_val == 0] == 0).float().mean()
        
        # Compute validation loss using BCEWithLogitsLoss (same as training)
        criterion = nn.BCEWithLogitsLoss()
        y_val_float = y_val.float().unsqueeze(1)
        val_loss = criterion(scores_val.unsqueeze(1), y_val_float).item()
        
        metrics['val_accuracy'] = accuracy_val.item()
        metrics['val_pos_accuracy'] = pos_acc_val.item()
        metrics['val_neg_accuracy'] = neg_acc_val.item()
        metrics['val_loss'] = val_loss
    
    return w.cpu(), torch.norm(w).cpu().item(), metrics

for layer_idx in range(num_layers):
    print(f"\nComputing LDA for layer {layer_idx}/{num_layers}...")
    
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
    
    # Compute LDA using GPU
    steering_vector, coef_norm, metrics = compute_lda_torch(X, y, X_val, y_val)
    
    # Store the steering vector
    lda_steering_vectors[layer_idx] = steering_vector
    
    print(f"  LDA Coef Norm: {coef_norm:.4f}")
    print(f"  Training - Overall Acc: {metrics['train_accuracy']:.4f} | Pos Acc: {metrics['train_pos_accuracy']:.4f} | Neg Acc: {metrics['train_neg_accuracy']:.4f}")
    
    if 'val_accuracy' in metrics:
        print(f"  Validation - Overall Acc: {metrics['val_accuracy']:.4f} | Pos Acc: {metrics['val_pos_accuracy']:.4f} | Neg Acc: {metrics['val_neg_accuracy']:.4f} | Loss: {metrics['val_loss']:.4f}")
    
    print(f"  Fisher Ratio: {metrics['fisher_ratio']:.4f}")
    
    lda_results[layer_idx] = {
        'coef_norm': float(coef_norm),
        'num_positive': num_positive,
        'num_negative': len(negative_acts),
        **metrics  # Add all knowledge quantification metrics and accuracy metrics
    }


print(f"\n{'='*60}")
print("Saving LDA steering vectors...")
print(f"{'='*60}")

# Save LDA results
output = {
    'steering_vectors': lda_steering_vectors.to('cpu'),  # [num_layers, hidden_size]
    'lda_results': lda_results,
    'target_language': args.lang,
    'negative_languages': negative_langs,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'method': 'LDA_GPU_with_metrics'
}

import os

os.makedirs(f'data_{save[0]}', exist_ok=True)
output_path = f'data_{save[0]}/lda.{args.lang}.{save[1]}'
torch.save(output, output_path)
print(f"Saved LDA steering vectors to {output_path}")
print(f"Shape: {lda_steering_vectors.shape}")

# Print summary statistics
print(f"\n{'='*60}")
print("Summary Statistics")
print(f"{'='*60}")
if lda_results:
    train_accs = [v['train_accuracy'] for v in lda_results.values()]
    fisher_ratios = [v['fisher_ratio'] for v in lda_results.values()]
    
    print(f"Training Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"  Min: {np.min(train_accs):.4f} | Max: {np.max(train_accs):.4f}")
    
    val_accs = [v['val_accuracy'] for v in lda_results.values()]
    print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"  Min: {np.min(val_accs):.4f} | Max: {np.max(val_accs):.4f}")
    
    print(f"\nMean Fisher Ratio: {np.mean(fisher_ratios):.4f} ± {np.std(fisher_ratios):.4f}")
    print(f"\nNumber of LDA vectors computed: {len(lda_results)}/{num_layers}")