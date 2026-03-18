import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json

languages = [
    "en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", #  "af", "tl", "ur", "bn",
    "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk",
    "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"
]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="data_aya", help="Directory containing steering vectors")
parser.add_argument("-l", "--languages", type=str, nargs='+', default=languages, help="Language codes to compare")
parser.add_argument("-s", "--save_name", type=str, default="aya",  help="Save name (e.g., 'llama')")
parser.add_argument("-o", "--output_dir", type=str, default="plots_aya/lda", help="Output directory for plots and results")
parser.add_argument("--heatmap_layer", type=str, default="-1", help="Layer to use for similarity heatmap. -1 for average across all layers.")
args = parser.parse_args()

# --- ACL Publication Quality Style Settings ---
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
})
# --- End ACL Style Settings ---

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

def load_lda_results(data_dir, lang, save_name):
    """Load LDA results for a specific language."""
    lda_filepath = Path(data_dir) / f"lda.{lang}.{save_name}"
    
    if lda_filepath.exists():
        print(f"Loading {lda_filepath}")
        data = torch.load(lda_filepath)
        return data
    else:
        raise FileNotFoundError(f"LDA file not found: {lda_filepath}")

def get_language_families():
    """Define language families for grouping"""
    return {
        'Romance': ['es', 'fr', 'pt', 'it', 'ro', 'ca', 'gl'],
        'Germanic': ['en', 'de', 'nl', 'sv', 'da', 'no', 'af', 'is'],
        'Slavic': ['ru', 'pl', 'uk', 'cs', 'sk', 'bg', 'hr', 'sr', 'sl', 'be', 'mk'],
        'Indo-Aryan': ['hi', 'bn', 'pa', 'gu', 'mr', 'ne', 'si', 'ur', 'as', 'or'],
        'Dravidian': ['te', 'ta', 'kn', 'ml'],
        'Turkic': ['tr', 'az', 'uz', 'kk'],
        'Sino-Tibetan': ['zh', 'my', 'bo'],
        'Semitic': ['ar', 'he', 'am', 'mt'],
        'Japonic': ['ja'],
        'Koreanic': ['ko'],
        'Austroasiatic': ['vi'],
        'Kra-Dai': ['th'],
        'Austronesian': ['id', 'tl'],
        'Niger-Congo': ['sw'],
        'Iranian': ['fa'],
        'Kartvelian': ['ka'],
        'Armenian': ['hy'],
        'Basque': ['eu'],
        'Celtic': ['ga', 'cy'],
        'Hellenic': ['el'],
        'Uralic': ['fi', 'hu', 'et'],
        'Baltic': ['lv', 'lt'],
    }

def get_non_latin_scripts():
    """Languages with non-Latin scripts that should have asterisks"""
    return {'ar', 'am', 'hy', 'bn', 'as', 'my', 'zh', 'bo', 'el', 'gu', 'he', 
            'hi', 'mr', 'ne', 'ja', 'kn', 'ka', 'kk', 'ko', 'ml', 'mk', 'or', 
            'pa', 'fa', 'ru', 'sr', 'si', 'ta', 'te', 'th', 'uk', 'ur', 'be', 'bg'}

def format_language_label(lang):
    """Add asterisk for non-Latin script languages"""
    non_latin = get_non_latin_scripts()
    return f"{lang}*" if lang in non_latin else lang

def order_languages_by_family(languages, lang_families):
    """Order languages by family grouping"""
    ordered = []
    for family, family_langs in lang_families.items():
        for lang in family_langs:
            if lang in languages:
                ordered.append(lang)
    
    # Add any remaining languages not in families
    for lang in languages:
        if lang not in ordered:
            ordered.append(lang)
    
    return ordered

def get_family_positions(ordered_langs, lang_families):
    """Get positions where family boundaries should be drawn"""
    positions = []
    current_pos = 0
    current_family = None
    
    for lang in ordered_langs:
        lang_family = None
        for family, family_langs in lang_families.items():
            if lang in family_langs:
                lang_family = family
                break
        
        if current_family is not None and lang_family != current_family:
            positions.append(current_pos)
        
        current_family = lang_family
        current_pos += 1
    
    return positions

def add_family_separators_and_labels(ax, ordered_langs, lang_families, axis='both'):
    """Add family separators and labels to the plot - improved for readability"""
    positions = get_family_positions(ordered_langs, lang_families)
    
    # Add separator lines
    if axis in ['both', 'x']:
        for pos in positions:
            ax.axvline(x=pos, color='black', linewidth=1.2, alpha=0.6)
    if axis in ['both', 'y']:
        for pos in positions:
            ax.axhline(y=pos, color='black', linewidth=1.2, alpha=0.6)
    
    # Add family labels - improved positioning to avoid overlap
    num_langs = len(ordered_langs)
    label_font_size = max(10, 8 - num_langs // 40)  # Better scaling for many languages
    
    current_pos = 0
    family_positions = []  # Track positions for better spacing
    
    for family, family_langs in lang_families.items():
        family_count = sum(1 for lang in family_langs if lang in ordered_langs)
        if family_count > 0:
            # Calculate center position for the family
            center_pos = current_pos + family_count / 2
            family_positions.append((center_pos, family))
            current_pos += family_count
    
    # Add labels with better positioning - moved closer to heatmap
    if axis in ['both', 'x']:
        for pos, family in family_positions:
            # Position below x-axis at center of family group, closer to heatmap
            ax.text(pos, -2.5, family, ha='center', va='top', 
                   fontsize=label_font_size, fontweight='normal', rotation=270)
    if axis in ['both', 'y']:
        for pos, family in family_positions:
            # Position to the left of y-axis at center of family group, closer to heatmap
            ax.text(-2.0, pos, family, ha='right', va='center', 
                   fontsize=label_font_size, fontweight='normal', rotation=0)

# Load all LDA results
print("Loading LDA results for all languages...")
lda_data = {}
all_steering_vectors = {}
all_accuracies = {}
all_fisher_ratios = {}

for lang in args.languages:
    data = load_lda_results(args.data_dir, lang, args.save_name)
    lda_data[lang] = data
    
    # Extract steering vectors and per-layer metrics
    steering_vectors = data['steering_vectors']  # [num_layers, hidden_size]
    lda_results = data['lda_results']
    
    all_steering_vectors[lang] = steering_vectors
    
    accuracies = []
    fisher_ratios = []
    
    for layer_idx in sorted(lda_results.keys()):
        result = lda_results[layer_idx]
        accuracies.append(result['val_accuracy'])
        fisher_ratios.append(result['fisher_ratio'])
    
    all_accuracies[lang] = np.array(accuracies)
    all_fisher_ratios[lang] = np.array(fisher_ratios)

num_layers = len(all_accuracies[args.languages[0]])
print(f"Number of layers: {num_layers}")

# ============================================================================
# PLOT 1: Cosine Similarity Between All Languages (Individual Language Norms)
# ============================================================================
print("\nGenerating cosine similarity between language steering vectors plot...")

def compute_cosine_similarity_layers(vec1, vec2):
    """Compute cosine similarity between two sets of layer vectors."""
    vec1_norm = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2_norm = vec2 / vec2.norm(dim=1, keepdim=True)
    cos_sim = (vec1_norm * vec2_norm).sum(dim=1)
    return cos_sim

# Compute pairwise cosine similarities
all_similarities = []
pairwise_avg_similarities = {}
language_pairs = []

for i, lang1 in enumerate(args.languages):
    for j, lang2 in enumerate(args.languages):
        if i < j:
            pair_name = f"{lang1}-{lang2}"
            language_pairs.append((lang1, lang2))
            
            cos_sim = compute_cosine_similarity_layers(all_steering_vectors[lang1], all_steering_vectors[lang2])
            all_similarities.append(cos_sim.cpu().numpy())
            
            avg_sim = cos_sim.mean().item()
            pairwise_avg_similarities[pair_name] = avg_sim
            std_sim = cos_sim.std().item()
            print(f"{lang1} vs {lang2}: avg={avg_sim:.4f}, std={std_sim:.4f}")

all_similarities = np.array(all_similarities)  # [num_pairs, num_layers]

mean_similarity_per_layer = all_similarities.mean(axis=0)
std_similarity_per_layer = all_similarities.std(axis=0)

overall_mean_sim = mean_similarity_per_layer.mean()
overall_std_sim = mean_similarity_per_layer.std()

print(f"\nOverall average similarity across all layers and pairs: {overall_mean_sim:.4f} ± {overall_std_sim:.4f}")

# Plot cosine similarity
fig, ax = plt.subplots(figsize=(3.5, 2.5))

layers = np.arange(num_layers)

std_similarity_clamped = np.minimum(std_similarity_per_layer, 1.0)
upper_bound = np.minimum(mean_similarity_per_layer + std_similarity_clamped, 1.0)
lower_bound = np.maximum(mean_similarity_per_layer - std_similarity_clamped, -1.0)

ax.plot(layers, mean_similarity_per_layer, color='#2E86AB', linewidth=1.5)
ax.fill_between(layers, 
                 lower_bound,
                 upper_bound,
                 color='#2E86AB', alpha=0.2)

ax.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax.set_ylabel('Cosine Similarity', fontsize=10, fontweight='normal')
ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([0, 1.05])

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "lda_cosine_similarity_per_layer.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "lda_cosine_similarity_per_layer.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved cosine similarity plot to {Path(args.output_dir) / 'lda_cosine_similarity_per_layer.png'}")

# ============================================================================
# PLOT 2: LDA Accuracy and Fisher Ratio (Dual Axis, Averaged Across Languages)
# ============================================================================
print("\nGenerating accuracy and Fisher ratio averaged plot (dual axis)...")

# Average across all languages
mean_accuracy_per_layer = np.array([all_accuracies[lang] for lang in args.languages]).mean(axis=0)
std_accuracy_per_layer = np.array([all_accuracies[lang] for lang in args.languages]).std(axis=0)

mean_fisher_per_layer = np.array([all_fisher_ratios[lang] for lang in args.languages]).mean(axis=0)
std_fisher_per_layer = np.array([all_fisher_ratios[lang] for lang in args.languages]).std(axis=0)

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

# Plot accuracy on left axis
color_acc = '#2E86AB'
ax1.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax1.set_ylabel('Classification Accuracy', fontsize=10, fontweight='normal', color=color_acc)
ax1.plot(layers, mean_accuracy_per_layer, color=color_acc, linewidth=1.5, label='Accuracy')
ax1.fill_between(layers, 
                  mean_accuracy_per_layer - std_accuracy_per_layer,
                  mean_accuracy_per_layer + std_accuracy_per_layer,
                  color=color_acc, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=9, width=0.8)
ax1.tick_params(axis='x', labelsize=9, width=0.8)
ax1.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax1.set_ylim([0, 1.05])
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Create second y-axis for Fisher ratio
ax2 = ax1.twinx()
color_fisher = '#D1495B'
ax2.set_ylabel('Fisher Ratio', fontsize=10, fontweight='normal', color=color_fisher)
ax2.plot(layers, mean_fisher_per_layer, color=color_fisher, linewidth=1.5, label='Fisher Ratio')
ax2.fill_between(layers, 
                  mean_fisher_per_layer - std_fisher_per_layer,
                  mean_fisher_per_layer + std_fisher_per_layer,
                  color=color_fisher, alpha=0.2)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor=color_fisher, labelsize=9, width=0.8)

# Styling
ax1.spines['left'].set_color(color_acc)
ax1.spines['left'].set_linewidth(0.8)
ax2.spines['right'].set_color(color_fisher)
ax2.spines['right'].set_linewidth(0.8)

for spine in ax1.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

for spine in ax2.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "lda_accuracy_fisher_ratio_dual_axis.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "lda_accuracy_fisher_ratio_dual_axis.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved accuracy and Fisher ratio plot to {Path(args.output_dir) / 'lda_accuracy_fisher_ratio_dual_axis.png'}")

# ============================================================================
# PLOT 3: Similarity Heatmap (Averaged Over All Layers)
# ============================================================================
if len(args.languages) > 1:
    print("\nGenerating similarity heatmap averaged over all layers...")
    
    lang_families = get_language_families()
    ordered_languages = order_languages_by_family(args.languages, lang_families)
    
    print(f"Ordered languages by family: {ordered_languages}")
    
    num_langs = len(ordered_languages)
    similarity_matrix = np.zeros((num_langs, num_langs))
    
    # Compute average similarity across all layers
    for i, lang1 in enumerate(ordered_languages):
        for j, lang2 in enumerate(ordered_languages):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                cos_sim = compute_cosine_similarity_layers(
                    all_steering_vectors[lang1], 
                    all_steering_vectors[lang2]
                )
                avg_sim = cos_sim.mean().item()
                similarity_matrix[i, j] = avg_sim
                similarity_matrix[j, i] = avg_sim
    
    # Create formatted labels
    formatted_labels = [format_language_label(lang) for lang in ordered_languages]
    
    # Dynamic sizing
    fig_size = max(20, num_langs * 0.35)
    annotation_font_size = max(6, 9 - num_langs // 20)
    tick_font_size = max(10, 9 - num_langs // 15)
    
    print(f"Creating heatmap with size {fig_size}x{fig_size}")
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    min_similarity = similarity_matrix.min()
    max_similarity = similarity_matrix.max()
    
    print(f"Similarity range: {min_similarity:.3f} to {max_similarity:.3f}")
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=formatted_labels,
        yticklabels=formatted_labels,
        cmap="viridis",
        annot=False,
        cbar=True,
        linewidths=0.05 if num_langs > 50 else 0.2,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        vmin=min_similarity,
        vmax=max_similarity
    )
    
    # Add family separators
    add_family_separators_and_labels(ax, ordered_languages, lang_families, 'both')
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Cosine Similarity (Avg. Over Layers)", fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=12, width=0.8)
    cbar.outline.set_linewidth(0.8)
    
    plt.xticks(rotation=90, ha="center", fontsize=tick_font_size)
    plt.yticks(rotation=0, fontsize=tick_font_size)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
        spine.set_visible(True)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.98, bottom=0.08)
    
    plt.savefig(Path(args.output_dir) / "lda_similarity_heatmap_avg_layers.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(Path(args.output_dir) / "lda_similarity_heatmap_avg_layers.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved similarity heatmap to {Path(args.output_dir) / 'lda_similarity_heatmap_avg_layers.png'}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

summary_stats = {}

for lang in args.languages:
    accuracies = all_accuracies[lang]
    fisher_ratios = all_fisher_ratios[lang]
    
    # Get detailed results from original data
    lda_results = lda_data[lang]['lda_results']
    pos_accuracies = [lda_results[i]['val_pos_accuracy'] for i in sorted(lda_results.keys())]
    neg_accuracies = [lda_results[i]['val_neg_accuracy'] for i in sorted(lda_results.keys())]
    
    stats = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'best_layer_accuracy': int(np.argmax(accuracies)),
        'mean_fisher_ratio': float(np.mean(fisher_ratios)),
        'std_fisher_ratio': float(np.std(fisher_ratios)),
        'mean_pos_accuracy': float(np.mean(pos_accuracies)),
        'mean_neg_accuracy': float(np.mean(neg_accuracies)),
    }
    
    summary_stats[lang] = stats
    
    print(f"\n{lang.upper()}:")
    print(f"  Classification Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    print(f"    Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]")
    print(f"    Best layer: {stats['best_layer_accuracy']}")
    print(f"  Positive class accuracy: {stats['mean_pos_accuracy']:.4f}")
    print(f"  Negative class accuracy: {stats['mean_neg_accuracy']:.4f}")
    print(f"  Fisher Ratio: {stats['mean_fisher_ratio']:.4f} ± {stats['std_fisher_ratio']:.4f}")

print(f"\nOverall:")
print(f"  Mean Cosine Similarity (Pairwise): {overall_mean_sim:.4f} ± {overall_std_sim:.4f}")

# Save results to JSON
results_path = Path(args.output_dir) / "lda_analysis_summary.json"
with open(results_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)