import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

languages = [
    "en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", #  "af", "tl", "ur", "bn",
    "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk",
    "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"
]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="data_aya", help="Directory containing steering vectors")
parser.add_argument("-l", "--languages", type=str, nargs='+', default=languages, help="Language codes to compare")
parser.add_argument("-s", "--save_name", type=str, default="aya",  help="Save name (e.g., 'llama')")
parser.add_argument("-o", "--output_dir", type=str, default="plots_aya/probes", help="Output directory for plots and results")
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

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

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

def load_probe_vectors(data_dir, lang, save_name):
    """Load probe vectors and results for a specific language."""
    probe_filepath = Path(data_dir) / f"probe.{lang}.{save_name}"
    
    if probe_filepath.exists():
        print(f"Loading {probe_filepath}")
        data = torch.load(probe_filepath)
        return data
    else:
        raise FileNotFoundError(f"Probe file not found: {probe_filepath}")

def get_probe_weight_vectors(probe_data):
    """Extract weight vectors from probe state dicts."""
    probes = probe_data['probes']
    weight_vectors = {}
    
    for layer_idx, state_dict in probes.items():
        # state_dict has 'weight' [1, hidden_size] and 'bias' [1]
        weight = state_dict['weight'].squeeze(0)  # [hidden_size]
        weight_vectors[layer_idx] = weight
    
    return weight_vectors

def cosine_similarity_layers(vec1, vec2):
    """Compute cosine similarity between two sets of layer vectors."""
    vec1_norm = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2_norm = vec2 / vec2.norm(dim=1, keepdim=True)
    
    cos_sim = (vec1_norm * vec2_norm).sum(dim=1)
    return cos_sim

def compute_vector_norms(vectors):
    """Compute L2 norm for each layer."""
    return vectors.norm(dim=1)

# Load all probe data
print("Loading probe vectors and results...")
lang_probe_data = {}
lang_weight_vectors = {}

for lang in args.languages:
    probe_data = load_probe_vectors(args.data_dir, lang, args.save_name)
    lang_probe_data[lang] = probe_data
    
    weight_vecs = get_probe_weight_vectors(probe_data)
    lang_weight_vectors[lang] = weight_vecs

num_layers = lang_probe_data[args.languages[0]]['num_layers']
print(f"Number of layers: {num_layers}")

# ============================================================================
# ACCURACY AND LOSS PLOTS
# ============================================================================

print("\n" + "="*60)
print("GENERATING ACCURACY AND LOSS PLOTS")
print("="*60)

# For each language, collect accuracy and loss across layers
accuracy_by_lang = {}
loss_by_lang = {}

for lang in args.languages:
    probe_results = lang_probe_data[lang]['probe_results']
    
    accuracies = []
    losses = []
    
    for layer_idx in range(num_layers):
        if layer_idx in probe_results:
            accuracies.append(probe_results[layer_idx]['val_accuracy'])
            losses.append(probe_results[layer_idx]['val_loss'])
        else:
            accuracies.append(np.nan)
            losses.append(np.nan)
    
    accuracy_by_lang[lang] = np.array(accuracies)
    loss_by_lang[lang] = np.array(losses)

# Compute statistics across languages
all_accuracies = np.array([accuracy_by_lang[lang] for lang in args.languages])
all_losses = np.array([loss_by_lang[lang] for lang in args.languages])

mean_accuracy_per_layer = np.nanmean(all_accuracies, axis=0)
std_accuracy_per_layer = np.nanstd(all_accuracies, axis=0)

mean_loss_per_layer = np.nanmean(all_losses, axis=0)
std_loss_per_layer = np.nanstd(all_losses, axis=0)

layers = np.arange(num_layers)

print(f"Mean Accuracy: {np.nanmean(mean_accuracy_per_layer):.4f} ± {np.nanstd(mean_accuracy_per_layer):.4f}")
print(f"Mean Loss: {np.nanmean(mean_loss_per_layer):.4f} ± {np.nanstd(mean_loss_per_layer):.4f}")

# --- Plot: Accuracy and Loss on same plot (dual axis) ---
print("Generating accuracy and loss plot (dual axis)...")
fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

# Plot accuracy on left axis
color_acc = '#2E86AB'
ax1.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax1.set_ylabel('Accuracy', fontsize=10, fontweight='normal', color=color_acc)
ax1.plot(layers, mean_accuracy_per_layer, color=color_acc, linewidth=1.5, label='Accuracy')
ax1.fill_between(layers, 
                  mean_accuracy_per_layer - std_accuracy_per_layer,
                  mean_accuracy_per_layer + std_accuracy_per_layer,
                  color=color_acc, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=9, width=0.8)
ax1.tick_params(axis='x', labelsize=9, width=0.8)
ax1.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Create second y-axis for loss
ax2 = ax1.twinx()
color_loss = '#D1495B'
ax2.set_ylabel('Loss', fontsize=10, fontweight='normal', color=color_loss)
ax2.plot(layers, mean_loss_per_layer, color=color_loss, linewidth=1.5, label='Loss')
ax2.fill_between(layers, 
                  mean_loss_per_layer - std_loss_per_layer,
                  mean_loss_per_layer + std_loss_per_layer,
                  color=color_loss, alpha=0.2)
ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=9, width=0.8)

# Styling
ax1.spines['left'].set_color(color_acc)
ax1.spines['left'].set_linewidth(0.8)
ax2.spines['right'].set_color(color_loss)
ax2.spines['right'].set_linewidth(0.8)

for spine in ax1.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

for spine in ax2.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "accuracy_loss_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "accuracy_loss_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved accuracy/loss plot to {Path(args.output_dir) / 'accuracy_loss_plot.png'}")

# --- Plot: Individual Language Accuracy Across Layers ---
if len(args.languages) > 1:
    print("Generating individual language accuracy plot...")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    n_colors_needed = len(args.languages)
    if n_colors_needed <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_colors_needed <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
        colors = np.vstack([colors1, colors2, colors3])

    line_width = 1.2 if n_colors_needed <= 30 else 0.8
    marker_size = 3 if n_colors_needed <= 20 else 0
    show_markers = n_colors_needed <= 20

    for i, lang in enumerate(args.languages):
        ax.plot(layers, accuracy_by_lang[lang], 
                marker='o' if show_markers else None,
                label=f'{lang.upper()}', 
                linewidth=line_width,
                markersize=marker_size,
                color=colors[i % len(colors)],
                alpha=0.75)

    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_cols = min(4, max(1, len(args.languages) // 10))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=legend_cols,
              fontsize=7, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "individual_language_accuracy_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(Path(args.output_dir) / "individual_language_accuracy_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual language accuracy plot to {Path(args.output_dir) / 'individual_language_accuracy_plot.png'}")

# ============================================================================
# SIMILARITY AND NORMS ANALYSIS (from probe weight vectors)
# ============================================================================

print("\n" + "="*60)
print("GENERATING PROBE WEIGHT SIMILARITY AND NORMS PLOTS")
print("="*60)

# Collect weight vectors for all layers
weight_vectors_by_lang = {}

for lang in args.languages:
    weight_vecs = lang_weight_vectors[lang]
    # Stack weight vectors across layers
    weight_tensor = torch.stack([weight_vecs[i] for i in range(num_layers)])
    weight_vectors_by_lang[lang] = weight_tensor

# Compute pairwise cosine similarities
print("\nComputing pairwise cosine similarities of probe weights...")
all_similarities = []
pairwise_avg_similarities = {}

for i, lang1 in enumerate(args.languages):
    for j, lang2 in enumerate(args.languages):
        if i < j:
            pair_name = f"{lang1}-{lang2}"
            
            cos_sim = cosine_similarity_layers(weight_vectors_by_lang[lang1], weight_vectors_by_lang[lang2])
            all_similarities.append(cos_sim.numpy())
            
            avg_sim = cos_sim.mean().item()
            pairwise_avg_similarities[pair_name] = avg_sim
            std_sim = cos_sim.std().item()
            print(f"{lang1} vs {lang2}: avg={avg_sim:.4f}, std={std_sim:.4f}")

all_similarities = np.array(all_similarities)

mean_similarity_per_layer = all_similarities.mean(axis=0)
std_similarity_per_layer = all_similarities.std(axis=0)

overall_mean_sim = mean_similarity_per_layer.mean()
overall_std_sim = mean_similarity_per_layer.std()

print(f"\nOverall average similarity across all layers and pairs: {overall_mean_sim:.4f} ± {overall_std_sim:.4f}")

# Compute vector norms
print("\nComputing probe weight vector norms...")
all_norms = []
language_avg_norms = {}

for lang in args.languages:
    norms = compute_vector_norms(weight_vectors_by_lang[lang])
    all_norms.append(norms.numpy())
    
    avg_norm = norms.mean().item()
    language_avg_norms[lang] = avg_norm
    std_norm = norms.std().item()
    print(f"{lang}: avg norm={avg_norm:.4f}, std={std_norm:.4f}")

all_norms = np.array(all_norms)

mean_norm_per_layer = all_norms.mean(axis=0)
std_norm_per_layer = all_norms.std(axis=0)

overall_mean_norm = mean_norm_per_layer.mean()
overall_std_norm = mean_norm_per_layer.std()

print(f"\nOverall average norm across all layers and languages: {overall_mean_norm:.4f} ± {overall_std_norm:.4f}")

# Compute pairwise difference norms
print("\nComputing average difference (diffmean) vector norms...")

diff_norms = []

for i, lang1 in enumerate(args.languages):
    for j, lang2 in enumerate(args.languages):
        if i < j:
            diff_vec = weight_vectors_by_lang[lang2] - weight_vectors_by_lang[lang1]
            diff_norm = diff_vec.norm(dim=1)
            diff_norms.append(diff_norm.numpy())

diff_norms = np.array(diff_norms)
mean_diff_norm_per_layer = diff_norms.mean(axis=0)
std_diff_norm_per_layer = diff_norms.std(axis=0)

overall_mean_diff_norm = mean_diff_norm_per_layer.mean()
overall_std_diff_norm = mean_diff_norm_per_layer.std()

print(f"Overall average diffmean norm: {overall_mean_diff_norm:.4f} ± {overall_std_diff_norm:.4f}")

# --- Plot: Average Diffmean Vector Norms Across Layers ---
print("Generating diffmean norms plot...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))

ax.plot(layers, mean_diff_norm_per_layer, color='#D1495B', linewidth=1.5, label='Mean Δμ norm')
ax.fill_between(
    layers,
    mean_diff_norm_per_layer - std_diff_norm_per_layer,
    mean_diff_norm_per_layer + std_diff_norm_per_layer,
    color='#D1495B', alpha=0.25
)

ax.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax.set_ylabel('L2 Norm (Δμ)', fontsize=10, fontweight='normal')
ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "diffmean_norms_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "diffmean_norms_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved diffmean norms plot to {Path(args.output_dir) / 'diffmean_norms_plot.png'}")

# --- Plot: Mean Cosine Similarity Across Layers ---
print("Generating mean cosine similarity plot...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Clamp standard deviation to not exceed 1.0
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

# Dynamically set y-axis limits
y_min = max(-1.0, lower_bound.min() - 0.05)
y_max = min(1.0, upper_bound.max() + 0.05)
ax.set_ylim([y_min, y_max])

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "cosine_similarity_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "cosine_similarity_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved cosine similarity plot to {Path(args.output_dir) / 'cosine_similarity_plot.png'}")

# --- Plot: Mean Vector Norms Across Layers ---
print("Generating mean vector norms plot...")
fig, ax = plt.subplots(figsize=(3.5, 2.5))

ax.plot(layers, mean_norm_per_layer, color='#2E86AB', linewidth=1.5)
ax.fill_between(layers, 
                 mean_norm_per_layer - std_norm_per_layer,
                 mean_norm_per_layer + std_norm_per_layer,
                 color='#2E86AB', alpha=0.2)

ax.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax.set_ylabel('L2 Norm', fontsize=10, fontweight='normal')
ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "vector_norms_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "vector_norms_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved vector norms plot to {Path(args.output_dir) / 'vector_norms_plot.png'}")

# --- Plot: Individual Language Probe Weight Norms Across Layers ---
if len(args.languages) > 1:
    print("Generating individual language probe weight norms plot...")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    n_colors_needed = len(args.languages)
    if n_colors_needed <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_colors_needed <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
        colors = np.vstack([colors1, colors2, colors3])

    line_width = 1.2 if n_colors_needed <= 30 else 0.8
    marker_size = 3 if n_colors_needed <= 20 else 0
    show_markers = n_colors_needed <= 20

    for i, lang in enumerate(args.languages):
        ax.plot(layers, all_norms[i], 
                marker='o' if show_markers else None,
                label=f'{lang.upper()}', 
                linewidth=line_width,
                markersize=marker_size,
                color=colors[i % len(colors)],
                alpha=0.75)

    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('L2 Norm', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_cols = min(4, max(1, len(args.languages) // 10))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=legend_cols,
              fontsize=7, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "individual_language_probe_norms_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(Path(args.output_dir) / "individual_language_probe_norms_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual language probe norms plot to {Path(args.output_dir) / 'individual_language_probe_norms_plot.png'}")

# --- Plot: Similarity Heatmap with Family Grouping ---
if len(args.languages) > 1:
    print("Generating similarity heatmap with language family grouping...")
    
    lang_families = get_language_families()
    ordered_languages = order_languages_by_family(args.languages, lang_families)
    
    print(f"Ordered languages by family: {ordered_languages}")
    
    num_langs = len(ordered_languages)
    similarity_matrix = np.zeros((num_langs, num_langs))
    
    # Parse heatmap layer argument
    if isinstance(args.heatmap_layer, str):
        parts = args.heatmap_layer.strip().split()
        if len(parts) == 1:
            heatmap_layers = [int(parts[0])]
        elif len(parts) == 2:
            heatmap_layers = [int(parts[0]), int(parts[1])]
        else:
            raise ValueError(f"Invalid value for --heatmap_layer: {args.heatmap_layer}")
    else:
        heatmap_layers = [int(args.heatmap_layer)]
    
    if len(heatmap_layers) == 1:
        if heatmap_layers[0] == -1:
            print("Using average similarity across all layers for heatmap.")
            for i, lang1 in enumerate(ordered_languages):
                for j, lang2 in enumerate(ordered_languages):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    elif i < j:
                        cos_sim = cosine_similarity_layers(weight_vectors_by_lang[lang1], weight_vectors_by_lang[lang2])
                        similarity_matrix[i, j] = cos_sim.mean().item()
                        similarity_matrix[j, i] = cos_sim.mean().item()
        else:
            layer_idx = heatmap_layers[0] if heatmap_layers[0] >= 0 else num_layers + heatmap_layers[0]
            print(f"Using layer {layer_idx} for similarity heatmap.")
            for i, lang1 in enumerate(ordered_languages):
                for j, lang2 in enumerate(ordered_languages):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    elif i < j:
                        vec1_layer = weight_vectors_by_lang[lang1][layer_idx].unsqueeze(0)
                        vec2_layer = weight_vectors_by_lang[lang2][layer_idx].unsqueeze(0)
                        cos_sim = cosine_similarity_layers(vec1_layer, vec2_layer).item()
                        similarity_matrix[i, j] = cos_sim
                        similarity_matrix[j, i] = cos_sim
    else:
        start_layer, end_layer = heatmap_layers
        print(f"Using average similarity across layers {start_layer}–{end_layer} for heatmap.")
        for i, lang1 in enumerate(ordered_languages):
            for j, lang2 in enumerate(ordered_languages):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    cos_sim = cosine_similarity_layers(
                        weight_vectors_by_lang[lang1][start_layer:end_layer+1],
                        weight_vectors_by_lang[lang2][start_layer:end_layer+1],
                    )
                    similarity_matrix[i, j] = cos_sim.mean().item()
                    similarity_matrix[j, i] = cos_sim.mean().item()
    
    # Create formatted labels
    formatted_labels = [format_language_label(lang) for lang in ordered_languages]
    
    # Dynamic sizing
    fig_size = max(20, num_langs * 0.35)
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
    cbar.ax.set_ylabel("Cosine Similarity", fontsize=16, labelpad=10)
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
    
    plt.savefig(Path(args.output_dir) / "similarity_heatmap.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(Path(args.output_dir) / "similarity_heatmap.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved similarity heatmap to {Path(args.output_dir) / 'similarity_heatmap.png'}")

# Save numerical results
results = {
    'languages': args.languages,
    'accuracy_per_layer': mean_accuracy_per_layer,
    'accuracy_std_per_layer': std_accuracy_per_layer,
    'loss_per_layer': mean_loss_per_layer,
    'loss_std_per_layer': std_loss_per_layer,
    'mean_similarity_per_layer': mean_similarity_per_layer,
    'std_similarity_per_layer': std_similarity_per_layer,
    'overall_mean_similarity': overall_mean_sim,
    'overall_std_similarity': overall_std_sim,
    'all_pairwise_similarities_per_layer': all_similarities,
    'pairwise_avg_similarities': pairwise_avg_similarities,
    'mean_norm_per_layer': mean_norm_per_layer,
    'std_norm_per_layer': std_norm_per_layer,
    'overall_mean_norm': overall_mean_norm,
    'overall_std_norm': overall_std_norm,
    'all_norms_per_language_layer': all_norms,
    'language_avg_norms': language_avg_norms,
}

results_path = Path(args.output_dir) / "probe_analysis_results.pt"
torch.save(results, results_path)
print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("ALL PLOTS AND RESULTS GENERATED SUCCESSFULLY!")
print("="*60)