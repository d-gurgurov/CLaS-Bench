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
parser.add_argument("-o", "--output_dir", type=str, default="plots_aya/vectors", help="Output directory for plots and results")
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

def load_steering_vectors(data_dir, lang, save_name):
    """Load steering vectors and variance explained for a specific language."""
    # Try loading from PCA file first (has variance_explained)
    steering_filepath = Path(data_dir) / f"vector.{lang}.{save_name}"
    
    print(f"Loading {steering_filepath}")
    data = torch.load(steering_filepath)
    return data['steering_vectors'], data.get('variance_explained', None)

def cosine_similarity_layers(vec1, vec2):
    """Compute cosine similarity between two sets of layer vectors."""
    # vec1, vec2: [num_layers, hidden_size]
    vec1_norm = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2_norm = vec2 / vec2.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity for each layer
    cos_sim = (vec1_norm * vec2_norm).sum(dim=1)  # [num_layers]
    return cos_sim

def compute_vector_norms(vectors):
    """Compute L2 norm for each layer."""
    # vectors: [num_layers, hidden_size]
    return vectors.norm(dim=1)  # [num_layers]

# Load all language vectors and variance explained
print("Loading steering vectors and variance explained...")
lang_vectors = {}
lang_variance = {}

for lang in args.languages:
    vectors, variance = load_steering_vectors(args.data_dir, lang, args.save_name)
    lang_vectors[lang] = vectors
    lang_variance[lang] = variance

num_layers = list(lang_vectors.values())[0].shape[0]
print(f"Number of layers: {num_layers}")

# ============================================================================
# SIMILARITY AND NORMS ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("GENERATING SIMILARITY AND NORMS PLOTS")
print("="*60)

# Compute pairwise cosine similarities
print("\nComputing pairwise cosine similarities...")
all_similarities = [] # Stores [num_pairs, num_layers]
pairwise_avg_similarities = {} # Stores avg sim for each pair
language_pairs = []

for i, lang1 in enumerate(args.languages):
    for j, lang2 in enumerate(args.languages):
        if i < j:  # Only compute upper triangle
            pair_name = f"{lang1}-{lang2}"
            language_pairs.append((lang1, lang2))
            
            cos_sim = cosine_similarity_layers(lang_vectors[lang1], lang_vectors[lang2])
            all_similarities.append(cos_sim.numpy())
            
            avg_sim = cos_sim.mean().item()
            pairwise_avg_similarities[pair_name] = avg_sim
            std_sim = cos_sim.std().item()
            print(f"{lang1} vs {lang2}: avg={avg_sim:.4f}, std={std_sim:.4f}")

all_similarities = np.array(all_similarities) # [num_pairs, num_layers]

mean_similarity_per_layer = all_similarities.mean(axis=0)  # [num_layers]
std_similarity_per_layer = all_similarities.std(axis=0)    # [num_layers]

overall_mean_sim = mean_similarity_per_layer.mean()
overall_std_sim = mean_similarity_per_layer.std()

print(f"\nOverall average similarity across all layers and pairs: {overall_mean_sim:.4f} ± {overall_std_sim:.4f}")


# Compute vector norms for all languages
print("\nComputing vector norms...")
all_norms = [] # Stores [num_languages, num_layers]
language_avg_norms = {}

for lang in args.languages:
    norms = compute_vector_norms(lang_vectors[lang])
    all_norms.append(norms.numpy())
    
    avg_norm = norms.mean().item()
    language_avg_norms[lang] = avg_norm
    std_norm = norms.std().item()
    print(f"{lang}: avg norm={avg_norm:.4f}, std={std_norm:.4f}")

all_norms = np.array(all_norms) # [num_languages, num_layers]

mean_norm_per_layer = all_norms.mean(axis=0)  # [num_layers]
std_norm_per_layer = all_norms.std(axis=0)    # [num_layers]

overall_mean_norm = mean_norm_per_layer.mean()
overall_std_norm = mean_norm_per_layer.std()

print(f"\nOverall average norm across all layers and languages: {overall_mean_norm:.4f} ± {overall_std_norm:.4f}")

# Compute pairwise difference norms
print("\nComputing average difference (diffmean) vector norms across languages...")

diff_norms = []  # [num_pairs, num_layers]

for i, lang1 in enumerate(args.languages):
    for j, lang2 in enumerate(args.languages):
        if i < j:
            diff_vec = lang_vectors[lang2] - lang_vectors[lang1]  # [num_layers, hidden_size]
            diff_norm = diff_vec.norm(dim=1)  # [num_layers]
            diff_norms.append(diff_norm.numpy())

diff_norms = np.array(diff_norms)  # [num_pairs, num_layers]
mean_diff_norm_per_layer = diff_norms.mean(axis=0)
std_diff_norm_per_layer = diff_norms.std(axis=0)

overall_mean_diff_norm = mean_diff_norm_per_layer.mean()
overall_std_diff_norm = mean_diff_norm_per_layer.std()

print(f"Overall average diffmean norm: {overall_mean_diff_norm:.4f} ± {overall_std_diff_norm:.4f}")

layers = np.arange(num_layers)

# --- Plot: Individual Language Similarities Across Layers ---
if len(args.languages) > 1:
    print("Generating individual language similarities plot...")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    # Compute average similarity of each language with all others per layer
    individual_similarities = []  # [num_languages, num_layers]
    
    for lang1 in args.languages:
        lang_sims = []  # [num_other_languages, num_layers]
        for lang2 in args.languages:
            if lang1 != lang2:
                cos_sim = cosine_similarity_layers(lang_vectors[lang1], lang_vectors[lang2])
                lang_sims.append(cos_sim.numpy())
        # Average similarity with all other languages per layer
        avg_sim_per_layer = np.mean(lang_sims, axis=0)  # [num_layers]
        individual_similarities.append(avg_sim_per_layer)
    
    individual_similarities = np.array(individual_similarities)  # [num_languages, num_layers]

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
        ax.plot(layers, individual_similarities[i], 
                marker='o' if show_markers else None,
                label=f'{lang.upper()}', 
                linewidth=line_width,
                markersize=marker_size,
                color=colors[i % len(colors)],
                alpha=0.75)

    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Avg Cosine Similarity', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)

    # Set y-axis limits appropriately
    y_min = max(-1.0, individual_similarities.min() - 0.05)
    y_max = min(1.0, individual_similarities.max() + 0.05)
    ax.set_ylim([y_min, y_max])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_cols = min(4, max(1, len(args.languages) // 10))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=legend_cols,
              fontsize=7, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "individual_language_similarities_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(Path(args.output_dir) / "individual_language_similarities_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual language similarities plot to {Path(args.output_dir) / 'individual_language_similarities_plot.png'}")

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

# Dynamically set y-axis limits to show all data
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

# --- Plot: Individual Language Similarities with English Across Layers ---
if 'en' in args.languages and len(args.languages) > 1:
    print("Generating language similarities with English plot...")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    # Compute similarity of each language with English per layer
    similarities_with_english = []  # [num_languages-1, num_layers]
    langs_without_en = [lang for lang in args.languages if lang != 'en']
    
    for lang in langs_without_en:
        cos_sim = cosine_similarity_layers(lang_vectors[lang], lang_vectors['en'])
        similarities_with_english.append(cos_sim.numpy())
    
    similarities_with_english = np.array(similarities_with_english)

    n_colors_needed = len(langs_without_en)
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

    for i, lang in enumerate(langs_without_en):
        ax.plot(layers, similarities_with_english[i], 
                marker='o' if show_markers else None,
                label=f'{lang.upper()}', 
                linewidth=line_width,
                markersize=marker_size,
                color=colors[i % len(colors)],
                alpha=0.75)

    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Cosine Similarity with English', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)

    y_min = max(-1.0, similarities_with_english.min() - 0.05)
    y_max = min(1.0, similarities_with_english.max() + 0.05)
    ax.set_ylim([y_min, y_max])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_cols = min(4, max(1, len(langs_without_en) // 10))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=legend_cols,
              fontsize=7, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "similarity_with_english_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(Path(args.output_dir) / "similarity_with_english_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity with English plot to {Path(args.output_dir) / 'similarity_with_english_plot.png'}")

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
                        cos_sim = cosine_similarity_layers(lang_vectors[lang1], lang_vectors[lang2])
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
                        vec1_layer = lang_vectors[lang1][layer_idx].unsqueeze(0)
                        vec2_layer = lang_vectors[lang2][layer_idx].unsqueeze(0)
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
                        lang_vectors[lang1][start_layer:end_layer+1],
                        lang_vectors[lang2][start_layer:end_layer+1],
                    )
                    similarity_matrix[i, j] = cos_sim.mean().item()
                    similarity_matrix[j, i] = cos_sim.mean().item()
    
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

results_path = Path(args.output_dir) / "steering_vector_analysis_results.pt"
torch.save(results, results_path)
print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("ALL PLOTS AND RESULTS GENERATED SUCCESSFULLY!")
print("="*60)