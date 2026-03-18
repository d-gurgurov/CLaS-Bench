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
parser.add_argument("-d", "--data_dir", type=str, default="data_aya", help="Directory containing PCA components")
parser.add_argument("-l", "--languages", type=str, nargs='+', default=languages, help="Language codes to compare")
parser.add_argument("-s", "--save_name", type=str, default="aya",  help="Save name (e.g., 'llama')")
parser.add_argument("-o", "--output_dir", type=str, default="plots_aya/pca", help="Output directory for plots and results")
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

def load_pca_results(data_dir, lang, save_name):
    """Load PCA results for a specific language."""
    pca_filepath = Path(data_dir) / f"pca.{lang}.{save_name}"
    
    if pca_filepath.exists():
        print(f"Loading {pca_filepath}")
        data = torch.load(pca_filepath)
        return data
    else:
        raise FileNotFoundError(f"PCA file not found: {pca_filepath}")

def get_language_families():
    """Define language families for grouping"""
    return {
        'Romance': ['fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'ron_Latn'],
        'Germanic': ['eng_Latn', 'deu_Latn', 'nld_Latn', 'swe_Latn', 'dan_Latn', 'nor_Latn', 'afr_Latn'],
        'Slavic': ['rus_Cyrl', 'pol_Latn', 'ukr_Cyrl', 'ces_Latn', 'slk_Latn', 'hrv_Latn', 'srp_Latn'],
        'Indo-Aryan': ['hin_Deva', 'ben_Beng', 'pan_Guru', 'urd_Arab'],
        'Sino-Tibetan': ['zho_Hans', 'mya_Mymr', 'bod_Tibt'],
        'Semitic': ['ara_Arab', 'heb_Hebr', 'amh_Ethi'],
        'Turkic': ['tur_Latn', 'kaz_Cyrl', 'uzb_Latn'],
        'Japonic': ['jpn_Jpan'],
        'Koreanic': ['kor_Hang'],
        'Austroasiatic': ['vie_Latn'],
        'Kra-Dai': ['tha_Thai'],
        'Austronesian': ['ind_Latn', 'tgl_Latn'],
        'Niger-Congo': ['swa_Latn'],
        'Iranian': ['fas_Arab'],
        'Kartvelian': ['kat_Geor'],
        'Hellenic': ['ell_Grek'],
    }

def get_non_latin_scripts():
    """Languages with non-Latin scripts that should have asterisks"""
    return {'ara_Arab', 'amh_Ethi', 'heb_Hebr', 'ben_Beng', 'asm_Beng', 'mya_Mymr', 
            'zho_Hans', 'bod_Tibt', 'ell_Grek', 'guj_Gujr', 'heb_Hebr', 'hin_Deva', 
            'mar_Deva', 'npi_Deva', 'jpn_Jpan', 'kan_Knda', 'kat_Geor', 'kaz_Cyrl', 
            'kor_Hang', 'mal_Mlym', 'mkd_Cyrl', 'ory_Orya', 'pan_Guru', 'fas_Arab', 
            'rus_Cyrl', 'srp_Cyrl', 'sin_Sinh', 'tam_Taml', 'tel_Telu', 'tha_Thai', 
            'ukr_Cyrl', 'urd_Arab', 'bel_Cyrl', 'bul_Cyrl'}

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

# Load all PCA results
print("Loading PCA results for all languages...")
pca_data = {}
all_pca_components = {}
all_variance_explained = {}

for lang in args.languages:
    data = load_pca_results(args.data_dir, lang, args.save_name)
    pca_data[lang] = data
    
    # Extract PCA components and variance explained
    pca_components = data['pca_components']  # [num_layers, n_components, hidden_size]
    variance_explained = data['variance_explained']  # [num_layers, n_components]
    
    all_pca_components[lang] = pca_components
    all_variance_explained[lang] = variance_explained

num_layers = all_pca_components[args.languages[0]].shape[0]
n_components = all_pca_components[args.languages[0]].shape[1]
print(f"Number of layers: {num_layers}")
print(f"Number of PCA components: {n_components}")

# ============================================================================
# PLOT 1: Cosine Similarity Between All Languages (Per-Layer)
# ============================================================================
print("\nGenerating cosine similarity between language PCA components plot...")

def compute_cosine_similarity_components(comp1, comp2):
    """Compute cosine similarity between PCA components.
    
    Args:
        comp1: [n_components, hidden_size]
        comp2: [n_components, hidden_size]
    
    Returns:
        Similarity score (0-1) based on component alignment
    """
    # Flatten components to 1D for comparison
    comp1_flat = comp1.reshape(-1)
    comp2_flat = comp2.reshape(-1)
    
    comp1_norm = comp1_flat / (torch.norm(comp1_flat) + 1e-8)
    comp2_norm = comp2_flat / (torch.norm(comp2_flat) + 1e-8)
    
    cos_sim = torch.dot(comp1_norm, comp2_norm)
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
            
            # Compute similarity for each layer
            layer_sims = []
            for layer_idx in range(num_layers):
                comp1 = all_pca_components[lang1][layer_idx]  # [n_components, hidden_size]
                comp2 = all_pca_components[lang2][layer_idx]
                
                sim = compute_cosine_similarity_components(comp1, comp2)
                layer_sims.append(sim.cpu().item())
            
            all_similarities.append(layer_sims)
            
            avg_sim = np.mean(layer_sims)
            std_sim = np.std(layer_sims)
            pairwise_avg_similarities[pair_name] = avg_sim
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
plt.savefig(Path(args.output_dir) / "pca_cosine_similarity_per_layer.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "pca_cosine_similarity_per_layer.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved cosine similarity plot to {Path(args.output_dir) / 'pca_cosine_similarity_per_layer.png'}")

# ============================================================================
# PLOT 2: Variance Explained Per Layer (Averaged Across Languages)
# ============================================================================
print("\nGenerating variance explained plot...")

# Average cumulative variance across all languages
cumulative_var_per_layer = []
for lang in args.languages:
    var_exp = all_variance_explained[lang]  # [num_layers, n_components]
    cumulative_var = var_exp.sum(dim=1)  # Sum across components for each layer
    cumulative_var_per_layer.append(cumulative_var.cpu().numpy())

cumulative_var_per_layer = np.array(cumulative_var_per_layer)  # [num_langs, num_layers]
mean_cumulative_var = cumulative_var_per_layer.mean(axis=0)
std_cumulative_var = cumulative_var_per_layer.std(axis=0)

fig, ax = plt.subplots(figsize=(3.5, 2.5))

ax.plot(layers, mean_cumulative_var, color='#2E86AB', linewidth=1.5, label='Cumulative Variance')
ax.fill_between(layers, 
                 mean_cumulative_var - std_cumulative_var,
                 mean_cumulative_var + std_cumulative_var,
                 color='#2E86AB', alpha=0.2)

ax.set_xlabel('Layer', fontsize=10, fontweight='normal')
ax.set_ylabel('Variance Explained', fontsize=10, fontweight='normal')
ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(Path(args.output_dir) / "pca_variance_explained_per_layer.pdf", dpi=300, bbox_inches='tight')
plt.savefig(Path(args.output_dir) / "pca_variance_explained_per_layer.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved variance explained plot to {Path(args.output_dir) / 'pca_variance_explained_per_layer.png'}")

# ============================================================================
# PLOT 2B: Components Needed to Explain Variance Thresholds (50%, 60%, 70%)
# ============================================================================
print("\nGenerating components needed for variance thresholds...")

def compute_components_for_threshold(variance_explained, threshold):
    """
    Compute number of components needed to reach a variance threshold.
    
    Args:
        variance_explained: [num_layers, n_components]
        threshold: float between 0 and 1 (e.g., 0.5 for 50%)
    
    Returns:
        [num_layers] array with components needed per layer
    """
    num_layers = variance_explained.shape[0]
    components_needed = np.zeros(num_layers)
    
    for layer_idx in range(num_layers):
        cumsum = torch.cumsum(variance_explained[layer_idx], dim=0)
        # Find first index where cumsum >= threshold
        idx = (cumsum >= threshold).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            components_needed[layer_idx] = idx[0].item() + 1  # +1 because indexing starts at 0
        else:
            components_needed[layer_idx] = n_components  # All components needed
    
    return components_needed

# Compute components needed per language for each threshold
thresholds = [0.5, 0.6, 0.7]
threshold_data = {t: [] for t in thresholds}

for lang in args.languages:
    var_exp = all_variance_explained[lang]  # [num_layers, n_components]
    for threshold in thresholds:
        comps_needed = compute_components_for_threshold(var_exp, threshold)
        threshold_data[threshold].append(comps_needed)

# Convert to numpy arrays [num_langs, num_layers]
for threshold in thresholds:
    threshold_data[threshold] = np.array(threshold_data[threshold])

# Create separate plots for each threshold
colors = ['#2E86AB', '#A23B72', '#F18F01']

for idx, threshold in enumerate(thresholds):
    data = threshold_data[threshold]  # [num_langs, num_layers]
    mean_comps = data.mean(axis=0)
    std_comps = data.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(layers, mean_comps, color=colors[idx], linewidth=1.5)
    ax.fill_between(layers,
                     mean_comps - std_comps,
                     mean_comps + std_comps,
                     color=colors[idx], alpha=0.2)
    
    ax.set_xlabel('Layer', fontsize=10, fontweight='normal')
    ax.set_ylabel('Components Needed', fontsize=10, fontweight='normal')
    ax.set_title(f'{int(threshold*100)}% Variance Explained', fontsize=10, fontweight='normal')
    ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    # Set y-axis to accommodate the maximum value in the data with some padding
    max_val = (mean_comps + std_comps).max()
    ax.set_ylim([0, max_val * 1.1])  # 10% padding above max value
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')
    
    # Print stats for this threshold
    print(f"\n{int(threshold*100)}% Variance Threshold:")
    print(f"  Mean components needed (avg over layers): {mean_comps.mean():.1f}")
    print(f"  Std dev: {mean_comps.std():.1f}")
    print(f"  Min: {mean_comps.min():.1f}, Max: {mean_comps.max():.1f}")
    
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / f"pca_components_for_{int(threshold*100)}percent_variance.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(Path(args.output_dir) / f"pca_components_for_{int(threshold*100)}percent_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {int(threshold*100)}% threshold plot to {Path(args.output_dir) / f'pca_components_for_{int(threshold*100)}percent_variance.png'}")

print(f"\nAll threshold plots saved!")

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
                layer_sims = []
                for layer_idx in range(num_layers):
                    comp1 = all_pca_components[lang1][layer_idx]
                    comp2 = all_pca_components[lang2][layer_idx]
                    
                    sim = compute_cosine_similarity_components(comp1, comp2)
                    layer_sims.append(sim.cpu().item())
                
                avg_sim = np.mean(layer_sims)
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
    
    plt.savefig(Path(args.output_dir) / "pca_similarity_heatmap_avg_layers.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(Path(args.output_dir) / "pca_similarity_heatmap_avg_layers.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved similarity heatmap to {Path(args.output_dir) / 'pca_similarity_heatmap_avg_layers.png'}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

summary_stats = {}

for lang in args.languages:
    var_exp = all_variance_explained[lang]  # [num_layers, n_components]
    
    # Cumulative variance per layer
    cumulative_var = var_exp.sum(dim=1).cpu().numpy()
    
    # Per-component variance (average across layers)
    per_component_var = var_exp.mean(dim=0).cpu().numpy()
    
    stats = {
        'mean_cumulative_variance': float(np.mean(cumulative_var)),
        'std_cumulative_variance': float(np.std(cumulative_var)),
        'min_cumulative_variance': float(np.min(cumulative_var)),
        'max_cumulative_variance': float(np.max(cumulative_var)),
        'best_layer_variance': int(np.argmax(cumulative_var)),
        'mean_per_component_variance': float(np.mean(per_component_var)),
        'top_component_variance': float(per_component_var[0]),
        'num_components': int(n_components),
    }
    
    summary_stats[lang] = stats
    
    print(f"\n{lang.upper()}:")
    print(f"  Cumulative Variance Explained: {stats['mean_cumulative_variance']:.4f} ± {stats['std_cumulative_variance']:.4f}")
    print(f"    Range: [{stats['min_cumulative_variance']:.4f}, {stats['max_cumulative_variance']:.4f}]")
    print(f"    Best layer: {stats['best_layer_variance']}")
    print(f"  Mean per-component variance: {stats['mean_per_component_variance']:.4f}")
    print(f"  Top component variance: {stats['top_component_variance']:.4f}")

print(f"\nOverall:")
print(f"  Mean Cosine Similarity (Pairwise): {overall_mean_sim:.4f} ± {overall_std_sim:.4f}")
print(f"  Number of PCA components: {n_components}")

# Save results to JSON
results_path = Path(args.output_dir) / "pca_analysis_summary.json"
with open(results_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)