import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="activation_mask/aya-1")
parser.add_argument("--output_path", type=str, default="plots_aya/neurons")
parser.add_argument("--font_size", type=int, default=8, help="Font size for heatmap annotations")
parser.add_argument("--figure_size", type=int, default=20, help="Figure size (will be square)")

global args
args = parser.parse_args()

# Set style for ACL publication quality
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

# Output directory
os.makedirs(args.output_path, exist_ok=True)

# Language codes used (same order as input)
langs = [
    "en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", #  "af", "tl", "ur", "bn",
    "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk",
    "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"
]

# Manually curated colors for maximum distinction and readability
distinct_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange  
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d3',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
    '#8B4513'   # saddle brown
]

# Create language to color mapping
lang_colors = {}
for i, lang in enumerate(langs):
    lang_colors[lang] = distinct_colors[i % len(distinct_colors)]

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

# Load activation mask
print(f"Loading activation mask from: {args.input_path}")
final_indice = torch.load(f"{args.input_path}")

num_languages = len(final_indice)
num_layers = len(final_indice[0])

print(f"Loaded data for {num_languages} languages and {num_layers} layers")

# Build sets of (layer, neuron) pairs per language
lang_neuron_sets = []
for lang_index in range(num_languages):
    neuron_set = set()
    for layer, heads in enumerate(final_indice[lang_index]):
        for head in heads.tolist():
            neuron_set.add((layer, head))
    lang_neuron_sets.append(neuron_set)

print("Built neuron sets for all languages")

# Get language families and order languages
lang_families = get_language_families()
ordered_languages = order_languages_by_family(langs, lang_families)
ordered_indices = [langs.index(lang) for lang in ordered_languages]

print(f"Language order: {ordered_languages}")
print(f"Number of languages: {len(ordered_languages)}")

# === Plot 1: Family-Grouped Overlap Heatmap ===
print("Creating family-grouped overlap heatmap...")

# Get the actual number of languages we're working with
actual_num_languages = len(ordered_languages)
print(f"Creating overlap matrix for {actual_num_languages} languages")

# Create overlap matrix with ordered languages (use actual count)
overlap_matrix = np.zeros((actual_num_languages, actual_num_languages), dtype=int)
for i, lang_i_idx in enumerate(ordered_indices):
    for j, lang_j_idx in enumerate(ordered_indices):
        intersection = len(lang_neuron_sets[lang_i_idx] & lang_neuron_sets[lang_j_idx])
        overlap_matrix[i, j] = intersection

# Store diagonal values (total neurons per language) for later use
diagonal_neuron_counts = np.diag(overlap_matrix).copy()

# Create formatted labels with asterisks
formatted_labels = [format_language_label(lang) for lang in ordered_languages]

# Improved dynamic sizing based on number of languages
base_size = args.figure_size
fig_size = max(base_size, actual_num_languages * 0.35)  # Increased scaling for better readability
annotation_font_size = max(6, 9 - actual_num_languages // 20)  # Smaller annotations for many languages
tick_font_size = max(10, 9 - actual_num_languages // 15)  # Better scaling for tick labels

print(f"Creating heatmap with size {fig_size}x{fig_size}, annotation font size {annotation_font_size}")

fig, ax = plt.subplots(figsize=(fig_size, fig_size))

# Determine whether to show annotations based on matrix size
show_annotations = actual_num_languages <= 40  # More conservative threshold for readability

# Create custom annotation array that shows diagonal values prominently
if show_annotations:
    annot_array = overlap_matrix.astype(str)
    # Make diagonal entries bold by wrapping in special formatting
    for i in range(actual_num_languages):
        annot_array[i, i] = f"{overlap_matrix[i, i]}"
else:
    annot_array = False

sns.heatmap(
    overlap_matrix,
    xticklabels=formatted_labels,
    yticklabels=formatted_labels,
    cmap="YlOrRd",  # ACL-style: Yellow-Orange-Red colormap
    annot=annot_array if show_annotations else False,
    fmt="" if show_annotations else None,
    cbar=True,
    linewidths=0.05 if actual_num_languages > 50 else 0.2,  # Very thin lines for many languages
    annot_kws={"size": annotation_font_size, "weight": "normal"} if show_annotations else {},
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.7, "pad": 0.02}
)

# Highlight diagonal with bold text (only if not showing all annotations)
if not show_annotations:
    # For large matrices, optionally show only diagonal values
    for i in range(actual_num_languages):
        text_color = 'white' if overlap_matrix[i, i] > overlap_matrix.max() * 0.6 else 'black'
        ax.text(i + 0.5, i + 0.5, f'{overlap_matrix[i, i]}',
                ha='center', va='center',
                fontsize=max(4, annotation_font_size),
                fontweight='bold',
                color=text_color)
elif show_annotations:
    # Highlight diagonal when showing all annotations
    for i in range(actual_num_languages):
        ax.text(i + 0.5, i + 0.5, f'{overlap_matrix[i, i]}',
                ha='center', va='center',
                fontsize=annotation_font_size,
                fontweight='bold',
                color='white' if overlap_matrix[i, i] > overlap_matrix.max() * 0.6 else 'black')

# Add family separators and labels with improved positioning
add_family_separators_and_labels(ax, ordered_languages, lang_families, 'both')

# Customize colorbar with ACL style
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel("Overlap Count", fontsize=16, labelpad=10)
cbar.ax.tick_params(labelsize=12, width=0.8)
cbar.outline.set_linewidth(0.8)

# Improved tick label positioning and rotation
plt.xticks(rotation=90, ha="center", fontsize=tick_font_size)  # Vertical rotation for better spacing
plt.yticks(rotation=0, fontsize=tick_font_size)

# Add border around the heatmap with ACL-appropriate thickness
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
    spine.set_visible(True)

plt.tight_layout()
# Add extra padding to prevent label cutoff
plt.subplots_adjust(left=0.08, right=0.95, top=0.98, bottom=0.08)

plt.savefig(f"{args.output_path}/language_overlap_family_grouped.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(f"{args.output_path}/language_overlap_family_grouped.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Saved overlap heatmap with {num_languages} languages")

# === Plot 2: Cumulative Distribution Across All Languages ===
print("Creating cumulative distribution plot...")

# Calculate cumulative neuron counts across all languages
layer_counts_all = np.zeros(num_layers)
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_all[layer] += len(heads)

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # ACL single-column width
bars = ax.bar(range(num_layers), layer_counts_all, color='#2E86AB', 
               edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xlabel("Layer", fontsize=10, fontweight='normal')
ax.set_ylabel("Neuron Count", fontsize=10, fontweight='normal')
ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# ACL-style frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(f"{args.output_path}/cumulative_neuron_distribution.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{args.output_path}/cumulative_neuron_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# === Plot 3: Comparative Line Plot for All Languages ===
print("Creating comparative line plot...")

# Prepare data for line plot
layer_counts_by_lang = np.zeros((num_languages, num_layers))
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_by_lang[lang_index, layer] = len(heads)

# Use ACL two-column width
fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5))

# Use a colorblind-friendly palette for ACL
# Generate distinct colors using a combination of colormaps
n_colors_needed = len(langs)
if n_colors_needed <= 10:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
elif n_colors_needed <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
else:
    # For more colors, combine multiple colormaps
    colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
    colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
    colors = np.vstack([colors1, colors2, colors3])

# Plot lines for each language
line_width = 1.2 if num_languages <= 30 else 0.8
marker_size = 3 if num_languages <= 20 else 0
show_markers = num_languages <= 20

for lang_index, lang in enumerate(langs):
    ax.plot(range(num_layers), layer_counts_by_lang[lang_index], 
            marker='o' if show_markers else None,
            label=f'{lang.upper()}', 
            linewidth=line_width,
            markersize=marker_size,
            color=colors[lang_index % len(colors)],
            alpha=0.75)

ax.set_xlabel('Layer', fontsize=10)
ax.set_ylabel('Neuron Count', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-0.5, num_layers - 0.5)
ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)

# ACL-style frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# Create a legend with 4 columns to fit within plot height
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=4, 
          fontsize=7, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(f"{args.output_path}/comparative_line_plot.pdf", 
            dpi=300, bbox_inches='tight')
plt.savefig(f"{args.output_path}/comparative_line_plot.png", 
            dpi=300, bbox_inches='tight')
plt.close()

# === Plot 4: Small Multiples - Individual Language Distributions (6x6 Grid) ===
print("Creating small multiples plot...")

# 6x6 grid for small multiples
cols = 6
rows = 6

# ACL figure size - aim for full page width or two-column
subplot_size = 1.0  # Size of each subplot in inches
fig, axes = plt.subplots(rows, cols, figsize=(cols*subplot_size, rows*subplot_size))

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Find global max for consistent y-axis scaling
global_max = 0
for lang_index in range(num_languages):
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    global_max = max(global_max, max(neuron_counts))

# Use consistent color for all subplots (ACL style)
bar_color = '#2E86AB'

# Create individual plots
for lang_index, lang in enumerate(langs):
    ax = axes_flat[lang_index]
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    
    # Create bar plot with consistent ACL styling
    bars = ax.bar(range(num_layers), neuron_counts, color=bar_color, 
                  edgecolor='black', linewidth=0.3, alpha=0.85)
    
    # Formatting
    ax.set_title(format_language_label(lang), fontsize=8, fontweight='normal', pad=2)
    ax.set_ylim(0, global_max * 1.1)
    ax.set_xticks(range(0, num_layers, max(1, num_layers//4)))
    ax.tick_params(axis='both', which='major', labelsize=6, width=0.5)
    ax.grid(axis='y', alpha=0.3, linewidth=0.3)
    
    # ACL-style frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
# Add common axes labels
fig.text(0.5, -0.01, 'Layer', ha='center', va='center', fontsize=10)
fig.text(-0.01, 0.5, 'Neuron Count', ha='center', va='center', 
         rotation=90, fontsize=10)

# Hide unused subplots
for i in range(len(langs), len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
# Increased spacing between subplots
plt.subplots_adjust(bottom=0.05, left=0.05, top=0.97, hspace=0.5, wspace=0.4)
plt.savefig(f"{args.output_path}/small_multiples_distribution.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{args.output_path}/small_multiples_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("All plots created successfully!")