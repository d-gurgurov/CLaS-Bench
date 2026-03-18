import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def parse_results_txt(filepath):
    """Parse a results text file and extract per-target-language metrics
    
    Returns:
        dict: {
            'method_name': str,
            'overall': {'forcing': float, 'judge': float, 'harmonic_mean': float},
            'per_target_language': {lang: {'forcing': float, 'judge': float, 'harmonic_mean': float}}
        }
    """
    result = {
        'method_name': '',
        'overall': {},
        'per_target_language': {}
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract method name
    method_match = re.search(r'RESULTS FOR METHOD:\s*(.+)', content)
    if method_match:
        result['method_name'] = method_match.group(1).strip()
    
    # Extract overall results
    overall_section = re.search(r'OVERALL RESULTS\s*\n=+\n(.*?)(?=\n=|\Z)', content, re.DOTALL)
    if overall_section:
        overall_text = overall_section.group(1)
        forcing_match = re.search(r'Language Forcing Success Rate:\s*([\d.]+)%', overall_text)
        judge_match = re.search(r'Judge Relevance Score:\s*([\d.]+)%', overall_text)
        harmonic_match = re.search(r'Steering Score \(Harmonic Mean\):\s*([\d.]+)%', overall_text)
        
        result['overall'] = {
            'forcing': float(forcing_match.group(1)) if forcing_match else 0.0,
            'judge': float(judge_match.group(1)) if judge_match else 0.0,
            'harmonic_mean': float(harmonic_match.group(1)) if harmonic_match else 0.0
        }
    
    # Extract per target language results
    target_section = re.search(r'PER TARGET LANGUAGE RESULTS\s*\n=+\n.*?\n-+\n(.*?)(?=\n=|\Z)', content, re.DOTALL)
    if target_section:
        lines = target_section.group(1).strip().split('\n')
        for line in lines:
            if line.strip():
                # Parse line like: "German               85.00%        75.00%         79.41%"
                parts = line.split()
                if len(parts) >= 4:
                    lang = parts[0]
                    # Extract percentages (remove % sign)
                    forcing = float(parts[1].replace('%', ''))
                    judge = float(parts[2].replace('%', ''))
                    harmonic = float(parts[3].replace('%', ''))
                    
                    result['per_target_language'][lang] = {
                        'forcing': forcing,
                        'judge': judge,
                        'harmonic_mean': harmonic
                    }
    
    return result


def create_comparison_plot(methods_data, output_dir, metric='harmonic_mean'):
    """Create a wide ACL-style plot comparing methods across languages
    
    Args:
        methods_data: list of dicts from parse_results_txt
        output_dir: output directory for plots
        metric: which metric to plot ('forcing', 'judge', 'harmonic_mean')
    """
    
    # Set style for ACL publication quality
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
    
    # Collect all languages across all methods
    all_languages = set()
    for data in methods_data:
        all_languages.update(data['per_target_language'].keys())
    
    # Sort languages alphabetically
    languages = sorted(all_languages)
    
    # Define markers and colors for each method
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))
    
    # Create wide figure for ACL (full page width)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))  # ACL full width, shorter height
    
    x = np.arange(len(languages))
    width = 0.8 / len(methods_data)  # Width for grouped bars or offset for lines
    
    for i, data in enumerate(methods_data):
        method_name = data['method_name'] or f"Method {i+1}"
        
        # Get values for each language
        y_values = []
        for lang in languages:
            if lang in data['per_target_language']:
                y_values.append(data['per_target_language'][lang][metric])
            else:
                y_values.append(0)  # Missing data
        
        # Plot with offset for visibility (no connecting lines)
        offset = (i - len(methods_data) / 2 + 0.5) * width
        ax.scatter(x + offset, y_values, 
                   marker=markers[i % len(markers)], 
                   s=50,  # marker size
                   label=method_name, 
                   color=colors[i],
                   alpha=0.9,
                   edgecolors='black',
                   linewidths=0.5)
    
    # Configure axes
    ax.set_xlabel("Target Language", fontsize=11)
    
    metric_labels = {
        'forcing': 'Success Rate (%)',
        'judge': 'Judge Score (%)',
        'harmonic_mean': 'Steering Score (%)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
    
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend outside plot on right side
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, 
              frameon=True, edgecolor='black')
    
    # ACL-style frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')
    
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    metric_name = metric.replace('_', '-')
    pdf_path = f"{output_dir}/methods_comparison_{metric_name}.pdf"
    png_path = f"{output_dir}/methods_comparison_{metric_name}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved methods_comparison_{metric_name}.pdf and .png")


def create_bar_plot(methods_data, output_dir, metric='harmonic_mean'):
    """Create a grouped bar plot comparing methods across languages
    
    Args:
        methods_data: list of dicts from parse_results_txt
        output_dir: output directory for plots
        metric: which metric to plot ('forcing', 'judge', 'harmonic_mean')
    """
    
    # Set style for ACL publication quality
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
    })
    
    # Collect all languages across all methods
    all_languages = set()
    for data in methods_data:
        all_languages.update(data['per_target_language'].keys())
    
    # Sort languages alphabetically
    languages = sorted(all_languages)
    
    # Colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))
    
    # Create wide figure for ACL (full page width)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    
    x = np.arange(len(languages))
    n_methods = len(methods_data)
    width = 0.8 / n_methods
    
    for i, data in enumerate(methods_data):
        method_name = data['method_name'] or f"Method {i+1}"
        
        # Get values for each language
        y_values = []
        for lang in languages:
            if lang in data['per_target_language']:
                y_values.append(data['per_target_language'][lang][metric])
            else:
                y_values.append(0)
        
        # Calculate bar position
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, y_values, width * 0.9, label=method_name, color=colors[i], alpha=0.85)
    
    # Configure axes
    ax.set_xlabel("Target Language", fontsize=11)
    
    metric_labels = {
        'forcing': 'Success Rate (%)',
        'judge': 'Judge Score (%)',
        'harmonic_mean': 'Steering Score (%)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
    
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=True, edgecolor='black')
    
    # ACL-style frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')
    
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
    
    plt.tight_layout()
    
    # Save
    metric_name = metric.replace('_', '-')
    pdf_path = f"{output_dir}/methods_comparison_bar_{metric_name}.pdf"
    png_path = f"{output_dir}/methods_comparison_bar_{metric_name}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved methods_comparison_bar_{metric_name}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot steering metrics comparison across methods and languages"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
        required=True,
        help="List of result text files to compare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['line', 'bar', 'both'],
        default='both',
        help="Type of plot to create"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=['forcing', 'judge', 'harmonic_mean'],
        default=['harmonic_mean'],
        help="Which metrics to plot"
    )
    parser.add_argument(
        "--latex_table",
        action='store_true',
        help="Generate LaTeX table"
    )
    parser.add_argument(
        "--metric_for_table",
        type=str,
        choices=['forcing', 'judge', 'harmonic_mean'],
        default='harmonic_mean',
        help="Which metric to use for the LaTeX table"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse all input files
    methods_data = []
    for filepath in args.input_files:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Loading: {filepath}")
        data = parse_results_txt(filepath)
        if data['per_target_language']:
            methods_data.append(data)
            print(f"  Method: {data['method_name']}, Languages: {len(data['per_target_language'])}")
        else:
            print(f"  Warning: No per-language data found")
    
    if not methods_data:
        print("Error: No valid data found!")
        return
    
    print(f"\nLoaded {len(methods_data)} methods")
    
    # Create plots for each metric
    for metric in args.metrics:
        print(f"\nCreating plots for metric: {metric}")
        
        if args.plot_type in ['line', 'both']:
            create_comparison_plot(methods_data, output_dir, metric)
        
        if args.plot_type in ['bar', 'both']:
            create_bar_plot(methods_data, output_dir, metric)
    
    # Generate LaTeX table if requested
    if args.latex_table:
        print(f"\nGenerating LaTeX table for metric: {args.metric_for_table}")
        generate_latex_table(methods_data, output_dir, args.metric_for_table)
    
    print(f"\n✓ All outputs saved to: {output_dir}/")


def generate_latex_table(methods_data, output_dir, metric='harmonic_mean'):
    """Generate a LaTeX table with languages as rows and methods as columns
    
    Args:
        methods_data: list of dicts from parse_results_txt
        output_dir: output directory for the table
        metric: which metric to use ('forcing', 'judge', 'harmonic_mean')
    """
    
    # Collect all languages across all methods
    all_languages = set()
    for data in methods_data:
        all_languages.update(data['per_target_language'].keys())
    
    # Sort languages alphabetically
    languages = sorted(all_languages)
    
    # Get method names
    method_names = [data['method_name'] or f"Method {i+1}" for i, data in enumerate(methods_data)]
    
    # Method display names and symbols for LaTeX header
    method_display = {
        'baseline-I': r'$\mathcal{E}$ Base.-I',
        'baseline-II': r'$\mathcal{E}$ Base.-II',
        'LAPE': r'$\odot$ LAPE',
        'diffmean': r'$\vec{\Delta}$ DiffM.',
        'probe': r'$\mathbf{w}$ Probe',
        'LDA': r'$\mathbf{v}$ LDA',
        'PCA': r'$\mathbf{u}$ PCA',
        'SAE-DiffMean': r'$\vec{\Delta}$ SAE-DM.',
    }
    
    # Cell colors for methods
    method_colors = {
        'baseline-I': r'\cellcolor{gray!20}',
        'baseline-II': r'\cellcolor{gray!20}',
        'LAPE': r'\cellcolor{orange!20}',
        'diffmean': r'\cellcolor{yellow!25}',
    }
    
    # Get display name for method
    def get_display_name(method_name):
        for key, display in method_display.items():
            if key.lower() in method_name.lower():
                return display
        return method_name.replace('_', r'\_').replace('-', '-')
    
    # Get cell color for method
    def get_color(method_name):
        for key, color in method_colors.items():
            if key.lower() in method_name.lower():
                return color
        return ''
    
    # Build the table
    n_methods = len(methods_data)
    
    # Create column specification: language column + method columns
    col_spec = 'l|' + 'c' * n_methods
    
    # Build header row with method names
    header_methods = [get_display_name(name) for name in method_names]
    header_row = "\\textbf{Lang.} & " + " & ".join(header_methods) + r" \\"
    
    # Build score matrix for finding best per language
    # scores_matrix[lang][method_idx] = score (full precision)
    scores_matrix = {}
    for lang in languages:
        scores_matrix[lang] = []
        for data in methods_data:
            score = data['per_target_language'].get(lang, {}).get(metric, 0.0)
            scores_matrix[lang].append(score)
    
    # Calculate method averages (full precision)
    method_avgs = []
    for m_idx, data in enumerate(methods_data):
        values = [data['per_target_language'].get(lang, {}).get(metric, 0.0) for lang in languages]
        # Only include non-zero values in average
        non_zero = [v for v in values if v > 0]
        avg = np.mean(non_zero) if non_zero else 0.0
        method_avgs.append(avg)
    
    # Find best average
    best_avg_idx = np.argmax(method_avgs)
    
    # Build data rows (one per language)
    data_rows = []
    for lang in languages:
        scores = scores_matrix[lang]
        
        # Find best score for this language (among non-zero)
        non_zero_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        if non_zero_scores:
            best_idx = max(non_zero_scores, key=lambda x: x[1])[0]
        else:
            best_idx = -1
        
        # Build row
        row_values = []
        for m_idx, score in enumerate(scores):
            color = get_color(method_names[m_idx])
            if score > 0:
                if m_idx == best_idx:
                    row_values.append(f"{color}\\textbf{{{score:.1f}}}")
                else:
                    row_values.append(f"{color}{score:.1f}")
            else:
                row_values.append(f"{color}-")
        
        row = f"{lang} & " + " & ".join(row_values) + r" \\"
        data_rows.append(row)
    
    # Build average row
    avg_values = []
    for m_idx, avg in enumerate(method_avgs):
        color = get_color(method_names[m_idx])
        if m_idx == best_avg_idx:
            avg_values.append(f"{color}\\textbf{{{avg:.1f}}}")
        else:
            avg_values.append(f"{color}{avg:.1f}")
    
    avg_row = "Avg. & " + " & ".join(avg_values) + r" \\"
    
    n_languages = len(languages)
    
    # Assemble full table
    latex_table = rf"""\begin{{table*}}[t!]
\centering
\small
\begin{{tabular}}{{{col_spec}}}
\toprule
{header_row}
\midrule
{chr(10).join(data_rows)}
\midrule
{avg_row}
\bottomrule
\end{{tabular}}
\caption{{Language steering scores (i.e. harmonic means of language forcing and output relevance scores) across all methods for {n_languages} ablation languages for \texttt{{Llama-3.1-8B-Instruct}}. Language steering score is a harmonic mean of language forcing success and output relevance.}}
\label{{tab:methods_languages}}
\end{{table*}}
"""
    
    # Save to file
    output_file = output_dir / f"methods_comparison_table_{metric}.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"✓ Saved LaTeX table to: {output_file}")
    
    # Also print to console
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(latex_table)


if __name__ == "__main__":
    main()