import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def harmonic_mean(a, b):
    """Compute harmonic mean of two values"""
    if a == 0 or b == 0:
        return 0.0
    return 2 * (a * b) / (a + b)


def extract_judge_statistics(eval_dir):
    """Extract judge scores from judge_analysis.json
    
    Returns:
        float: Average judge score across all language pairs (normalized to 0-100)
    """
    eval_dir = Path(eval_dir)
    
    if not eval_dir.exists():
        print(f"Warning: Evaluation directory not found: {eval_dir}")
        return 0.0
    
    # Look for judge_analysis.json
    judge_analysis_file = eval_dir / "judge_analysis.json"
    
    if not judge_analysis_file.exists():
        print(f"Warning: judge_analysis.json not found in {eval_dir}")
        return 0.0
    
    try:
        data = load_json(judge_analysis_file)
        
        # Extract overall average judge score from summary
        summary = data.get('summary', {})
        overall_avg_score = summary.get('overall_average_judge_score')
        total_evaluations = summary.get('total_evaluations')
        
        if overall_avg_score is None:
            print(f"Warning: overall_average_judge_score not found in {judge_analysis_file}")
            return 0.0
        
        # Convert 0-2 scale to 0-100 percentage
        avg_rating_pct = (overall_avg_score / 2.0) * 100  # Normalize to 0-100
        
        print(f"  Found overall judge score: {avg_rating_pct:.1f}% ({total_evaluations} evaluations)")
        
        return avg_rating_pct
        
    except Exception as e:
        print(f"Error processing {judge_analysis_file}: {e}")
        return 0.0


def extract_forcing_statistics(forcing_file):
    """Extract forcing success rate from analysis_results.json
    
    Returns:
        float: Average success rate across all language pairs (0-100)
    """
    forcing_file = Path(forcing_file)
    
    if not forcing_file.exists():
        print(f"Warning: Forcing results file not found: {forcing_file}")
        return 0.0
    
    try:
        data = load_json(forcing_file)
        
        # Extract language pair statistics
        pair_stats = data.get("language_pairs", {})
        
        if not pair_stats:
            print(f"Warning: No language pair statistics in {forcing_file}")
            return 0.0
        
        success_rates = [p["success_rate"] for p in pair_stats.values()]
        avg_success_rate = np.mean(success_rates) if success_rates else 0.0
        
        print(f"  Found {len(pair_stats)} language pairs: {avg_success_rate:.1f}%")
        
        return avg_success_rate
        
    except Exception as e:
        print(f"Error processing {forcing_file}: {e}")
        return 0.0


def parse_directory_path(dirpath):
    """Extract neuron percentage, config type, and method from directory path
    
    Expected format: .../lape-{method}/{MODEL}_{percent}/{config_type}_{method}/...
    Returns: (neuron_percent, config_type, method) or (None, None, None) if not found
    """
    import re
    
    path_str = str(dirpath)
    
    # Match the neuron percentage: look for _\d+ followed by / or \ or end of path
    # This gets the last number after underscore which is the percentage
    percent_matches = re.findall(r'_(\d+)(?:/|\\|$)', path_str)
    neuron_percent = int(percent_matches[-1]) if percent_matches else None
    
    # Match config types: deactivate_activate or activate (with optional _additive or _replacement suffix)
    config_match = re.search(r'(deactivate_activate|activate)(?:_(additive|replacement))?(?:/|\\|$)', path_str, re.IGNORECASE)
    config_type = config_match.group(1) if config_match else None
    
    # Extract method from path: look for lape-additive or lape-replacement in path
    method_match = re.search(r'lape-(additive|replacement)', path_str, re.IGNORECASE)
    if method_match:
        method = method_match.group(1).lower()
    else:
        # Fallback: check if _additive or _replacement is in the config folder name
        if config_match and config_match.group(2):
            method = config_match.group(2).lower()
        else:
            method = None
    
    return neuron_percent, config_type, method


def find_directory_data(input_dirs):
    """Find forcing and judge evaluation files in directories
    
    Returns:
        list: List of tuples (directory_path, neuron_percent, config_type, method)
    """
    directory_data = []
    
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        
        neuron_percent, config_type, method = parse_directory_path(dir_path)
        
        if neuron_percent is not None and config_type is not None and method is not None:
            directory_data.append((dir_path, neuron_percent, config_type, method))
        else:
            print(f"Warning: Could not parse neuron_percent/config_type/method from path: {dir_path}")
            print(f"  Parsed: neuron_percent={neuron_percent}, config_type={config_type}, method={method}")
    
    return directory_data


def aggregate_results(input_dirs):
    """Aggregate metrics across multiple directories
    
    Returns:
        dict: {(neuron_percent, config_type, method): {'success_rate': float, 'judge_score': float, 'harmonic_mean': float}}
    """
    # Find all directories
    directory_data = find_directory_data(input_dirs)
    
    if not directory_data:
        print("Error: No valid directories found!")
        return {}
    
    print(f"Found {len(directory_data)} directories to process\n")
    
    aggregated = defaultdict(lambda: {'success_rates': [], 'judge_scores': []})
    
    for dir_path, neuron_percent, config_type, method in directory_data:
        print(f"Processing {neuron_percent}% neurons, config={config_type}, method={method}: {dir_path}")
        
        try:
            # Look for analysis_results.json for forcing statistics
            forcing_file = dir_path / "analysis_results.json"
            if forcing_file.exists():
                success_rate = extract_forcing_statistics(forcing_file)
            else:
                print(f"  Warning: No analysis_results.json found")
                success_rate = 0.0
            
            # Extract judge statistics from judge_analysis.json
            judge_score = extract_judge_statistics(dir_path)
            
            if success_rate > 0 or judge_score > 0:
                aggregated[(neuron_percent, config_type, method)]['success_rates'].append(success_rate)
                aggregated[(neuron_percent, config_type, method)]['judge_scores'].append(judge_score)
                print(f"  ✓ SR={success_rate:.1f}%, JS={judge_score:.1f}%")
            else:
                print(f"  ✗ No valid data found")
            
        except Exception as e:
            print(f"  ✗ Error processing directory: {e}")
            continue
        
        print()
    
    # Compute averages and harmonic mean
    final_results = {}
    for (neuron_percent, config_type, method), metrics in aggregated.items():
        avg_success = np.mean(metrics['success_rates']) if metrics['success_rates'] else 0.0
        avg_judge = np.mean(metrics['judge_scores']) if metrics['judge_scores'] else 0.0
        h_mean = harmonic_mean(avg_success, avg_judge)
        
        final_results[(neuron_percent, config_type, method)] = {
            'success_rate': avg_success,
            'judge_score': avg_judge,
            'harmonic_mean': h_mean,
            'n_dirs': len(metrics['success_rates'])
        }
    
    return final_results


def get_config_label(config_type, method):
    """Map config type and method to display label for plots"""
    config_labels = {
        'activate': 'Act.',
        'deactivate_activate': 'Act.+Deact.'
    }
    method_labels = {
        'additive': 'Add.',
        'replacement': 'Repl.'
    }
    config_label = config_labels.get(config_type, config_type)
    method_label = method_labels.get(method, method)
    return f"{config_label} ({method_label})"


def create_acl_plots(results, output_dir):
    """Create three ACL-style plots: success rate, judge score, and steering score"""
    
    # Set style for ACL publication quality
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
    })
    
    # Organize data by (config_type, method) combinations
    config_method_combos = sorted(set((config, method) for _, config, method in results.keys()))
    neuron_percents = sorted(set(percent for percent, _, _ in results.keys()))
    
    # Define colors for different combinations using viridis colormap (same as before)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(config_method_combos)))
    
    # Plot configurations
    plot_configs = [
        {
            'metric': 'success_rate',
            'ylabel': 'Success Rate (%)',
            'title': 'Language Forcing Success Rate',
            'filename': 'success_rate_by_neuron_percent'
        },
        {
            'metric': 'judge_score',
            'ylabel': 'Judge Score (%)',
            'title': 'Judge Relevance Score',
            'filename': 'judge_score_by_neuron_percent'
        },
        {
            'metric': 'harmonic_mean',
            'ylabel': 'Steering Score (%)',
            'title': 'Steering Score (Harmonic Mean)',
            'filename': 'steering_score_by_neuron_percent'
        }
    ]
    
    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # ACL single-column width
        
        for i, (config_type, method) in enumerate(config_method_combos):
            # Get data for this config type and method
            y_values = []
            x_values = []
            
            for neuron_percent in neuron_percents:
                if (neuron_percent, config_type, method) in results:
                    x_values.append(neuron_percent)
                    y_values.append(results[(neuron_percent, config_type, method)][config['metric']])
            
            if x_values:
                ax.plot(x_values, y_values, marker='o', markersize=4,
                       label=get_config_label(config_type, method), 
                       color=colors[i], linewidth=1.5)
        
        if config['metric'] == 'harmonic_mean':
            ax.set_xlabel("Neurons Changed (%)", fontsize=10, fontweight='normal')  
        # ax.set_ylabel(config['ylabel'], fontsize=10, fontweight='normal')
        ax.set_ylim(0, 100)
        ax.set_xticks(neuron_percents)
        ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=7)
        
        # ACL-style frame
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor('black')
        
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
        
        plt.tight_layout()
        
        # Save as PDF and PNG
        pdf_path = f"{output_dir}/{config['filename']}.pdf"
        png_path = f"{output_dir}/{config['filename']}.png"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {config['filename']}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot steering metrics across neuron percentages, config types, and methods"
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of directories containing results (e.g., generation/lape-{additive,replacement}/MODEL_*/[deactivate_activate|activate]_{additive,replacement})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(args.input_dirs)} directories...\n")
    print("="*80)
    
    # Aggregate results
    results = aggregate_results(args.input_dirs)
    
    if not results:
        print("\nError: No valid results found!")
        return
    
    print("="*80)
    print(f"\nFound data for {len(results)} (neuron_percent, config_type, method) combinations")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Neurons %':<12} {'Config Type':<20} {'Method':<12} {'Success %':<12} {'Judge %':<12} {'Steering %':<12} {'N Dirs':<8}")
    print("-"*90)
    
    for (neuron_percent, config_type, method), metrics in sorted(results.items()):
        print(f"{neuron_percent:<12} {config_type:<20} {method:<12} {metrics['success_rate']:>9.1f}%   "
              f"{metrics['judge_score']:>9.1f}%   {metrics['harmonic_mean']:>10.1f}%   "
              f"{metrics['n_dirs']:>6}")
    
    # Create plots
    print("\n" + "="*80)
    print("Generating ACL-style plots...")
    print("="*80)
    create_acl_plots(results, output_dir)
    
    # Save aggregated results
    output_json = output_dir / "aggregated_results.json"
    output_data = {
        'results': {
            f"neurons_{neuron_percent}_config_{config_type}_method_{method}": metrics
            for (neuron_percent, config_type, method), metrics in results.items()
        },
        'summary': {
            'total_combinations': len(results),
            'neuron_percents': sorted(set(percent for percent, _, _ in results.keys())),
            'config_types': sorted(set(config for _, config, _ in results.keys())),
            'methods': sorted(set(method for _, _, method in results.keys()))
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Aggregated results saved to: {output_json}")
    print(f"✓ All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()