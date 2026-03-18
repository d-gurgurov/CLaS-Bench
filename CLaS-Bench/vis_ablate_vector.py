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
    """Extract judge scores from evaluation files (with embedded judge_evaluation)
    
    Returns:
        dict: Language pair statistics with average judge scores (normalized to 0-100)
    """
    eval_dir = Path(eval_dir)
    
    if not eval_dir.exists():
        print(f"Warning: Evaluation directory not found: {eval_dir}")
        return {}
    
    # Find all JSON files
    eval_files = list(eval_dir.rglob("*.json"))
    
    if not eval_files:
        print(f"Warning: No JSON files found in {eval_dir}")
        return {}
    
    pair_statistics = {}
    files_processed = 0
    
    for eval_file in eval_files:
        try:
            data = load_json(eval_file)
            
            # Extract language pair
            source_lang = data.get("deactivate_language")
            target_lang = data.get("activate_language")
            
            # For script 2 (residual-based)
            if source_lang is None or source_lang == "none":
                source_lang = data.get("source_language")
            if target_lang is None:
                target_lang = data.get("target_language")
            
            if not source_lang or not target_lang:
                # Skip files without language info (might be analysis files)
                continue
            
            pair_key = f"{source_lang}_to_{target_lang}"
            
            # Extract judge ratings from results with embedded judge_evaluation
            results = data.get('results', [])
            
            if not results:
                continue
            
            valid_ratings = []
            for result in results:
                judge_eval = result.get('judge_evaluation', {})
                if judge_eval:
                    judge_score = judge_eval.get('judge_score')
                    if judge_score is not None and judge_score in [0, 1, 2]:
                        valid_ratings.append(judge_score)
            
            if not valid_ratings:
                continue
            
            # Compute statistics
            # Convert 0-2 scale to 0-100 percentage
            avg_rating = np.mean(valid_ratings)
            avg_rating_pct = (avg_rating / 2.0) * 100  # Normalize to 0-100
            
            pair_statistics[pair_key] = {
                "average_judge_score": avg_rating_pct,
                "raw_average_rating": avg_rating,
                "total": len(results),
                "valid_ratings": len(valid_ratings),
            }
            
            files_processed += 1
            
        except Exception as e:
            continue
    
    return pair_statistics


def extract_forcing_statistics(forcing_file):
    """Extract forcing success rates from analysis_results.json
    
    Returns:
        dict: Language pair statistics with success rates (0-100)
    """
    forcing_file = Path(forcing_file)
    
    if not forcing_file.exists():
        print(f"Warning: Forcing results file not found: {forcing_file}")
        return {}
    
    try:
        data = load_json(forcing_file)
        
        # Extract language pair statistics
        forcing_pairs = data.get("language_pairs", {})
        
        return forcing_pairs
        
    except Exception as e:
        print(f"Error processing {forcing_file}: {e}")
        return {}


def compute_combined_scores(forcing_pairs, judge_pairs):
    """Compute harmonic mean for matching language pairs
    
    Returns:
        tuple: (combined_scores dict, overall_success_rate, overall_judge_score, overall_harmonic_mean)
    """
    combined_scores = {}
    
    for pair in forcing_pairs:
        if pair in judge_pairs:
            forcing_rate = forcing_pairs[pair]["success_rate"]  # 0-100
            judge_score = judge_pairs[pair]["average_judge_score"]  # 0-100 (already normalized)
            
            h_mean = harmonic_mean(forcing_rate, judge_score)
            
            combined_scores[pair] = {
                "forcing_success_rate": forcing_rate,
                "judge_relevance_score": judge_score,
                "harmonic_mean": h_mean,
            }
    
    if not combined_scores:
        return {}, 0.0, 0.0, 0.0
    
    # Compute overall statistics
    all_forcing = [s["forcing_success_rate"] for s in combined_scores.values()]
    all_judge = [s["judge_relevance_score"] for s in combined_scores.values()]
    all_harmonic = [s["harmonic_mean"] for s in combined_scores.values()]
    
    overall_forcing = np.mean(all_forcing)
    overall_judge = np.mean(all_judge)
    overall_harmonic = np.mean(all_harmonic)
    
    return combined_scores, overall_forcing, overall_judge, overall_harmonic


def parse_directory_path(dirpath):
    """Extract layer and alpha (strength) from directory path
    
    Expected format: .../layer_{X}_strength_{Y}/...
    Returns: (layer, alpha) or (None, None) if not found
    """
    import re
    
    path_str = str(dirpath)
    
    # Match patterns like layer_4, layer_8, etc.
    layer_match = re.search(r'layer[_-](\d+)', path_str, re.IGNORECASE)
    
    # Match patterns like strength_1.0, strength_2.5, etc.
    strength_match = re.search(r'strength[_-]([\d._]+)', path_str, re.IGNORECASE)

    layer = int(layer_match.group(1)) if layer_match else None
    
    if strength_match:
        strength_str = strength_match.group(1)
        # Convert 1_0 format to 1.0
        alpha = float(strength_str.replace('_', '.'))
    else:
        alpha = None
    
    return layer, alpha


def find_directory_data(input_dirs):
    """Find directories with valid layer/alpha info
    
    Returns:
        list: List of tuples (directory_path, layer, alpha)
    """
    directory_data = []
    
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        
        layer, alpha = parse_directory_path(dir_path)
        
        if layer is not None and alpha is not None:
            directory_data.append((dir_path, layer, alpha))
        else:
            print(f"Warning: Could not parse layer/alpha from path: {dir_path}")
    
    return directory_data


def aggregate_results(input_dirs):
    """Aggregate metrics across multiple directories
    
    Returns:
        dict: {(layer, alpha): {'success_rate': float, 'judge_score': float, 'harmonic_mean': float}}
    """
    # Find all directories
    directory_data = find_directory_data(input_dirs)
    
    if not directory_data:
        print("Error: No valid directories found!")
        return {}
    
    print(f"Found {len(directory_data)} directories to process\n")
    
    aggregated = defaultdict(lambda: {'success_rates': [], 'judge_scores': [], 'harmonic_means': []})
    
    for dir_path, layer, alpha in directory_data:
        print(f"Processing layer {layer}, α={alpha}: {dir_path}")
        
        try:
            # Look for analysis_results.json for forcing statistics
            forcing_file = dir_path / "analysis_results.json"
            if not forcing_file.exists():
                print(f"  Warning: No analysis_results.json found")
                print()
                continue
            
            # Extract forcing pairs
            print(f"  Loading forcing results...")
            forcing_pairs = extract_forcing_statistics(forcing_file)
            
            # Extract judge pairs from all evaluation files
            print(f"  Extracting judge scores from evaluation files...")
            judge_pairs = extract_judge_statistics(dir_path)
            
            print(f"  Found {len(forcing_pairs)} pairs in forcing results")
            print(f"  Found {len(judge_pairs)} pairs in judge evaluations")
            
            # Compute combined scores
            combined_scores, overall_forcing, overall_judge, overall_harmonic = \
                compute_combined_scores(forcing_pairs, judge_pairs)
            
            if combined_scores:
                aggregated[(layer, alpha)]['success_rates'].append(overall_forcing)
                aggregated[(layer, alpha)]['judge_scores'].append(overall_judge)
                aggregated[(layer, alpha)]['harmonic_means'].append(overall_harmonic)
                print(f"  ✓ Overall: SR={overall_forcing:.1f}%, JS={overall_judge:.1f}%, H-Mean={overall_harmonic:.1f}%")
            else:
                print(f"  ✗ No matching pairs found between forcing and judge evaluations")
            
        except Exception as e:
            print(f"  ✗ Error processing directory: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Compute averages across runs (if multiple directories have same layer/alpha)
    final_results = {}
    for (layer, alpha), metrics in aggregated.items():
        avg_success = np.mean(metrics['success_rates']) if metrics['success_rates'] else 0.0
        avg_judge = np.mean(metrics['judge_scores']) if metrics['judge_scores'] else 0.0
        avg_harmonic = np.mean(metrics['harmonic_means']) if metrics['harmonic_means'] else 0.0
        
        final_results[(layer, alpha)] = {
            'success_rate': avg_success,
            'judge_score': avg_judge,
            'harmonic_mean': avg_harmonic,
            'n_runs': len(metrics['success_rates'])
        }
    
    return final_results


def create_acl_plots(results, output_dir):
    """Create three ACL-style plots: success rate, judge score, and steering score"""
    
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
    
    # Organize data by alpha values
    alpha_values = sorted(set(alpha for _, alpha in results.keys()))
    layers = sorted(set(layer for layer, _ in results.keys()))
    
    # Define colors for different alpha values
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(alpha_values)))
    
    # Plot configurations
    plot_configs = [
        {
            'metric': 'success_rate',
            'ylabel': 'Success Rate (%)',
            'title': 'Language Forcing Success Rate',
            'filename': 'success_rate_by_layer'
        },
        {
            'metric': 'judge_score',
            'ylabel': 'Judge Score (%)',
            'title': 'Judge Relevance Score',
            'filename': 'judge_score_by_layer'
        },
        {
            'metric': 'harmonic_mean',
            'ylabel': 'Steering Score (%)',
            'title': 'Steering Score (Harmonic Mean)',
            'filename': 'steering_score_by_layer'
        }
    ]
    
    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # ACL single-column width
        
        for i, alpha in enumerate(alpha_values):
            # Get data for this alpha value
            y_values = []
            x_values = []
            
            for layer in layers:
                if (layer, alpha) in results:
                    x_values.append(layer)
                    y_values.append(results[(layer, alpha)][config['metric']])
            
            if x_values:
                ax.plot(x_values, y_values, marker='o', markersize=4,
                       label=f'α={alpha}', color=colors[i], linewidth=1.5)
        
        if config['metric'] == 'harmonic_mean':
            ax.set_xlabel("Layer", fontsize=10, fontweight='normal')
        # ax.set_ylabel(config['ylabel'], fontsize=10, fontweight='normal')
        ax.set_ylim(0, 100)
        ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=8)
        
        # ACL-style frame
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')
        
        ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
        
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
        description="Aggregate and plot steering metrics across layers and alpha values"
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of directories containing results (e.g., generation_ablate/diffmean/layers/*/strength_*)"
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
    print(f"\nFound data for {len(results)} (layer, alpha) combinations")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Layer':<8} {'Alpha':<8} {'Success %':<12} {'Judge %':<12} {'Steering %':<12} {'N Runs':<8}")
    print("-"*80)
    
    for (layer, alpha), metrics in sorted(results.items()):
        print(f"{layer:<8} {alpha:<8.2f} {metrics['success_rate']:>9.1f}%   "
              f"{metrics['judge_score']:>9.1f}%   {metrics['harmonic_mean']:>10.1f}%   "
              f"{metrics['n_runs']:>6}")
    
    # Create plots
    print("\n" + "="*80)
    print("Generating ACL-style plots...")
    print("="*80)
    create_acl_plots(results, output_dir)
    
    # Save aggregated results
    output_json = output_dir / "aggregated_results.json"
    output_data = {
        'results': {
            f"layer_{layer}_alpha_{alpha}": metrics
            for (layer, alpha), metrics in results.items()
        },
        'summary': {
            'total_combinations': len(results),
            'layers': sorted(set(layer for layer, _ in results.keys())),
            'alphas': sorted(set(alpha for _, alpha in results.keys()))
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Aggregated results saved to: {output_json}")
    print(f"✓ All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()