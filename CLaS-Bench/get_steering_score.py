import json
import argparse
import numpy as np
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def harmonic_mean(a, b):
    """Compute harmonic mean of two values"""
    if a == 0 or b == 0:
        return 0.0
    return 2 * (a * b) / (a + b)


def extract_judge_per_language(judge_analysis_file):
    """Extract judge scores per target and source language from judge_analysis.json
    
    Returns:
        dict: {
            'per_target_language': {lang: avg_score},
            'per_source_language': {lang: avg_score},
            'overall': avg_score
        }
    """
    data = load_json(judge_analysis_file)
    
    result = {
        'per_target_language': {},
        'per_source_language': {},
        'overall': 0.0
    }
    
    # Extract per-target-language scores
    per_target = data.get('per_target_language', {})
    for lang, stats in per_target.items():
        avg_score = stats.get('average_judge_score', 0.0)
        # Convert 0-2 scale to 0-100
        result['per_target_language'][lang] = (avg_score / 2.0) * 100
    
    # Extract per-source-language scores
    per_source = data.get('per_source_language', {})
    for lang, stats in per_source.items():
        avg_score = stats.get('average_judge_score', 0.0)
        # Convert 0-2 scale to 0-100
        result['per_source_language'][lang] = (avg_score / 2.0) * 100
    
    # Extract overall score
    summary = data.get('summary', {})
    overall_avg = summary.get('overall_average_judge_score', 0.0)
    result['overall'] = (overall_avg / 2.0) * 100
    
    return result


def extract_forcing_per_language(analysis_results_file):
    """Extract forcing success rates per target and source language from analysis_results.json
    
    Returns:
        dict: {
            'per_target_language': {lang: success_rate},
            'per_source_language': {lang: success_rate},
            'overall': success_rate
        }
    """
    data = load_json(analysis_results_file)
    
    result = {
        'per_target_language': {},
        'per_source_language': {},
        'overall': 0.0
    }
    
    # Extract per-target-language success rates (already in 0-100 format)
    per_target = data.get('per_target_language', {})
    for lang, stats in per_target.items():
        success_rate = stats.get('success_rate', 0.0)
        # Check if already in 0-100 format or 0-1 format
        result['per_target_language'][lang] = success_rate if success_rate > 1 else success_rate * 100
    
    # Extract per-source-language success rates (already in 0-100 format)
    per_source = data.get('per_source_language', {})
    for lang, stats in per_source.items():
        success_rate = stats.get('success_rate', 0.0)
        # Check if already in 0-100 format or 0-1 format
        result['per_source_language'][lang] = success_rate if success_rate > 1 else success_rate * 100
    
    # Extract overall success rate
    summary = data.get('summary', {})
    overall_success = summary.get('overall_success_rate', 0.0)
    result['overall'] = overall_success if overall_success > 1 else overall_success * 100
    
    return result


def compute_results(forcing_data, judge_data):
    """Compute harmonic means and aggregate results
    
    Returns:
        dict: Aggregated results with harmonic means
    """
    results = {
        'per_target_language': {},
        'per_source_language': {},
        'overall': {}
    }
    
    # Per target language
    target_langs = set(forcing_data['per_target_language'].keys()) | set(judge_data['per_target_language'].keys())
    for lang in sorted(target_langs):
        forcing_score = forcing_data['per_target_language'].get(lang, 0.0)
        judge_score = judge_data['per_target_language'].get(lang, 0.0)
        h_mean = harmonic_mean(forcing_score, judge_score)
        
        results['per_target_language'][lang] = {
            'forcing': forcing_score,
            'judge': judge_score,
            'harmonic_mean': h_mean
        }
    
    # Per source language
    source_langs = set(forcing_data['per_source_language'].keys()) | set(judge_data['per_source_language'].keys())
    for lang in sorted(source_langs):
        forcing_score = forcing_data['per_source_language'].get(lang, 0.0)
        judge_score = judge_data['per_source_language'].get(lang, 0.0)
        h_mean = harmonic_mean(forcing_score, judge_score)
        
        results['per_source_language'][lang] = {
            'forcing': forcing_score,
            'judge': judge_score,
            'harmonic_mean': h_mean
        }
    
    # Overall
    forcing_overall = forcing_data['overall']
    judge_overall = judge_data['overall']
    h_mean_overall = harmonic_mean(forcing_overall, judge_overall)
    
    results['overall'] = {
        'forcing': forcing_overall,
        'judge': judge_overall,
        'harmonic_mean': h_mean_overall
    }
    
    return results


def write_results_txt(results, output_file, method_name=""):
    """Write results to a text file
    
    Args:
        results: dict with per_target_language, per_source_language, overall
        output_file: path to output text file
        method_name: optional name of the method
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        if method_name:
            f.write(f"{'='*80}\n")
            f.write(f"RESULTS FOR METHOD: {method_name}\n")
            f.write(f"{'='*80}\n\n")
        
        # Overall results
        f.write(f"{'='*80}\n")
        f.write("OVERALL RESULTS\n")
        f.write(f"{'='*80}\n")
        overall = results['overall']
        f.write(f"Language Forcing Success Rate:  {overall['forcing']:>7.2f}%\n")
        f.write(f"Judge Relevance Score:          {overall['judge']:>7.2f}%\n")
        f.write(f"Steering Score (Harmonic Mean): {overall['harmonic_mean']:>7.2f}%\n")
        f.write("\n")
        
        # Per target language
        f.write(f"{'='*80}\n")
        f.write("PER TARGET LANGUAGE RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Target Language':<20} {'Forcing %':<15} {'Judge %':<15} {'Harmonic Mean %':<15}\n")
        f.write(f"{'-'*80}\n")
        
        for lang in sorted(results['per_target_language'].keys()):
            metrics = results['per_target_language'][lang]
            f.write(f"{lang:<20} {metrics['forcing']:>12.2f}% {metrics['judge']:>12.2f}% {metrics['harmonic_mean']:>13.2f}%\n")
        
        f.write("\n")
        
        # Per source language
        f.write(f"{'='*80}\n")
        f.write("PER SOURCE LANGUAGE RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Source Language':<20} {'Forcing %':<15} {'Judge %':<15} {'Harmonic Mean %':<15}\n")
        f.write(f"{'-'*80}\n")
        
        for lang in sorted(results['per_source_language'].keys()):
            metrics = results['per_source_language'][lang]
            f.write(f"{lang:<20} {metrics['forcing']:>12.2f}% {metrics['judge']:>12.2f}% {metrics['harmonic_mean']:>13.2f}%\n")
        
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute steering metrics (language forcing + judge scores) with harmonic mean aggregation"
    )
    parser.add_argument(
        "--forcing_json",
        type=str,
        required=True,
        help="Path to analysis_results.json (language forcing results)"
    )
    parser.add_argument(
        "--judge_json",
        type=str,
        required=True,
        help="Path to judge_analysis.json (judge evaluation results)"
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        required=True,
        help="Path to output text file for results"
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="",
        help="Optional name of the steering method (for display)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    forcing_file = Path(args.forcing_json)
    judge_file = Path(args.judge_json)
    
    if not forcing_file.exists():
        print(f"Error: Forcing JSON file not found: {forcing_file}")
        return
    
    if not judge_file.exists():
        print(f"Error: Judge JSON file not found: {judge_file}")
        return
    
    print(f"Loading forcing results from: {forcing_file}")
    print(f"Loading judge results from: {judge_file}")
    
    try:
        forcing_data = extract_forcing_per_language(forcing_file)
        judge_data = extract_judge_per_language(judge_file)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return
    
    print("Computing harmonic means...")
    results = compute_results(forcing_data, judge_data)
    
    # Print results to console
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    overall = results['overall']
    print(f"Language Forcing Success Rate:  {overall['forcing']:>7.2f}%")
    print(f"Judge Relevance Score:          {overall['judge']:>7.2f}%")
    print(f"Steering Score (Harmonic Mean): {overall['harmonic_mean']:>7.2f}%")
    
    # Write to file
    output_file = Path(args.output_txt)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    write_results_txt(results, output_file, args.method_name)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
