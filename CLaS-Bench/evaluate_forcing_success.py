import json
import os
import glob
from collections import defaultdict, Counter
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

def get_flores_to_original_mapping():
    """
    FLORES-200 language code to ISO 639-1 (2-letter code) mapping.
    
    Supports all 34 requested languages:
    en, af, ar, bo, da, de, es, fr, hi, it, ja, ko, mt, nl, no, pl, pt, 
    ru, sv, tr, zh, sk, ur, el, kk, sw, ka, uk, fa, th, id, vi, bn, cs, ro, tl
    
    Note: Some ISO 639-1 codes map to multiple FLORES-200 codes due to script
    variations (e.g., Norwegian has both Bokmål and Nynorsk, Chinese has 
    Simplified, Traditional, and Cantonese variations).
    """
    return {
        # Core Western European languages
        "eng_Latn": "en",      # English
        "deu_Latn": "de",      # German
        "fra_Latn": "fr",      # French
        "ita_Latn": "it",      # Italian
        "spa_Latn": "es",      # Spanish
        "por_Latn": "pt",      # Portuguese
        "nld_Latn": "nl",      # Dutch
        "pol_Latn": "pl",      # Polish
        "ces_Latn": "cs",      # Czech
        "ron_Latn": "ro",      # Romanian
        "slk_Latn": "sk",      # Slovak
        
        # Scandinavian languages
        "swe_Latn": "sv",      # Swedish
        "dan_Latn": "da",      # Danish
        "nob_Latn": "no",      # Norwegian Bokmål
        "nno_Latn": "no",      # Norwegian Nynorsk
        "afr_Latn": "af",      # Afrikaans
        
        # Romance languages
        "mlt_Latn": "mt",      # Maltese
        
        # Slavic languages
        "rus_Cyrl": "ru",      # Russian
        "ukr_Cyrl": "uk",      # Ukrainian
        
        # Greek
        "ell_Grek": "el",      # Greek (Modern)
        
        # Asian languages - East Asian
        "jpn_Jpan": "ja",      # Japanese
        "kor_Hang": "ko",      # Korean
        "zho_Hans": "zh",      # Chinese (Simplified)
        "zho_Hant": "zh",      # Chinese (Traditional)
        "yue_Hant": "zh",      # Yue Chinese (Cantonese)
        
        # Asian languages - South Asian
        "hin_Deva": "hi",      # Hindi
        "ben_Beng": "bn",      # Bengali
        
        # Asian languages - Southeast Asian
        "tha_Thai": "th",      # Thai
        "ind_Latn": "id",      # Indonesian
        "vie_Latn": "vi",      # Vietnamese
        "tgl_Latn": "tl",      # Tagalog (Filipino)
        
        # Other Asian languages
        "bod_Tibt": "bo",      # Standard Tibetan (Tibetan script)
        "kat_Geor": "ka",      # Georgian
        "kaz_Cyrl": "kk",      # Kazakh (Cyrillic script)
        
        # Middle Eastern and African languages
        "arb_Arab": "ar",      # Modern Standard Arabic
        "tur_Latn": "tr",      # Turkish
        "fas_Arab": "fa",      # Persian (Farsi)
        "pes_Arab": "fa",
        "urd_Arab": "ur",      # Urdu
        "swh_Latn": "sw",      # Swahili
    }

try:
    model = fasttext.load_model('lid218e.bin')
except:
    import urllib.request
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin', 'lid218e.bin')
    model = fasttext.load_model('lid218e.bin')


def detect_language(text, debug=False):
    """
    Detect language using fasttext.
    
    Returns ISO 639-1 language code (e.g., "en", "de", "fr")
    
    Flow:
    1. FastText predicts FLORES code (e.g., "eng_Latn")
    2. Map FLORES code to ISO 639-1 (e.g., "en")
    3. Return ISO 639-1 code
    """
    if not text.strip():
        return "unknown"
    
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return "unknown"
    
    predictions = model.predict(clean_text, k=1)
    flores_code = predictions[0][0].replace('__label__', '')
    iso_code = flores_to_original_mapping.get(flores_code, flores_code)
    
    if debug:
        print(f"  Lang: {predictions[0]} Prob: {predictions[1]}")
    
    return iso_code


def create_visualizations(language_success, detected_counts, all_detections):
    """Create activation vs source language heatmap visualization"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    lang_list = sorted(set(d["activate"] for d in all_detections if d["activate"]))
    source_list = sorted(set(d["deactivate"] for d in all_detections if d["deactivate"]))
    
    heatmap_data = np.zeros((len(source_list), len(lang_list)))
    
    for i, source_lang in enumerate(source_list):
        for j, target_lang in enumerate(lang_list):
            matches = [d for d in all_detections 
                      if d["deactivate"] == source_lang and d["activate"] == target_lang]
            if matches:
                successes = sum(1 for m in matches if m["detected"] == target_lang)
                heatmap_data[i, j] = (successes / len(matches)) * 100
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(lang_list)))
    ax.set_yticks(range(len(source_list)))
    ax.set_xticklabels(lang_list, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(source_list, fontsize=14)
    
    ax.set_xlabel('Target Language', fontsize=18)
    ax.set_ylabel('Source Language', fontsize=18)
    ax.set_title('Language Activation Success Rate (%)', fontsize=20, pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{args.output_path}/heatmap_activation_success.pdf')
    plt.close()


def analyze_results():
    """Analyze all intervention results"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="generation/Llama-3.1-8B_1/activate")
    parser.add_argument("--output_path", type=str, default="generation/Llama-3.1-8B_1/activate")
    
    global args
    args = parser.parse_args()
    
    results_dir = args.input_path
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    print(f"Found {len(json_files)} result files\n")
    
    # Extended language mapping (ISO 639-1 to itself for reference)
    lang_mapping = {
        "de": "de", "es": "es", "fr": "fr", "it": "it", "pt": "pt", "ru": "ru",
        "zh": "zh", "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi", "tr": "tr",
        "pl": "pl", "nl": "nl", "sv": "sv", "da": "da", "no": "no",
        "af": "af", "mt": "mt", "bo": "bo", "en": "en",
        "sk": "sk", "ur": "ur", "el": "el", "kk": "kk", "sw": "sw", "ka": "ka",
        "uk": "uk", "fa": "fa", "th": "th", "id": "id", "vi": "vi", "bn": "bn",
        "cs": "cs", "ro": "ro", "tl": "tl"
    }
    
    global flores_to_original_mapping
    flores_to_original_mapping = get_flores_to_original_mapping()
    
    # Statistics containers
    language_success = defaultdict(lambda: {"total": 0, "success": 0, "by_question": defaultdict(int)})
    language_pair_success = defaultdict(lambda: {"total": 0, "success": 0, "success_rate": 0.0})
    all_detections = []
    max_questions = 0
    
    # ============ LOAD AND PROCESS DATA ============
    print("=" * 80)
    print("LOADING AND PROCESSING DATA")
    print("=" * 80)
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract language codes (handle both naming conventions)
        deactivate_lang = data.get("deactivate_language") or data.get("source_language")
        activate_lang = data.get("activate_language") or data.get("target_language")
        results = data.get("results", [])
        
        if not activate_lang or not results:
            continue
        
        # Ensure we're using ISO 639-1 codes
        # (if your JSON uses FLORES codes, add conversion here)
        target_lang_code = lang_mapping.get(activate_lang, activate_lang)
        pair_key = f"{deactivate_lang}_to_{activate_lang}" if deactivate_lang else activate_lang
        
        max_questions = max(max_questions, len(results))
        
        # Process each result
        for i, result in enumerate(results):
            output_text = result.get("output", "")
            detected_lang = detect_language(output_text)  # Returns ISO 639-1
            
            # Store detection record
            all_detections.append({
                "deactivate": deactivate_lang,
                "activate": activate_lang,
                "question_idx": i,
                "detected": detected_lang,
                "output": output_text[:100] + "..." if len(output_text) > 100 else output_text,
                "success": detected_lang == target_lang_code  # Store success as boolean
            })
            
            # Update per-target-language statistics
            language_success[activate_lang]["total"] += 1
            if detected_lang == target_lang_code:
                language_success[activate_lang]["success"] += 1
                language_success[activate_lang]["by_question"][i] += 1
            
            # Update per-language-pair statistics
            language_pair_success[pair_key]["total"] += 1
            if detected_lang == target_lang_code:
                language_pair_success[pair_key]["success"] += 1
    
    # Calculate success rates for pairs
    for pair_key in language_pair_success:
        stats = language_pair_success[pair_key]
        stats["success_rate"] = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    
    print(f"\nSuccessfully processed {len(all_detections)} detections")
    print(f"Across {len(json_files)} language pairs\n")
    
    # ============ PER-TARGET-LANGUAGE STATISTICS ============
    print("\n" + "=" * 80)
    print("PER-TARGET-LANGUAGE SUCCESS RATES")
    print("(How often was the model successfully forced to generate each language?)")
    print("=" * 80)
    print(f"{'Target Lang':<12} {'Success Rate':<15} {'Successes':<12} {'Total Tests':<12}")
    print("-" * 80)
    
    overall_success = 0
    overall_total = 0
    
    for lang in sorted(language_success.keys()):
        stats = language_success[lang]
        success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{lang:<12} {success_rate:>7.1f}%         {stats['success']:<12} {stats['total']:<12}")
        overall_success += stats["success"]
        overall_total += stats["total"]
    
    print("-" * 80)
    overall_rate = (overall_success / overall_total) * 100 if overall_total > 0 else 0
    print(f"{'OVERALL':<12} {overall_rate:>7.1f}%         {overall_success:<12} {overall_total:<12}")
    
    # ============ PER-SOURCE-LANGUAGE STATISTICS ============
    print("\n" + "=" * 80)
    print("PER-SOURCE-LANGUAGE SUCCESS RATES")
    print("(How often did activation succeed starting from each language?)")
    print("=" * 80)
    
    source_lang_stats = defaultdict(lambda: {"success": 0, "total": 0, "target_langs": set()})
    
    for detection in all_detections:
        target_lang = detection["activate"]
        source_lang = detection["deactivate"]
        
        source_lang_stats[source_lang]["total"] += 1
        source_lang_stats[source_lang]["target_langs"].add(target_lang)
        
        if detection["success"]:  # Use stored success boolean
            source_lang_stats[source_lang]["success"] += 1
    
    print(f"{'Source Lang':<12} {'Success Rate':<15} {'Successes':<12} {'Total Tests':<12} {'Target Langs':<12}")
    print("-" * 80)
    
    sorted_source_langs = sorted(source_lang_stats.items(), 
                                key=lambda x: x[1]["success"]/x[1]["total"] if x[1]["total"] > 0 else 0, 
                                reverse=True)
    
    for source_lang, stats in sorted_source_langs:
        success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        target_count = len(stats["target_langs"])
        print(f"{source_lang:<12} {success_rate:>7.1f}%         {stats['success']:<12} {stats['total']:<12} {target_count:<12}")
    
    # ============ PER-TARGET-LANGUAGE ANALYSIS ============
    print("\n" + "=" * 80)
    print("ACTIVATION SUCCESS BY TARGET LANGUAGE")
    print("(Tested across all source languages)")
    print("=" * 80)
    
    target_lang_stats = defaultdict(lambda: {"success": 0, "total": 0, "source_langs": set()})
    
    for detection in all_detections:
        target_lang = detection["activate"]
        source_lang = detection["deactivate"]
        
        target_lang_stats[target_lang]["total"] += 1
        target_lang_stats[target_lang]["source_langs"].add(source_lang)
        
        if detection["success"]:
            target_lang_stats[target_lang]["success"] += 1
    
    print(f"{'Target Lang':<12} {'Success Rate':<15} {'Successes':<12} {'Total Tests':<12} {'Source Langs':<12}")
    print("-" * 80)
    
    sorted_target_langs = sorted(target_lang_stats.items(), 
                                key=lambda x: x[1]["success"]/x[1]["total"] if x[1]["total"] > 0 else 0, 
                                reverse=True)
    
    for target_lang, stats in sorted_target_langs:
        success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        source_count = len(stats["source_langs"])
        print(f"{target_lang:<12} {success_rate:>7.1f}%         {stats['success']:<12} {stats['total']:<12} {source_count:<12}")
    
    # ============ QUESTION-WISE STATISTICS ============
    print("\n" + "=" * 80)
    print("QUESTION-WISE SUCCESS RATES")
    print("=" * 80)
    print(f"{'Question':<12} {'Success Rate':<15} {'Successes':<12} {'Total':<12}")
    print("-" * 80)
    
    for q_idx in range(max_questions):
        q_successes = sum(1 for d in all_detections if d["question_idx"] == q_idx and d["success"])
        q_total = sum(1 for d in all_detections if d["question_idx"] == q_idx)
        q_rate = (q_successes / q_total) * 100 if q_total > 0 else 0
        print(f"Q{q_idx+1:<11} {q_rate:>7.1f}%         {q_successes:<12} {q_total:<12}")
    
    # ============ LANGUAGE PAIR STATISTICS ============
    print("\n" + "=" * 80)
    print("LANGUAGE PAIR SUCCESS RATES")
    print("=" * 80)
    print(f"{'Source -> Target':<25} {'Success Rate':<15} {'Successes':<12} {'Total':<12}")
    print("-" * 80)
    
    sorted_pairs = sorted(language_pair_success.items(),
                         key=lambda x: x[1]["success_rate"],
                         reverse=True)
    
    # ============ DETECTED LANGUAGE DISTRIBUTION ============
    print("\n" + "=" * 80)
    print("DETECTED LANGUAGE DISTRIBUTION")
    print("(What languages did fasttext detect in the outputs?)")
    print("=" * 80)
    
    detected_counts = Counter([d["detected"] for d in all_detections])
    print(f"{'Language':<12} {'Count':<12} {'Percentage':<12}")
    print("-" * 80)
    
    for lang, count in detected_counts.most_common(20):
        percentage = (count / len(all_detections)) * 100
        print(f"{lang:<12} {count:<12} {percentage:>7.1f}%")
    
    # ============ GENERATE VISUALIZATIONS ============
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    create_visualizations(language_success, detected_counts, all_detections)
    
    # ============ SAVE RESULTS ============
    output_file = f"{args.output_path}/analysis_results.json"
    
    serializable_stats = {}
    for lang, stats in language_success.items():
        serializable_stats[lang] = {
            "total": stats["total"],
            "success": stats["success"],
            "success_rate": (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0,
            "by_question": dict(stats["by_question"])
        }
    
    analysis_results = {
        "summary": {
            "overall_success_rate": overall_rate,
            "total_tests": overall_total,
            "total_language_pairs": len(json_files),
            "max_questions_per_pair": max_questions,
        },
        "per_target_language": {
            lang: {
                "success_rate": (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0,
                "successes": stats["success"],
                "total_tests": stats["total"],
                "source_languages_count": len(stats["source_langs"])
            }
            for lang, stats in target_lang_stats.items()
        },
        "per_source_language": {
            lang: {
                "success_rate": (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0,
                "successes": stats["success"],
                "total_tests": stats["total"],
                "target_languages_count": len(stats["target_langs"])
            }
            for lang, stats in source_lang_stats.items()
        },
        "language_pairs": {
            pair: {
                "success_rate": stats["success_rate"],
                "successes": stats["success"],
                "total": stats["total"]
            }
            for pair, stats in language_pair_success.items()
        },
        "question_wise": {
            f"Q{i+1}": {
                "successes": sum(1 for d in all_detections if d["question_idx"] == i and d["success"]),
                "total": sum(1 for d in all_detections if d["question_idx"] == i),
                "success_rate": (sum(1 for d in all_detections if d["question_idx"] == i and d["success"]) / 
                               sum(1 for d in all_detections if d["question_idx"] == i)) * 100 
                               if sum(1 for d in all_detections if d["question_idx"] == i) > 0 else 0
            }
            for i in range(max_questions)
        },
        "detected_language_distribution": dict(detected_counts),
        "sample_detections": all_detections[:20]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {output_file}\n")


if __name__ == "__main__":
    analyze_results()