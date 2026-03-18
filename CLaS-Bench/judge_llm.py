import argparse
import json
import os
import glob
import re
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm

# This is the "non-thinking" prompt you designed, adapted for this task.
SYSTEM_PROMPT_TEMPLATE = """You are an impartial judge. Your task is to evaluate a multilingual response from a language steering experiment. A question was asked in {source_language}, and the model was steered to respond in {target_language}.

Important: do NOT evaluate *which language* the response is in or whether steering succeeded. Ignore the target language entirely for scoring - evaluate the final response solely on its content, regardless of the language used.

You must evaluate the response based on two criteria:
1.  **Relevance:** Is the response a relevant answer to the question?
2.  **Coherence:** Is the response coherent text, or is it gibberish, repetitive, or useless?

Focus only on these criteria and do not consider language conformity or steering success.

**Scoring:**
- **0:** The response is completely unrelated to the question OR it is total gibberish/useless text.
- **1:** The response is *somewhat* related to the question but may be incomplete, partially off-target, or minimally useful.
- **2:** The response is clearly and directly related to the question and is coherent and useful text.

Begin your evaluation with a brief explanation (a few sentences) of your reasoning.
After your explanation, provide the rating in this exact format: "Rating: [[score]]".
"""


# This user prompt template provides the context to the judge.
USER_PROMPT_TEMPLATE = """Question ({source_language}):
{question}

Response ({target_language}):
{response}
"""

def parse_rating(text: str) -> int | None:
    """
    Robustly parses the [[score]] from the judge's response.
    """
    # Try to find the exact format first
    match = re.search(r"Rating: \[\[([012])\]\]", text)
    if match:
        return int(match.group(1))
    
    # Fallback: find any [[score]]
    match = re.search(r"\[\[([012])\]\]", text)
    if match:
        return int(match.group(1))
        
    # Fallback: find a single 0, 1, or 2 at the very end of the string
    match = re.search(r"\b([012])\b\s*$", text.strip())
    if match:
        return int(match.group(1))

    print(f"Warning: Could not parse rating from output: {text[:100]}...")
    return None

def create_chat_messages(system_prompt: str, user_prompts: list[str]) -> list[list[dict]]:
    """
    Creates a batch of chat message lists for vLLM's .chat() method.
    """
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for user_prompt in user_prompts
    ]

def main():
    parser = argparse.ArgumentParser(description="Evaluate steering experiment outputs using a judge LLM.")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-8B", 
                        help="The vLLM model to use as the judge (e.g., Qwen/Qwen2-7B-Instruct).")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the JSON results from steering experiments (e.g., 'results' or 'steering_results').")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the new JSON files with evaluations.")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    
    args = parser.parse_args()

    print(f"Loading judge model: {args.judge_model}")
    
    # Load the judge model with vLLM
    try:
        judge_llm = LLM(
            model=args.judge_model,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"Error loading model {args.judge_model}. Your specified 'qwen-3-8B' may not exist.")
        print(f"Please try a valid model name from Hugging Face, like 'Qwen/Qwen2-7B-Instruct'.")
        print(f"Error details: {e}")
        return

    # Judge sampling params: no randomness, short output for efficiency
    # We get Qwen-specific stop tokens from the tokenizer
    tokenizer = judge_llm.get_tokenizer()
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.eos_token_id] + tokenizer.encode(stop_tokens, add_special_tokens=False)
    
    judge_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=16384,
        stop_token_ids=list(set(stop_token_ids)) # Ensure unique IDs
    )
    
    # Find all JSON files recursively in the input directory
    json_files = glob.glob(os.path.join(args.input_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} experiment files to evaluate.")
    
    for file_path in tqdm(json_files, desc="Evaluating Files"):
        print(f"\nProcessing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # --- Determine source and target languages from your result files ---
        # For script 1 (activation-based)
        source_lang = data.get("deactivate_language")
        target_lang = data.get("activate_language")
        
        # For script 2 (residual-based)
        if source_lang is None or source_lang == "none":
            source_lang = data.get("source_language")
        if target_lang is None:
            target_lang = data.get("target_language")

        if source_lang is None or target_lang is None:
            print(f"Could not determine languages for {file_path}. Skipping.")
            continue
        
        print(f"Languages: {source_lang} -> {target_lang}")

        # --- Prepare prompts for the batch ---
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_language=source_lang, target_language=target_lang)
        user_prompts = []
        
        original_results = data.get("results", [])
        if not original_results:
            print(f"No 'results' list found in {file_path}. Skipping.")
            continue

        for item in original_results:
            user_prompts.append(
                USER_PROMPT_TEMPLATE.format(
                    source_language=source_lang,
                    question=item.get("input", ""),
                    target_language=target_lang,
                    response=item.get("output", "")
                )
            )

        # Create the batch
        messages_batch = create_chat_messages(system_prompt, user_prompts)
        
        print(f"Running judge batch inference on {len(messages_batch)} items...")
        
        # Run batch evaluation
        judge_outputs = judge_llm.chat(
            messages_batch, 
            sampling_params=judge_sampling_params,
            chat_template_kwargs={"enable_thinking": False} 
        )
        # --- Process and store results ---
        total_score = 0
        valid_scores = 0
        score_counts = {0: 0, 1: 0, 2: 0}
        
        for i, output in enumerate(judge_outputs):
            judge_response = output.outputs[0].text.strip()
            judge_score = parse_rating(judge_response)
            
            # Add evaluation to the original data item
            original_results[i]["judge_evaluation"] = {
                "judge_response": judge_response,
                "judge_score": judge_score
            }
            
            if judge_score is not None:
                total_score += judge_score
                valid_scores += 1
                score_counts[judge_score] += 1

        # Calculate average score
        avg_score = (total_score / valid_scores) if valid_scores > 0 else 0
        data["evaluation_summary"] = {
            "judge_model": args.judge_model,
            "average_score": avg_score,
            "total_items": len(original_results),
            "valid_scores": valid_scores,
            "score_counts": score_counts
        }
        
        print(f"Average score: {avg_score:.4f} | Counts (0/1/2): {score_counts[0]}/{score_counts[1]}/{score_counts[2]}")

        # --- Save the new evaluated file ---
        # Recreate the directory structure from input_dir in output_dir
        relative_path = os.path.relpath(file_path, args.input_dir)
        output_file_path = os.path.join(args.output_dir, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"✓ Saved evaluated results to: {output_file_path}")

    print("\nEvaluation complete.")
    
    # --- Analyze and save judge scores to judge_analysis.json ---
    print("\nAnalyzing judge scores across all files...")
    
    target_lang_stats = {}
    source_lang_stats = {}
    language_pair_success = {}
    all_judge_data = []
    max_questions = 0
    
    # Re-read all evaluated files to collect statistics
    json_files = glob.glob(os.path.join(args.output_dir, "**", "*.json"), recursive=True)
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        source_lang = data.get("deactivate_language") or data.get("source_language")
        target_lang = data.get("activate_language") or data.get("target_language")
        
        if source_lang is None or target_lang is None:
            continue
        
        results = data.get("results", [])
        max_questions = max(max_questions, len(results))
        pair_key = f"{source_lang}->{target_lang}"
        
        # Initialize dicts if needed
        if target_lang not in target_lang_stats:
            target_lang_stats[target_lang] = {"judge_scores": [], "total": 0, "source_langs": set()}
        if source_lang not in source_lang_stats:
            source_lang_stats[source_lang] = {"judge_scores": [], "total": 0, "target_langs": set()}
        if pair_key not in language_pair_success:
            language_pair_success[pair_key] = {"judge_scores": [], "total": 0}
        
        target_lang_stats[target_lang]["source_langs"].add(source_lang)
        source_lang_stats[source_lang]["target_langs"].add(target_lang)
        
        for q_idx, item in enumerate(results):
            judge_eval = item.get("judge_evaluation", {})
            judge_score = judge_eval.get("judge_score")
            
            if judge_score is not None:
                all_judge_data.append({
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "question_idx": q_idx,
                    "judge_score": judge_score
                })
                
                target_lang_stats[target_lang]["judge_scores"].append(judge_score)
                target_lang_stats[target_lang]["total"] += 1
                
                source_lang_stats[source_lang]["judge_scores"].append(judge_score)
                source_lang_stats[source_lang]["total"] += 1
                
                language_pair_success[pair_key]["judge_scores"].append(judge_score)
                language_pair_success[pair_key]["total"] += 1
    
    # Calculate overall average judge score
    all_scores = [d["judge_score"] for d in all_judge_data]
    overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    overall_total = len(all_scores)
    
    # Build analysis results
    analysis_results = {
        "summary": {
            "overall_average_judge_score": overall_avg_score,
            "total_evaluations": overall_total,
            "total_language_pairs": len(language_pair_success),
            "max_questions_per_pair": max_questions,
        },
        "per_target_language": {
            lang: {
                "average_judge_score": sum(stats["judge_scores"]) / len(stats["judge_scores"]) if stats["judge_scores"] else 0,
                "judge_scores": stats["judge_scores"],
                "total_evaluations": stats["total"],
                "source_languages_count": len(stats["source_langs"])
            }
            for lang, stats in target_lang_stats.items()
        },
        "per_source_language": {
            lang: {
                "average_judge_score": sum(stats["judge_scores"]) / len(stats["judge_scores"]) if stats["judge_scores"] else 0,
                "judge_scores": stats["judge_scores"],
                "total_evaluations": stats["total"],
                "target_languages_count": len(stats["target_langs"])
            }
            for lang, stats in source_lang_stats.items()
        },
        "language_pairs": {
            pair: {
                "average_judge_score": sum(stats["judge_scores"]) / len(stats["judge_scores"]) if stats["judge_scores"] else 0,
                "judge_scores": stats["judge_scores"],
                "total_evaluations": stats["total"]
            }
            for pair, stats in language_pair_success.items()
        },
        "question_wise": {
            f"Q{i+1}": {
                "judge_scores": [d["judge_score"] for d in all_judge_data if d["question_idx"] == i],
                "average_judge_score": sum(d["judge_score"] for d in all_judge_data if d["question_idx"] == i) / 
                                      len([d for d in all_judge_data if d["question_idx"] == i])
                                      if any(d["question_idx"] == i for d in all_judge_data) else 0,
                "total_evaluations": len([d for d in all_judge_data if d["question_idx"] == i])
            }
            for i in range(max_questions)
        }
    }
    
    # Save to judge_analysis.json
    judge_analysis_path = os.path.join(args.output_dir, "judge_analysis.json")
    with open(judge_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Saved judge analysis to: {judge_analysis_path}")
    print(f"Overall average judge score: {overall_avg_score:.4f}")

if __name__ == "__main__":
    main()