import argparse
import json
import os
import torch
from vllm import LLM, SamplingParams
from utils.utils import get_test_questions

def get_language_names():
    """Return a mapping from language codes to language names"""
    return {
        "bo": "Tibetan",
        "mt": "Maltese",
        "it": "Italian",
        "es": "Spanish",
        "de": "German",
        "ja": "Japanese",
        "ar": "Arabic",
        "zh": "Chinese",
        "af": "Afrikaans",
        "nl": "Dutch",
        "fr": "French",
        "pt": "Portuguese",
        "ru": "Russian",
        "ko": "Korean",
        "hi": "Hindi",
        "tr": "Turkish",
        "pl": "Polish",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "en": "English",
        "sk": "Slovak",
        "ur": "Urdu",
        "el": "Greek",
        "kk": "Kazakh",
        "sw": "Swahili",
        "ka": "Georgian",
        "uk": "Ukrainian",
        "fa": "Persian",
        "th": "Thai",
        "id": "Indonesian",
        "vi": "Vietnamese",
        "bn": "Bengali",
        "cs": "Czech",
        "ro": "Romanian",
        "tl": "Tagalog",
    }

def get_target_language_instructions():
    """Return 'Respond in X' instruction in each target language"""
    return {
        "bo": "བོད་སྐད་ནང་ལན་འདེབས།", 
        "mt": "Wieġeb bil-Malti.", 
        "it": "Rispondi in italiano.",
        "es": "Responde en español.",
        "de": "Antworte auf Deutsch.",
        "ja": "日本語で答えてください。",
        "ar": "أجب بالعربية.",
        "zh": "用中文回答。",
        "af": "Antwoord in Afrikaans.",
        "nl": "Antwoord in het Nederlands.",
        "fr": "Réponds en français.",
        "pt": "Responda em português.",
        "ru": "Ответь на русском.",
        "ko": "한국어로 답하세요.",
        "hi": "हिंदी में उत्तर दें।",
        "tr": "Türkçe cevap verin.",
        "pl": "Odpowiedz po polsku.",
        "sv": "Svara på svenska.",
        "da": "Svar på dansk.",
        "no": "Svar på norsk.",
        "en": "Respond in English.",
        "sk": "Odpovedajte po slovensky.", 
        "ur": "اردو میں جواب دیں۔", 
        "el": "Απαντήστε στα ελληνικά.", 
        "kk": "Қазақша жауап беріңіз.", 
        "sw": "Jibu kwa Kiswahili.", 
        "ka": "უპასუხეთ ქართულად.", 
        "uk": "Відповідайте українською.", 
        "fa": "به فارسی پاسخ دهید.", 
        "th": "ตอบในภาษาไทย", 
        "id": "Jawab dalam bahasa Indonesia.", 
        "vi": "Trả lời bằng tiếng Việt.", 
        "bn": "বাংলায় উত্তর দিন।", 
        "cs": "Odpovězte v češtině.", 
        "ro": "Răspundeți în limba română.", 
        "tl": "Sumagot sa Tagalog.", 
    }

def create_chat_messages(prompts, source_lang, target_lang, instruction_mode):
    """Create chat messages based on instruction mode"""
    lang_names = get_language_names()
    source_lang_name = lang_names.get(source_lang, source_lang)
    target_lang_name = lang_names.get(target_lang, target_lang)
    
    messages_batch = []
    for prompt in prompts:
        if instruction_mode == "no_instruction":
            user_content = prompt
        elif instruction_mode == "language_instruction":
            user_content = f"{prompt} Respond in {target_lang_name}."
        elif instruction_mode == "source_language_instruction":
            source_instructions = get_target_language_instructions()
            source_instruction = source_instructions.get(source_lang, f"Respond in {source_lang_name}.")
            user_content = f"{prompt} {source_instruction}"
        elif instruction_mode == "target_language_instruction":
            target_instructions = get_target_language_instructions()
            target_instruction = target_instructions.get(target_lang, f"Respond in {target_lang_name}.")
            user_content = f"{prompt} {target_instruction}"
        else:
            raise ValueError(f"Unknown instruction mode: {instruction_mode}")
        
        # Return as list of message dicts (chat format)
        messages_batch.append([{"role": "user", "content": user_content}])
    
    return messages_batch

def run_baseline_experiment(model, sampling_params, test_questions, 
                           source_lang, target_lang, instruction_mode):
    """Run baseline experiment with batched chat inference"""
    
    print(f"\n{'='*60}")
    print(f"Baseline ({instruction_mode}): {source_lang} → {target_lang}")
    print(f"{'='*60}")
    
    test_prompts = test_questions[source_lang]
    
    # Create batch of chat messages
    messages_batch = create_chat_messages(test_prompts, source_lang, target_lang, instruction_mode)
    
    print(f"Running batched inference on {len(messages_batch)} questions...")
    
    outputs = model.chat(messages_batch, sampling_params=sampling_params)
    
    # Process results
    all_results = []
    for i, (output, original_prompt) in enumerate(zip(outputs, test_prompts)):
        response = output.outputs[0].text.strip()
        
        all_results.append({
            "question_idx": i,
            "input": messages_batch[i][0]["content"],
            "output": response,
            "original_prompt": original_prompt
        })
        
        if i < 3:  # Print first 3 for verification
            print(f"Q{i+1}: {original_prompt[:50]}...")
            print(f"R{i+1}: {response[:100]}...\n")
    
    # Save results
    results = {
        "source_language": source_lang,
        "target_language": target_lang,
        "instruction_mode": instruction_mode,
        "model": args.model,
        "results": all_results
    }
    
    output_dir = os.path.join(args.output, instruction_mode)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{source_lang}_to_{target_lang}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to: {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--batch_mode", action='store_true')
    parser.add_argument("--source_lang", type=str, default="de")
    parser.add_argument("--target_lang", type=str, default="ru")
    parser.add_argument("--languages", nargs='+', 
                       default=["en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt", "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk", "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"])
    parser.add_argument("--instruction_mode", type=str, default="all",
                       choices=["no_instruction", "language_instruction", 
                               "source_language_instruction", "target_language_instruction", "all"])
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    global args
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = LLM(
        model=args.model, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )
    
    # Get model-specific stop tokens
    tokenizer = model.get_tokenizer()
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    print(f"Using stop tokens: {stop_token_ids}")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        repetition_penalty=1.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        skip_special_tokens=True
    )
    
    test_questions = get_test_questions(k=70)
    
    # Determine instruction modes to run
    if args.instruction_mode == "all":
        modes = ["language_instruction", "target_language_instruction"]
    else:
        modes = [args.instruction_mode]
    
    if args.batch_mode:
        languages = args.languages
        total = len(languages) * len(languages) * len(modes)
        counter = 0
        
        print(f"\nRunning {total} experiments...")
        
        for mode in modes:
            for source_lang in languages:
                for target_lang in languages:
                    counter += 1
                    print(f"\n[{counter}/{total}]")
                    
                    try:
                        run_baseline_experiment(
                            model, sampling_params, test_questions,
                            source_lang, target_lang, mode
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
    else:
        for mode in modes:
            run_baseline_experiment(
                model, sampling_params, test_questions,
                args.source_lang, args.target_lang, mode
            )
    
    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()