import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, default="data_aya",
                    help="Directory where processed data will be saved")
parser.add_argument("--model_name", type=str, default="CohereLabs/aya-23-8B",
                    help="Name or path of the HF model")


args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

# Language mapping: FLORES code -> CulturaX code
language_mapping = {
    # Major world languages
    'eng_Latn': 'en',
    'zho_Hans': 'zh',
    'spa_Latn': 'es',
    'fra_Latn': 'fr',
    'arb_Arab': 'ar',
    'rus_Cyrl': 'ru',
    'por_Latn': 'pt',
    'deu_Latn': 'de',
    'jpn_Jpan': 'ja',
    'hin_Deva': 'hi',
    'ita_Latn': 'it',
    'tur_Latn': 'tr',
    'kor_Hang': 'ko',
    'pol_Latn': 'pl',
    'ukr_Cyrl': 'uk',
    'nld_Latn': 'nl',
    'vie_Latn': 'vi',
    'tha_Thai': 'th',
    'ind_Latn': 'id',
    
    # 100M-500M speakers
    'ben_Beng': 'bn',
    'pes_Arab': 'fa',
    'urd_Arab': 'ur',
    'swh_Latn': 'sw',
    'pan_Guru': 'pa',
    'tel_Telu': 'te',
    'mar_Deva': 'mr',
    'tam_Taml': 'ta',
    'guj_Gujr': 'gu',
    'kan_Knda': 'kn',
    'mal_Mlym': 'ml',
    'sin_Sinh': 'si',
    'mya_Mymr': 'my',
    'uzn_Latn': 'uz',
    'ron_Latn': 'ro',
    'npi_Deva': 'ne',
    
    # 50M-100M speakers
    'amh_Ethi': 'am',
    'ory_Orya': 'or',
    'asm_Beng': 'as',
    'ces_Latn': 'cs',
    'ell_Grek': 'el',
    'swe_Latn': 'sv',
    'hun_Latn': 'hu',
    'azj_Latn': 'az',
    'bel_Cyrl': 'be',
    'cat_Latn': 'ca',
    'heb_Hebr': 'he',
    'srp_Cyrl': 'sr',
    'fin_Latn': 'fi',
    'dan_Latn': 'da',
    'slk_Latn': 'sk',
    
    # 20M-50M speakers
    'bul_Cyrl': 'bg',
    'hrv_Latn': 'hr',
    'lit_Latn': 'lt',
    'slv_Latn': 'sl',
    'est_Latn': 'et',
    'lvs_Latn': 'lv',
    'mkd_Cyrl': 'mk',
    'afr_Latn': 'af',
    'tgl_Latn': 'tl',
    'nob_Latn': 'no',
    'isl_Latn': 'is',
    'gle_Latn': 'ga',
    'cym_Latn': 'cy',
    'eus_Latn': 'eu',
    'glg_Latn': 'gl',
    'kat_Geor': 'ka',
    'hye_Armn': 'hy',
    'bod_Tibt': 'bo',
    'mlt_Latn': 'mt',
    'kaz_Cyrl': 'kk'
}

# Languages not in CulturaX - will use GlotCC
glotcc_languages = {}

def flores_to_glotcc(flores_code):
    """Convert FLORES code (xxx_Yyyy) to GlotCC format (xxx-Yyyy)."""
    return flores_code.replace('_', '-')

# Get unique FLORES codes
languages = [
    "bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl",
    "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no",
    "en", "sk", "ur", "el", "kk", "sw", "ka", "uk", "fa", "th",
    "id", "vi", "bn", "cs", "ro", "tl"
]

print(f"Number of languages: {len(languages)}")
print(f"CulturaX languages: {len(languages)}")

target_tokens = 10_000_000

def process_culturax_language(flores_code, culturax_code, target_tokens, save_dir, seed=42):
    """Extract text from CulturaX and tokenize up to target tokens, tracking bytes."""
    save_path = os.path.join(save_dir, f"culturax_{culturax_code}.pt")
    
    # Skip if already exists
    if os.path.exists(save_path):
        print(f"✓ {flores_code} ({culturax_code}): already exists, skipping")
        return
    
    try:
        print(f"Loading {flores_code} ({culturax_code}) from CulturaX...")
        
        # Load CulturaX dataset in streaming mode (it's huge!)
        dataset = load_dataset(
            "uonlp/CulturaX",
            culturax_code,
            split="train",
            streaming=True,
        )
        
        # Set random seed for shuffling
        random.seed(seed)
        
        # Tokenize up to target
        ids = []
        total_tokens = 0
        total_bytes = 0
        num_docs = 0
        
        # Use buffer shuffling for streamed dataset
        dataset_shuffled = dataset.shuffle(seed=seed, buffer_size=10000)
        
        for example in dataset_shuffled:
            text = example.get('text', '')
            
            if text is None or text.strip() == "":
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Calculate bytes for these tokens
            byte_tensor = torch.LongTensor(tokens).numpy().tobytes()
            size_bytes = len(byte_tensor)
            
            # Stop if we would exceed target tokens (approximate: 4 bytes per token for int32)
            if total_tokens + len(tokens) > target_tokens:
                # Add only what we need to reach target
                remaining_tokens = target_tokens - total_tokens
                ids.extend(tokens[:remaining_tokens])
                total_tokens = target_tokens
                total_bytes += remaining_tokens * 4
                break
            
            ids.extend(tokens)
            total_tokens += len(tokens)
            total_bytes += size_bytes
            num_docs += 1
            
            # Progress update every 100 documents
            if num_docs % 100 == 0:
                print(f"  {flores_code}: {total_tokens:,} tokens (~{total_bytes / 1024 / 1024:.2f} MB) from {num_docs} docs")
        
        if total_tokens == 0:
            print(f"✗ {flores_code} ({culturax_code}): No data extracted")
            return
        
        # Save as tensor
        tensor = torch.LongTensor(ids)
        torch.save(tensor, save_path)
        tensor_bytes = tensor.numpy().nbytes
        print(f"✓ {flores_code} ({culturax_code}): saved {total_tokens:,} tokens (~{tensor_bytes / 1024 / 1024:.2f} MB) from {num_docs} documents")
        
    except Exception as e:
        print(f"✗ {flores_code} ({culturax_code}): ERROR - {e}")

# Process all languages
print("\n=== Processing Languages ===\n")
for flores_code in languages:
    if flores_code in glotcc_languages:
        # Use GlotCC for these languages
        continue
    else:
        # Use CulturaX for most languages
        culturax_code = flores_code
        process_culturax_language(flores_code, culturax_code, target_tokens, args.save_dir)

print("\n=== Summary ===")
total_found = 0
total_missing = 0
total_tokens_sum = 0

for flores_code in languages:
    # Check both CulturaX and GlotCC paths
    if flores_code in glotcc_languages:
        save_path = os.path.join(args.save_dir, f"glotcc_{flores_code}.pt")
    else:
        culturax_code = flores_code
        save_path = os.path.join(args.save_dir, f"culturax_{culturax_code}.pt")
    
    if os.path.exists(save_path):
        tensor = torch.load(save_path)
        num_tokens = len(tensor)
        tensor_bytes = tensor.numpy().nbytes
        source = "GlotCC" if flores_code in glotcc_languages else "CulturaX"
        print(f"{flores_code} ({source}): {num_tokens:,} tokens (~{tensor_bytes / 1024 / 1024:.2f} MB)")
        total_found += 1
        total_tokens_sum += num_tokens
    else:
        print(f"{flores_code}: NOT FOUND")
        total_missing += 1

print(f"\nTotal: {total_found} found, {total_missing} missing")
print(f"Total tokens across all languages: {total_tokens_sum:,}")
print(f"Average tokens per language: {total_tokens_sum // total_found if total_found > 0 else 0:,}")