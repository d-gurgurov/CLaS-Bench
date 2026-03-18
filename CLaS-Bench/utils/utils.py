import os

def get_test_questions(data_dir="data", k=70):
    """
    Load test questions from language-specific data files.
    
    Args:
        data_dir: Path to the data directory containing language files
        k: Number of examples to load per language (default: 70)
    
    Returns:
        Dictionary mapping language codes to lists of questions
        Example: {"en": [question1, question2, ...], "de": [...], ...}
    """
    test_questions = {}
    
    # Get all .txt files in data directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            lang_code = filename.replace(".txt", "")
            filepath = os.path.join(data_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # Strip whitespace and filter out empty lines
                questions = [line.strip() for line in lines if line.strip()]
                
                # Take only k examples
                questions = questions[:k]
                
                test_questions[lang_code] = questions
                print(f"Loaded {len(questions)} questions for {lang_code}")
            
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not test_questions:
        raise ValueError(f"No language files found in {data_dir}")
    
    print(f"\nSuccessfully loaded {len(test_questions)} languages")
    return test_questions