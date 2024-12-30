import json
import sys
import os

def match_category(content, category_list):
    """
    Given the raw content from the batch output (e.g. "Factual information (general or professional)"),
    do a case-insensitive substring match against the known category list.
    Return the first matching category if found; otherwise return 'other'.
    """
    content_lower = content.lower() if content else ""
    for cat in category_list:
        if cat.lower() in content_lower:
            return cat
    return "other - please continue to list them as (other: [unknown])"

def main(input_json_path, batch_output_file):
    """
    Reads:
      1) A JSON file (wrapped in '[]') where each element is an object
         that includes 'preference' (1.0 => first is preferred).
      2) A JSONL file (one JSON object per line), from which we extract
         the category in data_batch["response"]["body"]["choices"][0]["message"]["content"].
    
    Then calculates how often 'model_1' is preferred
    in each category (i.e., model_1 wins / total in that category).
    """
    categories = [
        'linguistics',
        'factual information (general or professional)',
        'history or common practices',
        'recommendation',
        'tips, opinions or advice',
        'analysis or decision explanation',
        'mathematical reasoning or calculation',
        'logical reasoning',
        'coding',
        'assisting or creative writing',
        'roleplay',
        'editing or rewriting',
        'information extraction or summarization',
        'classification',
        'multilinguality or translation',
        'awareness of ethics and other risks',
        'other - please continue to list them as (other: [category name])'
    ]

    # We'll store stats like: { category: {"total": int, "model_1_wins": int} }
    category_stats = {}
    for cat in categories:
        category_stats[cat] = {"total": 0, "model_1_wins": 0}
    other_cat = "other - please continue to list them as (other: [unknown])"
    if other_cat not in category_stats:
        category_stats[other_cat] = {"total": 0, "model_1_wins": 0}

    # 1) Read the entire JSON file (wrapped in '[]') => list of objects
    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            data_input_list = json.load(f)  # a list of objects
        except json.JSONDecodeError:
            print(f"Error: Could not parse {input_json_path} as JSON.")
            sys.exit(1)
    
    # 2) Read the batch output file line-by-line (JSONL)
    batch_output_list = []
    with open(batch_output_file, "r", encoding="utf-8") as f_batch:
        for line in f_batch:
            line = line.strip()
            if not line:
                continue
            try:
                batch_output_list.append(json.loads(line))
            except json.JSONDecodeError:
                print("Warning: Could not decode a line in the batch output file.")
                batch_output_list.append({})  # keep placeholder so indexing matches

    # Check if the lengths match
    if len(data_input_list) != len(batch_output_list):
        print(f"Warning: input file has {len(data_input_list)} items, "
              f"batch output has {len(batch_output_list)} lines. They may not align perfectly.")
    
    # 3) Process them in parallel, by index
    for i, data_input in enumerate(data_input_list):
        if i >= len(batch_output_list):
            break
        data_batch = batch_output_list[i]
        
        # a) Extract category from batch output
        try:
            category_raw = data_batch["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            category_raw = "other"

        category_matched = match_category(category_raw, categories)
        
        # b) Determine if model_1 is the winner by using "preference"
        preference = data_input.get("preference", None)
        # We'll assume preference=1.0 => model_1, else => model_2 or no preference
        is_model_1_winner = (preference == 1.0)

        # c) Update stats
        category_stats.setdefault(category_matched, {"total": 0, "model_1_wins": 0})
        category_stats[category_matched]["total"] += 1
        if is_model_1_winner:
            category_stats[category_matched]["model_1_wins"] += 1

    # 4) Print out the per-category win rate for model_1
    print("\nPer-category win rate for model_1 over model_2 (based on 'preference' == 1.0):\n")
    for cat, stats in category_stats.items():
        total = stats["total"]
        wins = stats["model_1_wins"]
        if total > 0:
            rate = wins / total
            print(f"Category: {cat}\n  total = {total}, model_1_wins = {wins}, win_rate = {rate:.2f}\n")
        else:
            print(f"Category: {cat}\n  No data.\n")


if __name__ == "__main__":
    """
    Usage:
      python script.py <input_json> <batch_output_jsonl>

    Where:
      <input_json> is a file containing a JSON array of objects, e.g.:
        [
          {
            "instruction": "...",
            "output_1": "...",
            "generator_1": "...",
            "dataset": "...",
            "output_2": "...",
            "generator_2": "...",
            "annotator": "...",
            "preference": 1.0,
            "raw_completion": "...",
            "time_per_example": ...,
            ...
          },
          {
            ...
          }
        ]

      <batch_output_jsonl> is a file with one JSON object per line, each containing
      the category in ["response"]["body"]["choices"][0]["message"]["content"].
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_json> <batch_output_jsonl>")
        sys.exit(1)

    input_json_path = sys.argv[1]
    batch_output_jsonl = sys.argv[2]

    if not os.path.exists(input_json_path):
        print(f"Error: input file not found: {input_json_path}")
        sys.exit(1)
    if not os.path.exists(batch_output_jsonl):
        print(f"Error: batch output file not found: {batch_output_jsonl}")
        sys.exit(1)

    main(input_json_path, batch_output_jsonl)

