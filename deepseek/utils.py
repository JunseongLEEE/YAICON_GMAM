import torch
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Summarization/Story generation using Deepseek")
    parser.add_argument("--task", choices=["summary", "story", "likert_summary", "likert_story", "recognition_summary", "recognition_story", 
                                           "pairwise_summary", "pairwise_story", "pairwise_comparison_summary", "pairwise_comparison_story"], required=True)
    parser.add_argument("--generated_file", type=str, nargs='+', help="Path(s) to JSON file(s) containing generated outputs to score")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--output_path", type=str, default="results/results.json", help="Path to save outputs")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt file")
    parser.add_argument("--target_model", type=str, required=True, help="Name of the target model for results grouping")
    return parser.parse_args()

def load_prompt_template(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = f.read().split("User Prompt:")
    system_prompt = lines[0].replace("System Prompt:", "").strip()
    user_prompt_template = lines[1].strip()
    return system_prompt, user_prompt_template

def build_prompt(system_prompt, user_prompt_template, content_dict):
    prompt = user_prompt_template
    for k, v in content_dict.items():
        prompt = prompt.replace(f"{{{k}}}", v)
    return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"

def compute_expected_score(output_logits, tokenizer):
    token_probs = torch.softmax(output_logits[0, -1], dim=-1)
    scores = [1, 2, 3, 4, 5]
    token_ids = [tokenizer(str(s), add_special_tokens=False).input_ids[0] for s in scores]
    probs = [token_probs[token_id].item() for token_id in token_ids]
    expected_score = sum(s * p for s, p in zip(scores, probs))
    return expected_score, dict(zip(map(str, scores), probs))

def compute_yes_no_probs(output_logits, tokenizer):
    token_probs = torch.softmax(output_logits[0, -1], dim=-1)
    yes_ids = tokenizer("Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer("No", add_special_tokens=False).input_ids

    if len(yes_ids) == 1 and len(no_ids) == 1:
        yes_prob = token_probs[yes_ids[0]].item()
        no_prob = token_probs[no_ids[0]].item()
    else:
        print(f"Warning: 'Yes' or 'No' is not a single token. yes_ids={yes_ids}, no_ids={no_ids}")
        yes_prob = 0.0
        no_prob = 0.0

    return {
        "Yes": yes_prob,
        "No": no_prob
    }

def compute_choice_probs(output_logits, tokenizer):
    token_probs = torch.softmax(output_logits[0, -1], dim=-1)
    ids_1 = tokenizer("1", add_special_tokens=False).input_ids
    ids_2 = tokenizer("2", add_special_tokens=False).input_ids

    if len(ids_1) == 1 and len(ids_2) == 1:
        prob_1 = token_probs[ids_1[0]].item()
        prob_2 = token_probs[ids_2[0]].item()
    else:
        print(f"Warning: '1' or '2' is not a single token. ids_1={ids_1}, ids_2={ids_2}")
        prob_1 = 0.0
        prob_2 = 0.0

    return {"1": prob_1, "2": prob_2}

def extract_instruction_text(raw_instruction):
    match = re.search(r"### Instruction:\n(.*?)\n+### Response:", raw_instruction, re.DOTALL)
    return match.group(1).strip() if match else raw_instruction
