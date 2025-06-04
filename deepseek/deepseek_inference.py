import torch
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

if __name__ == "__main__":
    args = parse_args()
    system_prompt, user_prompt_template = load_prompt_template(args.prompt_file)

    model_id = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    results = {}
    likert_scores = []
    yes_probs = []
    no_probs = []
    first_order_probs = []
    second_order_probs = []
    final_confidences = []

    if args.task in ["likert_summary", "likert_story", "recognition_summary", "recognition_story"]:
        if not args.generated_file:
            raise ValueError("--generated_file is required for this task.")
        if isinstance(args.generated_file, list):
            if len(args.generated_file) != 1:
                raise ValueError("--generated_file must be a single file path for this task.")
            generated_file_path = args.generated_file[0]
        else:
            generated_file_path = args.generated_file
    
        with open(generated_file_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)
        if "summary" in args.task:
            dataset = load_dataset("xsum", split=f"train[:{args.num_samples}]")
        else:
            dataset = load_dataset("krisha05/story-generation-dataset", split=f"train[:{args.num_samples}]")
    elif args.task in ["pairwise_summary", "pairwise_story", "pairwise_comparison_summary", "pairwise_comparison_story"]:
        if len(args.generated_file) != 2:
            raise ValueError("--generated_file must contain exactly two files for pairwise tasks.")
        with open(args.generated_file[0], "r", encoding="utf-8") as f1, \
             open(args.generated_file[1], "r", encoding="utf-8") as f2:
            gen_data_1 = json.load(f1)
            gen_data_2 = json.load(f2)
        if "summary" in args.task:
            dataset = load_dataset("xsum", split=f"train[:{args.num_samples}]")
        else:
            dataset = load_dataset("krisha05/story-generation-dataset", split=f"train[:{args.num_samples}]")
    else:
        raise ValueError("Unsupported task")

    for idx, item in enumerate(tqdm(dataset, desc=f"Running {args.task}")):
        doc_id = item["id"] if "summary" in args.task else item.get("id", str(idx))
        if args.task == "likert_summary":
            if doc_id not in gen_data:
                continue
            prompt = build_prompt(system_prompt, user_prompt_template, {
                "article": item["document"],
                "summary": gen_data[doc_id]
            })
        elif args.task == "likert_story":
            if doc_id not in gen_data:
                continue
            clean_instruction = extract_instruction_text(item["instruction"])
            prompt = build_prompt(system_prompt, user_prompt_template, {
                "instruction": clean_instruction,
                "story": gen_data[doc_id]
            })
        elif args.task == "recognition_summary":
            if doc_id not in gen_data:
                continue
            prompt = build_prompt(system_prompt, user_prompt_template, {
                "article": item["document"],
                "summary": gen_data[doc_id]
            })
        elif args.task == "recognition_story":
            if doc_id not in gen_data:
                continue
            clean_instruction = extract_instruction_text(item["instruction"])
            prompt = build_prompt(system_prompt, user_prompt_template, {
                "instruction": clean_instruction,
                "story": gen_data[doc_id]
            })
        elif args.task.startswith("pairwise"):
            if doc_id not in gen_data_1 or doc_id not in gen_data_2:
                continue

            clean_instruction = extract_instruction_text(item["instruction"]) if "story" in args.task else ""

            content_dicts = []

            if args.task in ["pairwise_summary", "pairwise_comparison_summary"]:
                content_dicts = [
                    {"article": item["document"], "summary1": gen_data_1[doc_id], "summary2": gen_data_2[doc_id]},
                    {"article": item["document"], "summary1": gen_data_2[doc_id], "summary2": gen_data_1[doc_id]}
                ]
            elif args.task in ["pairwise_story", "pairwise_comparison_story"]:
                content_dicts = [
                    {"instruction": clean_instruction, "story1": gen_data_1[doc_id], "story2": gen_data_2[doc_id]},
                    {"instruction": clean_instruction, "story1": gen_data_2[doc_id], "story2": gen_data_1[doc_id]}
                ]

            all_probs = []
            for content in content_dicts:
                prompt = build_prompt(system_prompt, user_prompt_template, content)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                with torch.no_grad():
                    output = model(**inputs)
                    choice_probs = compute_choice_probs(output.logits, tokenizer)
                    all_probs.append(choice_probs)
                    
            confidence_score = (all_probs[0]["1"] + all_probs[1]["2"]) / 2

            results[doc_id] = {
                "first_order_probs": all_probs[0],
                "second_order_probs": all_probs[1],
                "confidence_score": confidence_score
            }
            first_order_probs.append(all_probs[0]["1"])
            second_order_probs.append(all_probs[1]["2"])
            final_confidences.append(confidence_score)
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            if args.task.startswith("likert"):
                output = model(**inputs)
                expected_score, prob_dict = compute_expected_score(output.logits, tokenizer)
                results[str(doc_id)] = {
                    "likert_score": expected_score,
                    "score_probabilities": prob_dict
                }
                likert_scores.append(expected_score)
            elif args.task.startswith("recognition"):
                output = model(**inputs)
                yes_no_probs = compute_yes_no_probs(output.logits, tokenizer)
                results[str(doc_id)] = yes_no_probs
                yes_probs.append(yes_no_probs["Yes"])
                no_probs.append(yes_no_probs["No"])
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                result_text = decoded.split("<|assistant|>\n")[-1].strip()
                results[str(doc_id)] = result_text

    final_output = {
        "target_model": args.target_model,
        "results": results
    }

    if args.task.startswith("likert") and likert_scores:
        final_output["summary_statistics"] = {
            "preference": {
                "mean_likert_score": round(float(np.mean(likert_scores)), 4),
                "std_likert_score": round(float(np.std(likert_scores)), 4)
            }
        }
    elif args.task.startswith("recognition") and yes_probs and no_probs:
        final_output["summary_statistics"] = {
            "recognition": {
                "mean_yes_probability": round(float(np.mean(yes_probs)), 4),
                "mean_no_probability": round(float(np.mean(no_probs)), 4)
            }
        }
    elif args.task in ["pairwise_summary", "pairwise_story"] and final_confidences:
        final_output["summary_statistics"] = {
            "detection": {
                "First order mean": round(float(np.mean(first_order_probs)), 4),
                "Second order mean": round(float(np.mean(second_order_probs)), 4),
                "mean_confidence_score": round(float(np.mean(final_confidences)), 4),
                "std_confidence_score": round(float(np.std(final_confidences)), 4)
            }
        }
    elif args.task in ["pairwise_comparison_summary", "pairwise_comparison_story"] and final_confidences:
        final_output["summary_statistics"] = {
            "comparison": {
                "First order mean": round(float(np.mean(first_order_probs)), 4),
                "Second order mean": round(float(np.mean(second_order_probs)), 4),
                "mean_confidence_score": round(float(np.mean(final_confidences)), 4),
                "std_confidence_score": round(float(np.std(final_confidences)), 4)
            }
        }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
