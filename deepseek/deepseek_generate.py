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

    if args.task == "summary":
        dataset = load_dataset("xsum", split=f"train[:{args.num_samples}]")
    elif args.task == "story":
        dataset = load_dataset("krisha05/story-generation-dataset", split=f"train[:{args.num_samples}]")

    for idx, item in enumerate(tqdm(dataset, desc=f"Running {args.task}")):
        doc_id = item["id"] if "summary" in args.task else item.get("id", str(idx))

        if args.task == "summary":
            prompt = build_prompt(system_prompt, user_prompt_template, {"article": item["document"]})
        elif args.task == "story":
            prompt = build_prompt(system_prompt, user_prompt_template, {"instruction": item["instruction"]})

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
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

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
  