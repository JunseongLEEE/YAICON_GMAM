import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------- 설정 --------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"
story_json_dir = "/data4/yoonji/yai/qwen_neurips/output/yai_another"
output_base_path = "/data4/yoonji/yai/qwen_neurips/output/pairwise"

# -------------------- 모델 로딩 --------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir,
    torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# -------------------- Instruction 로딩 --------------------
instruction_ds = load_dataset("krisha05/story-generation-dataset", cache_dir="./story_cache")
instruction_list = instruction_ds["train"]["instruction"][:2000]

# -------------------- 프롬프트 템플릿 --------------------
system_prompt = (
    "You are a helpful assistant and a story critic who evaluates short stories based on given instructions. "
    "You help compare two stories and select the better one according to the instruction. You respond with only “1” or “2” and no other text."
)
user_template = (
    "Here are two short stories written based on the same instruction. "
    "Instruction: {instruction} "
    "Story 1: {summary1} "
    "Story 2: {summary2} "
    "Which story better follows the instruction and is overall better? This would be really useful to help me organize my story records. "
    "Please answer with only “1” or “2” and no other text."
)

# -------------------- 확률 계산 함수 --------------------
def extract_token_logits(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model(**input_ids)
        logits = output.logits[0, -1]

    token_1 = tokenizer("1", add_special_tokens=False).input_ids[0]
    token_2 = tokenizer("2", add_special_tokens=False).input_ids[0]

    logit_1 = logits[token_1].item()
    logit_2 = logits[token_2].item()
    probs = torch.softmax(torch.tensor([logit_1, logit_2]), dim=0)

    return round(probs[0].item(), 6), round(probs[1].item(), 6)

# -------------------- JSON 파일 리스트 --------------------
all_files = os.listdir(story_json_dir)
target_files = [f for f in all_files if f.endswith("_stories.json") and not f.startswith("qwen")]
target_models = [f.replace("_stories.json", "") for f in target_files]

# -------------------- Qwen 스토리 로딩 --------------------
with open(os.path.join(story_json_dir, "qwen_stories.json"), "r", encoding="utf-8") as f:
    qwen_stories = json.load(f)

# -------------------- 모델 루프 --------------------
for target_model in target_models:
    compare_path = os.path.join(story_json_dir, f"{target_model}_stories.json")
    with open(compare_path, "r", encoding="utf-8") as f:
        target_stories = json.load(f)

    flat_results = []

    for sid in tqdm(qwen_stories.keys(), desc=f"Qwen story vs {target_model}"):
        idx = int(sid)
        if sid not in target_stories or idx >= len(instruction_list):
            continue

        instruction = instruction_list[idx]
        story_qwen = qwen_stories[sid]
        story_other = target_stories[sid]

        # Order 1: Qwen = Story 1
        prompt1 = user_template.format(instruction=instruction, summary1=story_qwen, summary2=story_other)
        full_prompt1 = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt1}\n<|assistant|>"
        prob_1a, _ = extract_token_logits(full_prompt1)

        # Order 2: Qwen = Story 2
        prompt2 = user_template.format(instruction=instruction, summary1=story_other, summary2=story_qwen)
        full_prompt2 = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt2}\n<|assistant|>"
        _, prob_2b = extract_token_logits(full_prompt2)

        self_pref = round((prob_1a + prob_2b) / 2, 6)

        flat_results.append({
            "key": sid,
            "base_model": "qwen",
            "target_model": target_model,
            "forward_comparison": "1",
            "forward_comparison_probability": prob_1a,
            "backward_comparison": "2",
            "backward_comparison_probability": prob_2b,
            "self_preference": self_pref
        })

    output_json_path = os.path.join(output_base_path, f"story_references_qwen VS {target_model}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_results, f, indent=2)

    print(f"✅ 저장 완료: {output_json_path}")