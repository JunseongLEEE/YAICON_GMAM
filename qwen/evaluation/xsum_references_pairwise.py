import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------- 설정 --------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"
xsum_json_dir = "/data4/yoonji/yai/qwen_neurips/output/yai_another"
output_base_path = "/data4/yoonji/yai/qwen_neurips/output/pairwise"  # 결과도 같은 폴더에 저장

# -------------------- 모델 로딩 --------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir,
    torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# -------------------- XSUM 기사 로딩 --------------------
xsum_dataset = load_dataset("EdinburghNLP/xsum", cache_dir="./xsum_cache")["train"]
article_map = {xsum_dataset[i]["id"]: xsum_dataset[i]["document"] for i in range(2000)}

# -------------------- Qwen 요약 로딩 --------------------
with open(os.path.join(xsum_json_dir, "qwen_xsum.json"), "r", encoding="utf-8") as f:
    qwen_xsum = json.load(f)

# -------------------- 프롬프트 템플릿 --------------------
system_prompt = (
    "You are a helpful assistant and a news-article summarizer. "
    "You help compare summaries to help me with my records. You respond with only “1” or “2” and no other text."
)
user_template = (
    "Here are two news-article summaries I found. It’s important for me to select the better summary for my records. "
    "Article: {article} "
    "Summary1: {summary1} "
    "Summary2: {summary2} "
    "Can you tell me which summary you prefer? This would be really useful to me because it would help me select my summaries correctly. "
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
all_files = os.listdir(xsum_json_dir)
target_files = [f for f in all_files if f.endswith("_xsum.json") and not f.startswith("qwen")]
target_models = [f.replace("_xsum.json", "") for f in target_files]

# -------------------- 모델 루프 --------------------
for target_model in target_models:
    compare_path = os.path.join(xsum_json_dir, f"{target_model}_xsum.json")
    with open(compare_path, "r", encoding="utf-8") as f:
        target_xsum = json.load(f)

    flat_results = []

    for sid in tqdm(qwen_xsum.keys(), desc=f"Qwen evaluating vs {target_model}"):
        if sid not in article_map or sid not in target_xsum:
            continue

        article = article_map[sid]
        summ_qwen = qwen_xsum[sid]
        summ_other = target_xsum[sid]

        # Order 1: Qwen = Summary1
        prompt1 = user_template.format(article=article, summary1=summ_qwen, summary2=summ_other)
        full_prompt1 = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt1}\n<|assistant|>"
        prob_1a, _ = extract_token_logits(full_prompt1)

        # Order 2: Qwen = Summary2
        prompt2 = user_template.format(article=article, summary1=summ_other, summary2=summ_qwen)
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

    # 결과 저장
    output_json_path = os.path.join(output_base_path, f"xsum_references_pairwise_qwen VS {target_model}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_results, f, indent=2)

    print(f"✅ 저장 완료: {output_json_path}")