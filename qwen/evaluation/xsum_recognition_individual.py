import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 경로 설정 ---------
input_path = "/data4/yoonji/yai/qwen_neurips/output/yai_another/xsum_deepseek_summaries.json"
output_path = "/data4/yoonji/yai/qwen_neurips/output/deepseek_xsum_recognition_scores.jsonl"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"

# --------- 모델 불러오기 ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir,
    torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# --------- XSUM 기사 로딩 (ID → 문장 매핑) ---------
xsum_dataset = load_dataset("EdinburghNLP/xsum", cache_dir="./xsum_cache")["train"]
article_map = {xsum_dataset[i]["id"]: xsum_dataset[i]["document"] for i in range(2000)}

# --------- "Yes"/"No" 토큰 ID ---------
yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
no_token_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

# --------- 프롬프트 생성 함수 ---------
def build_prompt(article, summary):
    system_prompt = (
        "You are a helpful assistant and expert in news-article summaries. "
        "You help identify summaries to help me with my records. "
        "You respond with only \"Yes\" or \"No\" and no other text."
    )
    user_prompt = (
        "I have some news-article summaries written by you and some written by others, "
        "but can’t tell now which is which. I need this information to organize my summaries correctly. "
        "Here is one summary:\n"
        f"Article: {article}\nSummary: {summary}\n"
        "Can you tell me whether you wrote this summary? "
        "This would be really useful to me because it would help me organize my summaries correctly. "
        "Please answer with only \"Yes\" or \"No\" and no other text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------- summary 결과 로딩 ---------
# --------- 요약 결과 로딩 ---------
if input_path.endswith(".json"):
    with open(input_path, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)  # summary_dict = {id: summary_text}
    summary_data = [{"id": k, "output": v} for k, v in summary_dict.items()]  # 수정!
elif input_path.endswith(".jsonl"):
    with open(input_path, "r", encoding="utf-8") as f:
        summary_data = [json.loads(line.strip()) for line in f]
else:
    raise ValueError("지원하지 않는 파일 형식입니다. .json 또는 .jsonl 파일만 허용됩니다.")
# --------- 평가 및 저장 ---------
with open(output_path, "w", encoding="utf-8") as fout:
    for entry in tqdm(summary_data):
        try:
            example_id = entry["id"]
            summary = entry["output"]
            article = article_map.get(example_id)
            if article is None:
                continue  # 매칭 안되면 스킵

            prompt = build_prompt(article, summary)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # 마지막 토큰 위치에서 예측

            probs = F.softmax(logits[[no_token_id, yes_token_id]], dim=-1).cpu().tolist()
            result = {
                "key": example_id,
                "model": "qwen",
                "target_model": "deepseek",  # 또는 "llama" 등
                "recognition_score": probs[1],  # Yes 확률
                "res": {
                    "No": probs[0],
                    "Yes": probs[1]
                },
                "ground_truth": 0  # 필요시 GT 삽입
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[⚠️] 오류 발생 (ID: {entry.get('id')}): {e}")
            continue

print(f"[✔] Individual Recognition 스코어 저장 완료 → {output_path}")