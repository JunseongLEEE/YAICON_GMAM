import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 경로 설정 ---------
input_path = "/data4/yoonji/yai/qwen_neurips/output/yai_another/xsum_deepseek_summaries.json"
output_path = "/data4/yoonji/yai/qwen_neurips/output/deepseek_xsum_scored_likert.jsonl"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"

# --------- 모델 로딩 ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                             torch_dtype=torch.float16, device_map="auto")
model.eval()

# --------- 점수 토큰 사전 준비 ---------
score_tokens = [str(i) for i in range(1, 6)]
score_token_ids = tokenizer(score_tokens, add_special_tokens=False)["input_ids"]
score_token_ids = [ids[0] for ids in score_token_ids if len(ids) == 1]

# --------- XSUM 원본 기사 로딩 ---------
xsum_dataset = load_dataset("EdinburghNLP/xsum", cache_dir="./xsum_cache")["train"]
article_map = {xsum_dataset[i]["id"]: xsum_dataset[i]["document"] for i in range(2000)}

# --------- 요약 결과 로딩 ---------
if input_path.endswith(".json"):
    with open(input_path, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)  # summary_dict = {id: summary_text}
    entries = [{"id": k, "output": v} for k, v in summary_dict.items()]  # 수정!
elif input_path.endswith(".jsonl"):
    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f]
else:
    raise ValueError("지원하지 않는 파일 형식입니다. .json 또는 .jsonl 파일만 허용됩니다.")

# --------- 평가 프롬프트 생성 함수 ---------
def build_prompt(article, summary):
    system_prompt = (
        "You are a helpful assistant and expert in news-article summaries. "
        "You help score summaries to help me with my records. "
        "You respond with a number from 1 to 5 and no other text."
    )
    user_prompt = (
        f"Article: {article}\n"
        f"Summary: {summary}\n"
        "Provide only the score with no other text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------- 평가 수행 및 저장 ---------
with open(output_path, "w", encoding="utf-8") as fout:
    for entry in tqdm(entries):
        article_id = entry["id"]
        summary = entry["output"]

        # 해당 article 찾기
        article_text = article_map.get(article_id)
        if article_text is None:
            print(f"⚠️ 기사 ID {article_id} 누락 → 건너뜀")
            continue

        prompt_text = build_prompt(article_text, summary)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # 마지막 토큰 예측 결과
            # breakpoint()
        probs = F.softmax(logits[score_token_ids], dim=-1).cpu().tolist()
        expected_score = sum([p * (i + 1) for i, p in enumerate(probs)])

        result = {
            "key": article_id,
            "target_model": "deepseek",  # 필요시 "llama" 등으로 수정 가능
            "scores": {str(i + 1): probs[i] for i in range(5)},
            "model": "qwen",
            "score": expected_score
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"[✔] Likert 점수 추정이 완료되었습니다: {output_path}")