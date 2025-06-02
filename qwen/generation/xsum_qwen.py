import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset

# ✅ CUDA 환경 설정 (코드 상에서 가능하나 보통은 외부에서 설정 권장)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --------- 모델 로드 ---------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16,
    trust_remote_code=True, cache_dir=cache_dir
)
model.eval()

# --------- 요약 생성 함수 ---------
def generate_summary(article_text):
    system_prompt = (
        "You are a news-article summarizer. Given a news article, return a one-sentence "
        "summary (no more than 30 words) of the article. Return only the one-sentence summary with no other text."
    )
    user_prompt = (
        f"Article:\n{article_text}\n\n"
        "Provide a one-sentence summary (no more than 30 words) with no other text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False  # temperature 제거
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output_text

# --------- XSUM 로딩 및 2000개 사용 ---------
xsum_dataset = load_dataset("EdinburghNLP/xsum", cache_dir="./xsum_cache")
articles = xsum_dataset["train"]["document"][:2000]

# --------- 요약 생성 및 실시간 저장 (JSONL) ---------
output_path = "qwen_xsum_summaries.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for idx, article in tqdm(enumerate(articles), total=len(articles)):
        summary = generate_summary(article)
        item = {
            "id": idx,
            "article": article,
            "summary": summary,
            "prompt_type": "XSUM"
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")  # JSONL 형식으로 저장

print(f"[✔] 모든 요약이 '{output_path}' 파일에 JSONL 형식으로 저장되었습니다.")