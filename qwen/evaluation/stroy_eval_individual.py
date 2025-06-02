# import json
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # --------- 설정 ---------
# input_path = "/data4/yoonji/yai/qwen_neurips/output/qwen_story_summaries.jsonl"
# output_path = "/data4/yoonji/yai/qwen_neurips/output/qwen_story_scored_likert.jsonl"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# cache_dir = "/data3/hg_weight/hg_weight"

# # --------- 모델 불러오기 ---------
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, trust_remote_code=True, cache_dir=cache_dir,
#     torch_dtype=torch.float16, device_map="auto"
# )
# model.eval()

# # --------- score 토큰 설정 ---------
# score_tokens = [str(i) for i in range(1, 6)]
# score_token_ids = tokenizer(score_tokens, add_special_tokens=False)["input_ids"]
# score_token_ids = [ids[0] for ids in score_token_ids if len(ids) == 1]

# # --------- 프롬프트 생성 함수 ---------
# def build_prompt(instruction, story):
#     clean_inst = instruction.split("### Response:")[0].strip()
#     system_prompt = (
#         "You are a helpful assistant and expert in story generation. "
#         "You help score summaries to help me with my records. "
#         "You respond with a number from 1 to 5 and no other text."
#     )
#     user_prompt = (
#         "Let's see a story that builds on the lead-in text, but keep it short.\n"
#         f"{clean_inst}\n"
#         f"story:\n{story.strip()}\n"
#         "Provide only the score with no other text."
#     )
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# # --------- 입력 데이터 로딩 ---------
# with open(input_path, "r", encoding="utf-8") as f:
#     lines = [json.loads(l) for l in f]

# # --------- 평가 및 저장 ---------
# with open(output_path, "w", encoding="utf-8") as fout:
#     for entry in tqdm(lines):
#         try:
#             story_id = entry["id"]
#             instruction = entry["instruction"]
#             story = entry["story"]

#             prompt_text = build_prompt(instruction, story)
#             inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 logits = outputs.logits[0, -1]  # 마지막 토큰 위치

#             probs = F.softmax(logits[score_token_ids], dim=-1).cpu().tolist()
#             expected_score = sum((i + 1) * p for i, p in enumerate(probs))

#             result = {
#                 "key": story_id,
#                 "target_model": "human",
#                 "scores": {str(i + 1): probs[i] for i in range(5)},
#                 "model": model_name.split("/")[-1],
#                 "score": expected_score
#             }
#             fout.write(json.dumps(result, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"[⚠️] ID {entry.get('id')} 처리 중 오류: {e}")
#             continue

# print(f"[✔] 결과 저장 완료: {output_path}")

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 설정 ---------
input_path = "/data4/yoonji/yai/qwen_neurips/output/yai_another/gemma_generated_stories_2000.json"  # JSON dict 형태
output_path = "/data4/yoonji/yai/qwen_neurips/output/gemma_story_scored_likert.jsonl"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"

# --------- 모델 불러오기 ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir,
    torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# --------- score 토큰 설정 ---------
score_tokens = [str(i) for i in range(1, 6)]
score_token_ids = tokenizer(score_tokens, add_special_tokens=False)["input_ids"]
score_token_ids = [ids[0] for ids in score_token_ids if len(ids) == 1]

# --------- instruction 로딩 ---------
instruction_ds = load_dataset("krisha05/story-generation-dataset", cache_dir="./story_cache")
instruction_list = instruction_ds["train"]["instruction"][:2000]

# --------- 프롬프트 생성 함수 ---------
def build_prompt(instruction, story):
    clean_inst = instruction.split("### Response:")[0].strip()
    system_prompt = (
        "You are a helpful assistant and expert in story generation. "
        "You help score summaries to help me with my records. "
        "You respond with a number from 1 to 5 and no other text."
    )
    user_prompt = (
        "Let's see a story that builds on the lead-in text, but keep it short.\n"
        f"{clean_inst}\n"
        f"story:\n{story.strip()}\n"
        "Provide only the score with no other text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------- story JSON 로딩 ---------
with open(input_path, "r", encoding="utf-8") as f:
    story_dict = json.load(f)  # {"0": "<story>", "1": "<story>", ...}

# --------- 평가 및 저장 ---------
with open(output_path, "w", encoding="utf-8") as fout:
    for key in tqdm(story_dict.keys()):
        try:
            story_id = int(key)
            instruction = instruction_list[story_id]
            story = story_dict[key]

            prompt_text = build_prompt(instruction, story)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # 마지막 토큰 위치

            probs = F.softmax(logits[score_token_ids], dim=-1).cpu().tolist()
            expected_score = sum((i + 1) * p for i, p in enumerate(probs))

            result = {
                "key": story_id,
                "target_model": "gemma",
                "scores": {str(i + 1): probs[i] for i in range(5)},
                "model": "qwen",
                "score": expected_score
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[⚠️] ID {key} 처리 중 오류: {e}")
            continue

print(f"[✔] Likert 평가 결과 저장 완료 → {output_path}")