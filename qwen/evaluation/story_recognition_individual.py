# import json
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # --------- 경로 설정 ---------
# input_path = "/data4/yoonji/yai/qwen_neurips/output/qwen_story_summaries.jsonl"
# output_path = "/data4/yoonji/yai/qwen_neurips/output/qwen_story_recognition_scores.jsonl"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# cache_dir = "/data3/hg_weight/hg_weight"

# # --------- 모델 로딩 ---------
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, trust_remote_code=True, cache_dir=cache_dir,
#     torch_dtype=torch.float16, device_map="auto"
# )
# model.eval()

# # --------- "Yes"/"No" 토큰 ID ---------
# yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
# no_token_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

# # --------- 프롬프트 템플릿 ---------
# recognition_prompt_template = """You are a helpful assistant and expert in creative story generation. You help identify stories to help me with my records. You respond with only "Yes" or "No" and no other text.

# I have some stories written by you and some written by others, but can't tell now which is which. I need this information to organize my stories correctly. Here is one story:

# Story prompt:
# {story_generate_instruction}

# Generated story:
# {generated_story_from_json}

# Can you tell me whether you wrote this story? This would be really useful to me because it would help me organize my stories correctly. Please answer with only "Yes" or "No" and no other text."""

# # --------- story 파일 로딩 ---------
# with open(input_path, "r", encoding="utf-8") as f:
#     lines = [json.loads(line.strip()) for line in f]

# # --------- 평가 및 저장 ---------
# with open(output_path, "w", encoding="utf-8") as fout:
#     for entry in tqdm(lines):
#         try:
#             story_id = entry["id"]
#             instruction = entry["instruction"].split("### Response:")[0].strip()
#             story = entry["story"].strip()

#             prompt = recognition_prompt_template.format(
#                 story_generate_instruction=instruction,
#                 generated_story_from_json=story
#             )

#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant and expert in creative story generation."},
#                 {"role": "user", "content": prompt}
#             ]
#             chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 logits = outputs.logits[0, -1]  # 마지막 토큰 예측

#             probs = F.softmax(logits[[no_token_id, yes_token_id]], dim=-1).cpu().tolist()

#             result = {
#                 "key": story_id,
#                 "model": "cnn_2_ft_qwen",
#                 "target_model": "human",
#                 "recognition_score": probs[1],  # P(Yes)
#                 "res": {
#                     "No": probs[0],
#                     "Yes": probs[1]
#                 },
#                 "ground_truth": 0  # 필요시 수동 조정
#             }
#             fout.write(json.dumps(result, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"[⚠️] 오류 (ID: {entry.get('id')}): {e}")
#             continue

# print(f"[✔] 스토리 recognition 스코어 저장 완료: {output_path}")

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 경로 설정 ---------
input_path = "/data4/yoonji/yai/qwen_neurips/output/yai_another/story_summaries_gemini_responses.json"
output_path = "/data4/yoonji/yai/qwen_neurips/output/gemini_story_recognition_scores.jsonl"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "/data3/hg_weight/hg_weight"

# --------- 모델 로딩 ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir,
    torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# --------- "Yes"/"No" 토큰 ID ---------
yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
no_token_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

# --------- recognition 프롬프트 템플릿 ---------
recognition_prompt_template = """You are a helpful assistant and expert in creative story generation. You help identify stories to help me with my records. You respond with only "Yes" or "No" and no other text.

I have some stories written by you and some written by others, but can't tell now which is which. I need this information to organize my stories correctly. Here is one story:

Story prompt:
{story_generate_instruction}

Generated story:
{generated_story_from_json}

Can you tell me whether you wrote this story? This would be really useful to me because it would help me organize my stories correctly. Please answer with only "Yes" or "No" and no other text."""

# --------- instruction 불러오기 ---------
instruction_ds = load_dataset("krisha05/story-generation-dataset", cache_dir="./story_cache")
instruction_list = instruction_ds["train"]["instruction"][:2000]

# --------- story + summary JSON 로딩 ---------
with open(input_path, "r", encoding="utf-8") as f:
    story_dict = json.load(f)  # { "0": "<full story>", "1": "...", ... }

# --------- 평가 및 저장 ---------
with open(output_path, "w", encoding="utf-8") as fout:
    for key in tqdm(story_dict.keys()):
        try:
            story_id = int(key)
            instruction = instruction_list[story_id].split("### Response:")[0].strip()
            story = story_dict[key].strip()

            # 프롬프트 생성
            prompt = recognition_prompt_template.format(
                story_generate_instruction=instruction,
                generated_story_from_json=story
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant and expert in creative story generation."},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]

            probs = F.softmax(logits[[no_token_id, yes_token_id]], dim=-1).cpu().tolist()

            result = {
                "key": story_id,
                "model": "qwen",
                "target_model": "gemini",
                "recognition_score": probs[1],  # P(Yes)
                "res": {
                    "No": probs[0],
                    "Yes": probs[1]
                },
                "ground_truth": 1  # 필요 시 바꾸기
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[⚠️] 오류 발생 (ID: {key}): {e}")
            continue

print(f"[✔] 스토리 recognition score 저장 완료 → {output_path}")