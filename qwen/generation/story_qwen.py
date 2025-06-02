# import os
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from tqdm import tqdm
# from datasets import load_dataset

# # ✅ CUDA 환경 설정 (코드 상에서 가능하나 보통은 외부에서 설정 권장)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # --------- 모델 로드 ---------
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# cache_dir = "/data3/hg_weight/hg_weight"

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name, trust_remote_code=True, cache_dir=cache_dir
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, device_map="auto", torch_dtype=torch.float16,
#     trust_remote_code=True, cache_dir=cache_dir
# )
# model.eval()

# # --------- 스토리 생성 함수 ---------
# def generate_summary(instruction_text):
#     system_prompt = (
#         "Given a writing prompt, craft a complete short story with clear beginning, middle, and end. Include descriptive details, dialogue, and emotional depth. Keep the story between return a five-sentence summary."
#     )
#     user_prompt = (
#         f"Story Prompt:\n{instruction_text}\n\n"
#         "Write an engaging short story about 300 words with no other text."
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=300,  # 이야기 분량 고려
#         do_sample=False
#     )

#     generated_ids = outputs[0][inputs.input_ids.shape[1]:]
#     output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
#     return output_text


# # generation
# '''
# from datasets import load_dataset

# ds = load_dataset("krisha05/story-generation-dataset")
# ds["train"]["instruction"] => len 하면 4000개 나옴
# (Pdb) ds["train"]["instruction"][0]
# '### Instruction:\nWith the suggestion text as a catalyst, concoct a short story.\n\n### Response:\n'
# ''' 
# # --------- XSUM 로딩 및 2000개 사용 ---------
# ds = load_dataset("krisha05/story-generation-dataset", cache_dir="./story_cache")
# articles = ds["train"]["instruction"][:2000]

# # --------- 요약 생성 및 실시간 저장 (JSONL) ---------
# output_path = "qwen_story_summaries.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for idx, article in tqdm(enumerate(articles), total=len(articles)):
#         summary = generate_summary(article)
#         item = {
#             "id": idx,
#             "instruction": article,
#             "story": summary
#         }
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")  # JSONL 형식으로 저장

# print(f"[✔] 모든 요약이 '{output_path}' 파일에 JSONL 형식으로 저장되었습니다.")
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset

# ✅ CUDA 환경 설정 (외부에서 설정해도 됨)
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

# --------- 스토리 생성 함수 ---------
def generate_summary(instruction_text):
    system_prompt = (
        "Given a writing prompt, craft a complete short story with clear beginning, middle, and end. "
        "Include descriptive details, dialogue, and emotional depth. Keep the story between return a five-sentence summary."
    )
    user_prompt = (
        f"Story Prompt:\n{instruction_text.strip()}\n\n"
        "Write an engaging short story about 300 words with no other text."
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
        max_new_tokens=300,  # 약 300단어 분량
        do_sample=False
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output_text

# --------- 데이터셋 로딩 (2000개만 사용) ---------
ds = load_dataset("krisha05/story-generation-dataset", cache_dir="./story_cache")
instructions = ds["train"]["instruction"][:2000]

# --------- 요약 생성 및 실시간 저장 ---------
output_path = "qwen_story_summaries.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for idx, instruction in tqdm(enumerate(instructions), total=len(instructions)):
        story = generate_summary(instruction)
        item = {
            "id": idx,
            "instruction": instruction,
            "story": story
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")  # JSONL 형식

print(f"[✔] 모든 스토리가 '{output_path}'에 저장되었습니다.")