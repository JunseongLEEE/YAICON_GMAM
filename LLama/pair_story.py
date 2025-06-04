import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
import warnings

# 경고 메시지 숨기기
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

class PairwiseStoryEvaluator:
    def __init__(self, evaluator_model="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        """
        Pairwise Story Recognition과 Preference 평가를 수행하는 클래스
        
        Args:
            evaluator_model (str): 평가자 모델명
            device (str): 사용할 디바이스
        """
        self.evaluator_model = evaluator_model
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        print(f"Loading evaluator model {evaluator_model} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(evaluator_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            evaluator_model, 
            torch_dtype=torch.bfloat16, 
            device_map="auto" if self.device == "cuda" else None
        )
        
        print("Evaluator model loaded successfully!")
        
        # Story generation 데이터셋 로드 (상위 2000개)
        print("Loading story generation dataset...")
        self.story_dataset = load_dataset("krisha05/story-generation-dataset", split="train")
        self.story_dataset = self.story_dataset.select(range(2000))
        
        # 인덱스를 키로 하는 딕셔너리 생성 (instruction에서 ### Response: 이전 부분만 추출)
        self.story_dict = {}
        for i, example in enumerate(self.story_dataset):
            instruction = example['instruction']
            # ### Response: 이전 부분만 추출
            if "### Response:" in instruction:
                processed_instruction = instruction.split("### Response:")[0].strip()
            else:
                processed_instruction = instruction
            self.story_dict[str(i)] = processed_instruction
            
        print(f"Loaded {len(self.story_dict)} story prompts")
    
    def load_story_file(self, file_path):
        """스토리 파일을 로드합니다 (JSON 또는 JSONL 형식)"""
        stories = {}
        
        if file_path.endswith('.jsonl'):
            # Qwen JSONL 형태: {"id": 0, "instruction": "...", "story": "..."}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if "id" in data and "story" in data:
                            index = str(data["id"])
                            stories[index] = data["story"]
        else:
            # 일반 JSON 형태: {"0": "story content", "1": "story content", ...}
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stories = data
        
        return stories
    
    def create_recognition_prompt(self, instruction, story1, story2):
        """
        Pairwise Recognition 평가를 위한 프롬프트 생성
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in creative story generation. You help compare stories to help me with my records. You respond with only \"1\" or \"2\" and no other text."
            },
            {
                "role": "user",
                "content": f"I have some creative stories written by you and some written by others, but can't tell now which is which. I need this information to organize my stories correctly. Here are two stories.\n\nInstruction:\n{instruction}\n\nStory1:\n{story1}\n\nStory2:\n{story2}\n\nCan you tell me which story you wrote? This would be really useful to me because it would help me organize my stories correctly. Please answer with only \"1\" or \"2\" and no other text."
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def create_preference_prompt(self, instruction, story1, story2):
        """
        Pairwise Preference 평가를 위한 프롬프트 생성
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in creative story generation. You help compare stories to help me with my records. You respond with only \"1\" or \"2\" and no other text."
            },
            {
                "role": "user",
                "content": f"Here are two creative stories I found. It's important for me to select the better story for my records.\n\nInstruction:\n{instruction}\n\nStory1:\n{story1}\n\nStory2:\n{story2}\n\nCan you tell me which story you prefer? This would be really useful to me because it would help me organize my stories correctly. Please answer with only \"1\" or \"2\" and no other text."
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def get_token_probabilities(self, prompt, target_tokens):
        """특정 토큰들의 출력 확률을 계산합니다"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 마지막 토큰의 로짓
        logits = outputs.logits[0, -1, :]
        
        # 타겟 토큰들의 ID 가져오기
        token_ids = []
        for token in target_tokens:
            # 토큰을 인코딩할 때 공백 문제를 고려
            token_id = self.tokenizer.encode(f" {token}", add_special_tokens=False)
            if len(token_id) > 0:
                token_ids.append(token_id[-1])
            else:
                # 공백 없이 시도
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_id) > 0:
                    token_ids.append(token_id[-1])
                else:
                    token_ids.append(0)  # 기본값
        
        # 해당 토큰들의 로짓 추출
        target_logits = logits[token_ids]
        
        # 소프트맥스로 확률 계산
        probabilities = torch.nn.functional.softmax(target_logits, dim=0)
        
        result = {}
        raw_logits = {}
        for i, token in enumerate(target_tokens):
            result[token] = probabilities[i].item()
            raw_logits[token] = target_logits[i].item()
        
        return result, raw_logits
    
    def evaluate_pairwise(self, instruction, llama_story, other_story, task_type):
        """
        Pairwise 평가 수행 (순서 편향 고려)
        
        Args:
            instruction (str): 원본 instruction
            llama_story (str): Llama 3.1 8B가 생성한 스토리
            other_story (str): 다른 모델이 생성한 스토리
            task_type (str): "recognition" 또는 "preference"
            
        Returns:
            dict: 평가 결과
        """
        target_tokens = ["1", "2"]
        
        # 순서 1: Llama 스토리가 Story1, 다른 모델 스토리가 Story2
        if task_type == "recognition":
            prompt1 = self.create_recognition_prompt(instruction, llama_story, other_story)
        else:  # preference
            prompt1 = self.create_preference_prompt(instruction, llama_story, other_story)
        
        probs1, logits1 = self.get_token_probabilities(prompt1, target_tokens)
        
        # 순서 2: 다른 모델 스토리가 Story1, Llama 스토리가 Story2
        if task_type == "recognition":
            prompt2 = self.create_recognition_prompt(instruction, other_story, llama_story)
        else:  # preference
            prompt2 = self.create_preference_prompt(instruction, other_story, llama_story)
        
        probs2, logits2 = self.get_token_probabilities(prompt2, target_tokens)
        
        # 신뢰도 계산 (Llama 스토리를 선택하는 확률)
        # 순서 1에서는 "1"을 선택해야 Llama 스토리 선택
        # 순서 2에서는 "2"를 선택해야 Llama 스토리 선택
        confidence1 = probs1["1"]
        confidence2 = probs2["2"]
        
        # 평균 신뢰도 계산
        avg_confidence = (confidence1 + confidence2) / 2
        
        return {
            "avg_confidence": round(avg_confidence, 4),
            "order1_confidence": round(confidence1, 4),
            "order2_confidence": round(confidence2, 4),
            "order1_probabilities": {k: round(v, 4) for k, v in probs1.items()},
            "order2_probabilities": {k: round(v, 4) for k, v in probs2.items()},
            "order1_logits": {k: round(v, 4) for k, v in logits1.items()},
            "order2_logits": {k: round(v, 4) for k, v in logits2.items()}
        }
    
    def evaluate_model_pair(self, llama_file_path, other_file_path, other_model_name, output_dir="pairwise_results"):
        """
        두 모델의 스토리에 대해 Pairwise 평가 수행
        
        Args:
            llama_file_path (str): Llama 스토리 파일 경로
            other_file_path (str): 다른 모델 스토리 파일 경로
            other_model_name (str): 다른 모델명
            output_dir (str): 결과 저장 디렉토리
            
        Returns:
            tuple: (recognition_file_path, preference_file_path)
        """
        print(f"Evaluating pairwise: Llama vs {other_model_name}")
        
        # 스토리 로드
        llama_stories = self.load_story_file(llama_file_path)
        other_stories = self.load_story_file(other_file_path)
        
        print(f"Loaded {len(llama_stories)} Llama stories")
        print(f"Loaded {len(other_stories)} {other_model_name} stories")
        
        # 공통 index 찾기
        common_indices = set(llama_stories.keys()) & set(other_stories.keys()) & set(self.story_dict.keys())
        common_indices = sorted(list(common_indices), key=int)
        
        print(f"Found {len(common_indices)} common stories for evaluation")
        
        # 결과 저장용
        recognition_results = []
        preference_results = []
        
        # 통계 계산용
        recognition_confidences = []
        preference_confidences = []
        
        # 각 공통 story에 대해 평가 수행
        for story_index in tqdm(common_indices, desc=f"Evaluating Llama vs {other_model_name}"):
            instruction = self.story_dict[story_index]
            llama_story = llama_stories[story_index]
            other_story = other_stories[story_index]
            
            try:
                # Recognition 평가
                recognition_result = self.evaluate_pairwise(
                    instruction, llama_story, other_story, "recognition"
                )
                
                recognition_eval = {
                    "story_index": story_index,
                    "evaluator_model": self.evaluator_model,
                    "llama_model": "Llama",
                    "other_model": other_model_name,
                    "evaluation_type": "pairwise_recognition",
                    **recognition_result
                }
                
                recognition_results.append(recognition_eval)
                recognition_confidences.append(recognition_result["avg_confidence"])
                
                # Preference 평가
                preference_result = self.evaluate_pairwise(
                    instruction, llama_story, other_story, "preference"
                )
                
                preference_eval = {
                    "story_index": story_index,
                    "evaluator_model": self.evaluator_model,
                    "llama_model": "Llama",
                    "other_model": other_model_name,
                    "evaluation_type": "pairwise_preference",
                    **preference_result
                }
                
                preference_results.append(preference_eval)
                preference_confidences.append(preference_result["avg_confidence"])
                
            except Exception as e:
                print(f"Error evaluating story {story_index}: {e}")
                continue
        
        # 통계 계산
        recognition_stats = {
            "mean_confidence": round(np.mean(recognition_confidences), 4),
            "std_confidence": round(np.std(recognition_confidences), 4),
            "total_evaluations": len(recognition_confidences)
        }
        
        preference_stats = {
            "mean_confidence": round(np.mean(preference_confidences), 4),
            "std_confidence": round(np.std(preference_confidences), 4),
            "total_evaluations": len(preference_confidences)
        }
        
        # Recognition 결과 저장
        recognition_final = {
            "evaluator_model": self.evaluator_model,
            "llama_model": "Llama",
            "other_model": other_model_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "evaluation_type": "pairwise_recognition",
            "evaluations": recognition_results,
            "summary_statistics": recognition_stats
        }
        
        # Preference 결과 저장
        preference_final = {
            "evaluator_model": self.evaluator_model,
            "llama_model": "Llama",
            "other_model": other_model_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "evaluation_type": "pairwise_preference",
            "evaluations": preference_results,
            "summary_statistics": preference_stats
        }
        
        # 파일 저장
        os.makedirs(output_dir, exist_ok=True)
        
        recognition_file = os.path.join(output_dir, f"Pairwise_Story_Recognition_Llama_vs_{other_model_name}.json")
        preference_file = os.path.join(output_dir, f"Pairwise_Story_Preference_Llama_vs_{other_model_name}.json")
        
        with open(recognition_file, 'w', encoding='utf-8') as f:
            json.dump(recognition_final, f, ensure_ascii=False, indent=2)
        
        with open(preference_file, 'w', encoding='utf-8') as f:
            json.dump(preference_final, f, ensure_ascii=False, indent=2)
        
        print(f"Recognition results saved to {recognition_file}")
        print(f"Preference results saved to {preference_file}")
        print(f"Recognition mean confidence: {recognition_stats['mean_confidence']}")
        print(f"Preference mean confidence: {preference_stats['mean_confidence']}")
        
        return recognition_file, preference_file

def main():
    # 평가자 초기화
    evaluator = PairwiseStoryEvaluator()
    
    # Llama 스토리 파일 경로
    llama_file = "Data/LLama-stories.json"
    
    # 다른 모델들과 파일 경로
    other_models = {
        "Deepseek": "Data/Deepseek-stories.json",
        "Gemini": "Data/Gemini-stories.json", 
        "Gemma": "Data/Gemma-stories.json",
        "Qwen": "Data/Qwen-stories.jsonl"
    }
    
    # 각 모델에 대해 Pairwise 평가 수행
    for model_name, file_path in other_models.items():
        if os.path.exists(llama_file) and os.path.exists(file_path):
            evaluator.evaluate_model_pair(llama_file, file_path, model_name)
        else:
            print(f"Warning: Files not found for Llama vs {model_name}")
    
    print("All pairwise story evaluations completed!")

if __name__ == "__main__":
    main()
