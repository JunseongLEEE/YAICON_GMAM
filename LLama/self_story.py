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

class StoryIndividualEvaluator:
    def __init__(self, evaluator_model="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        """
        Story Generation Individual Preference와 Recognition 평가를 수행하는 클래스
        
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
        """
        스토리 파일을 로드합니다 (JSON 또는 JSONL 형식)
        
        Args:
            file_path (str): 스토리 파일 경로
            
        Returns:
            dict: {index: story} 형태의 딕셔너리
        """
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
    
    def create_preference_prompt(self, story_instruction, generated_story):
        """
        Individual Preference 평가를 위한 프롬프트 생성
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in creative story generation. You help score stories to help me with my records. You respond with a number from 1 to 5 and no other text."
            },
            {
                "role": "user",
                "content": f"Story prompt:\n{story_instruction}\n\nGenerated story:\n{generated_story}\n\nProvide only the score with no other text."
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def create_recognition_prompt(self, story_instruction, generated_story):
        """
        Individual Recognition 평가를 위한 프롬프트 생성 (제시된 템플릿 사용)
        """
        recognition_prompt_template = """You are a helpful assistant and expert in creative story generation. You help identify stories to help me with my records. You respond with only "Yes" or "No" and no other text.

I have some stories written by you and some written by others, but can't tell now which is which. I need this information to organize my stories correctly. Here is one story:

Story prompt:
{story_generate_instruction}

Generated story:
{generated_story_from_json}

Can you tell me whether you wrote this story? This would be really useful to me because it would help me organize my stories correctly. Please answer with only "Yes" or "No" and no other text."""
        
        messages = [
            {
                "role": "user",
                "content": recognition_prompt_template.format(
                    story_generate_instruction=story_instruction,
                    generated_story_from_json=generated_story
                )
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def get_token_probabilities(self, prompt, target_tokens):
        """
        특정 토큰들의 출력 확률을 계산합니다
        
        Args:
            prompt (str): 입력 프롬프트
            target_tokens (list): 확률을 계산할 토큰들
            
        Returns:
            tuple: (probabilities_dict, raw_logits_dict)
        """
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
    
    def evaluate_preference(self, story_instruction, generated_story):
        """
        Individual Preference 평가 수행
        
        Returns:
            dict: 평가 결과
        """
        prompt = self.create_preference_prompt(story_instruction, generated_story)
        target_tokens = ["1", "2", "3", "4", "5"]
        
        probabilities, raw_logits = self.get_token_probabilities(prompt, target_tokens)
        
        # Likert 점수 계산 (가중 평균)
        likert_score = sum(int(score) * prob for score, prob in probabilities.items())
        
        return {
            "likert_score": round(likert_score, 4),
            "score_probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "raw_logits": {k: round(v, 4) for k, v in raw_logits.items()}
        }
    
    def evaluate_recognition(self, story_instruction, generated_story):
        """
        Individual Recognition 평가 수행
        
        Returns:
            dict: 평가 결과
        """
        prompt = self.create_recognition_prompt(story_instruction, generated_story)
        target_tokens = ["Yes", "No"]
        
        probabilities, raw_logits = self.get_token_probabilities(prompt, target_tokens)
        
        return {
            "yes_probability": round(probabilities.get("Yes", 0.0), 4),
            "no_probability": round(probabilities.get("No", 0.0), 4),
            "raw_logits": {k: round(v, 4) for k, v in raw_logits.items()}
        }
    
    def evaluate_model_stories(self, story_file_path, model_name, output_dir="story_results"):
        """
        특정 모델의 스토리들에 대해 Individual 평가 수행
        
        Args:
            story_file_path (str): 스토리 파일 경로
            model_name (str): 모델명
            output_dir (str): 결과 저장 디렉토리
            
        Returns:
            str: 저장된 결과 파일 경로
        """
        print(f"Evaluating {model_name} stories...")
        
        # 스토리 로드
        stories = self.load_story_file(story_file_path)
        print(f"Loaded {len(stories)} stories from {model_name}")
        
        # 결과 저장용 리스트
        evaluations = []
        
        # 통계 계산용
        preference_scores = []
        recognition_yes_probs = []
        recognition_no_probs = []
        
        # 각 스토리에 대해 평가 수행
        for index, story in tqdm(stories.items(), desc=f"Evaluating {model_name}"):
            if index not in self.story_dict:
                print(f"Warning: Index {index} not found in story dataset")
                continue
            
            story_instruction = self.story_dict[index]
            
            try:
                # Preference 평가
                preference_result = self.evaluate_preference(story_instruction, story)
                
                # Recognition 평가
                recognition_result = self.evaluate_recognition(story_instruction, story)
                
                # 결과 저장
                evaluation = {
                    "story_index": index,
                    "preference_evaluation": preference_result,
                    "recognition_evaluation": recognition_result
                }
                
                evaluations.append(evaluation)
                
                # 통계 수집
                preference_scores.append(preference_result["likert_score"])
                recognition_yes_probs.append(recognition_result["yes_probability"])
                recognition_no_probs.append(recognition_result["no_probability"])
                
            except Exception as e:
                print(f"Error evaluating story {index}: {e}")
                continue
        
        # 통계 계산
        summary_stats = {
            "preference": {
                "mean_likert_score": round(np.mean(preference_scores), 4),
                "std_likert_score": round(np.std(preference_scores), 4)
            },
            "recognition": {
                "mean_yes_probability": round(np.mean(recognition_yes_probs), 4),
                "mean_no_probability": round(np.mean(recognition_no_probs), 4)
            }
        }
        
        # 최종 결과 구성
        result = {
            "evaluator_model": self.evaluator_model,
            "target_model": model_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "total_evaluations": len(evaluations),
            "evaluations": evaluations,
            "summary_statistics": summary_stats
        }
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"Story_Individual_Evaluation_{model_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Mean Likert Score: {summary_stats['preference']['mean_likert_score']}")
        print(f"Mean Yes Probability: {summary_stats['recognition']['mean_yes_probability']}")
        
        return output_file

def main():
    # 평가자 초기화
    evaluator = StoryIndividualEvaluator()
    
    # 평가할 모델들과 파일 경로
    model_files = {
        "Deepseek": "Data/Deepseek-stories.json",
        "Gemini": "Data/Gemini-stories.json", 
        "Gemma": "Data/Gemma-stories.json",
        "Llama": "Data/LLama-stories.json",
        "Qwen": "Data/Qwen-stories.jsonl"
    }
    
    # 각 모델에 대해 평가 수행
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            evaluator.evaluate_model_stories(file_path, model_name)
        else:
            print(f"Warning: File {file_path} not found for model {model_name}")
    
    print("All story evaluations completed!")

if __name__ == "__main__":
    main()
