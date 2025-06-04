import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class StoryGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        """
        Initialize the story generator with Llama 3.1 8B Instruct model.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto" if self.device == "cuda" else None
        )
        
        print("Model loaded successfully!")
        
    def create_story_prompt(self, instruction):
        """
        Create the prompt for story generation task.
        
        Args:
            instruction (str): The writing prompt/instruction
            
        Returns:
            str: Formatted prompt for the model
        """
        messages = [
            {
                "role": "user",
                "content": f"Given a writing prompt, craft a complete short story with clear beginning, middle, and end. Include descriptive details, dialogue, and emotional depth. Keep the story between return a five-sentence summary.\n\nStory Prompt:\n{instruction}\n\nWrite an engaging short story about 300 words with no other text."
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_story(self, instruction):
        """
        Generate a story for the given instruction.
        
        Args:
            instruction (str): The writing prompt
            
        Returns:
            str: The generated story
        """
        prompt = self.create_story_prompt(instruction)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        # Generate with temperature=0 (greedy decoding)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def process_story_dataset(self, num_samples=2000, output_file="stories.json"):
        """
        Process story generation dataset to generate stories.
        
        Args:
            num_samples (int): Number of samples to process
            output_file (str): Output JSON file name
            
        Returns:
            str: Path to the saved JSON file
        """
        print(f"Loading story generation dataset...")
        dataset = load_dataset("krisha05/story-generation-dataset", split="train")
        
        # Take first num_samples
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        results = {}
        
        print(f"Generating stories for {len(dataset)} prompts...")
        for i, example in enumerate(tqdm(dataset)):
            instruction = example['instruction']
            
            try:
                story = self.generate_story(instruction)
                results[str(i)] = story
            except Exception as e:
                print(f"Error processing instruction {i}: {e}")
                results[str(i)] = ""
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Stories saved to {output_file}")
        return output_file

def main():
    # Initialize generator
    generator = StoryGenerator()
    
    # Generate stories
    output_file = generator.process_story_dataset(num_samples=2000, output_file="stories.json")
    
    print(f"Story generation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
