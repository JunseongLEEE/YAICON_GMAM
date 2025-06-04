import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class SummaryGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        """
        Initialize the summary generator with Llama 3.1 8B Instruct model.
        
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
        
    def create_summary_prompt(self, article):
        """
        Create the prompt for summarization task.
        
        Args:
            article (str): The news article to summarize
            
        Returns:
            str: Formatted prompt for the model
        """
        messages = [
            {
                "role": "system",
                "content": "You are a news-article summarizer. Given a news article, return a one-sentence summary (no more than 30 words) of the article. This will really help us better understand the article. Return only the one-sentence summary with no other text."
            },
            {
                "role": "user", 
                "content": f"Article:\n{article}\nProvide a one-sentence summary (no more than 30 words) with no other text."
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_summary(self, article):
        """
        Generate a summary for the given article.
        
        Args:
            article (str): The article to summarize
            
        Returns:
            str: The generated summary
        """
        prompt = self.create_summary_prompt(article)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        # Generate with temperature=0 (greedy decoding)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
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
    
    def process_xsum_dataset(self, num_samples=2000, output_file="summaries.json"):
        """
        Process XSUM dataset to generate summaries.
        
        Args:
            num_samples (int): Number of samples to process
            output_file (str): Output JSON file name
            
        Returns:
            str: Path to the saved JSON file
        """
        print(f"Loading XSUM dataset...")
        dataset = load_dataset("EdinburghNLP/xsum", split="train")
        
        # Take first num_samples
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        results = {}
        
        print(f"Generating summaries for {len(dataset)} articles...")
        for i, example in enumerate(tqdm(dataset)):
            article_id = example['id']
            document = example['document']
            
            try:
                summary = self.generate_summary(document)
                results[article_id] = summary
            except Exception as e:
                print(f"Error processing article {article_id}: {e}")
                results[article_id] = ""
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Summaries saved to {output_file}")
        return output_file

def main():
    # Initialize generator
    generator = SummaryGenerator()
    
    # Generate summaries
    output_file = generator.process_xsum_dataset(num_samples=2000, output_file="summaries.json")
    
    print(f"Summary generation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
