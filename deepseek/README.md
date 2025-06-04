# Arguments
* `--task`: Model tasks

    * `summary`: Generate summary
    * `story`: Generate story
    * `likert_summary`: Calculate likert score for summaries
    * `likert_story`: Calculate likert score for stories
    * `recognition_summary`: Calculate mean probabilities for "Yes" / "No" for the given summaries
    * `recognition_story`: Calculate mean probabilities for "Yes" / "No" for the given stories
    * `pairwise_summary`: Calculate confidence score of self-detection for given summaries 
    * `pairwise_story`: Calculate confidence score of self-detection for given stories
    * `pairwise_comparison_summary`: Calculate confidence score of self-preference for given summaries
    * `pairwise_comparison_story`: Calculate confidence score of self-preference for given stories
* `--target_model`: Name of the target model for results grouping
* `--num_samples`: Number of samples
* `--max_new_tokens`: Max tokens to generate
* `--prompt_file`: Path to prompt file
* `--generated_file`: Path(s) to JSON file(s) containing generated outputs to score
* `--output_path`: Path to save outputs


<h1>Tasks</h1>
<h3>Generation</h3>

- Run `python deepseek_generate.py`, setting the `--task` argument to either "summary" or "story" depening on your desired task 

Ex.

```
python deepseek_generate.py --task summary --target_model Deepseek --num_samples 2000 --prompt_file prompts/generate_summary.txt --output_path results/summaires/xsum_deepseek_summaries.json
```
***

<h3>Evaluation</h3>
<h4>Individual-preference</h4>

- Run `python deepseek_inference.py`, setting the `--task` argument to either "likert_summary" or "likert_story" depending on your desired task

Ex.

```
python deepseek_inference.py --task likert-summary --target_model Deepseek --num_samples 2000 --prompt_file prompts/individual_score_summary.txt --generated_file results/summaries/xsum_deepseek_summaries.json --output_path results/individual/preference/individual_score_summary.json
```
<h4>Individual-recognition</h4>

- Run `python deepseek_inference.py`, setting the `--task` argument to either "recognition_summary" or "recognition_story" depending on your desired task

Ex.
```
python deepseek_inference.py --task recognition-summary --target_model Deepseek --num_samples 2000 --prompt_file prompts/individual_recognition_summary.txt --generated_file results/summaries/xsum_deepseek_summaries.json --output_path results/individual/recognition/individual_recognition_summary.json
```
<h4>Pairwise-comparision</h4>

- Run `python deepseek_inference.py`, setting the `--task` argument to either "pairwise_comparison_summary" or "pairwise_comparison_story" depending on your desired task

Ex.
```
python deepseek_inference.py --task pairwise_comparison_summary --target_model Llama --num_samples 2000 --prompt_file prompts/pairwise_comparision_summary.txt --generated_file results/summaries/xsum_deepseek_summaries.json results/summaries/LLama-summaries.json --output_path results/pairwise/comparision/llama_pairwise_comparison_summary.json
```
<h4>Pairwise-detection</h4>

- Run `python deepseek_inference.py`, setting the `--task` argument to either "pairwise_summary" or "pairwise_story" depending on your desired task

Ex.
```
python deepseek_inference.py --task pairwise_summary --target_model Llama --num_samples 2000 --prompt_file prompts/pairwise_detection_summary.txt --generated_file results/summaries/xsum_deepseek_summaries.json results/summaries/LLama-summaries.json --output_path results/pairwise/detection/llama_pairwise_detection_summary.json
```

# Plot
To plot the individual-preference and recognition results, run `plots/individual_eval.py`.


To plot the pairwise-comparison and detection results, run `plots/pairwise_eval.py`.
