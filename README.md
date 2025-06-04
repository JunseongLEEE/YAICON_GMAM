
# Can LLMs Distinguish Themselves? 🤖

**An empirical study on self-recognition and self-preference bias in large language models**

📄 *Based on research presented at Yaicon 2025*

## 📋 Overview

This repository investigates whether Large Language Models (LLMs) can recognize their own generated text and whether this capability leads to self-preference bias in evaluation tasks. We examine both summary generation and story generation across multiple state-of-the-art models.

## 🔍 Research Questions

- **Self-Recognition**: Can LLMs distinguish between text they generated vs. text from humans or other LLMs?
- **Self-Preference**: Do LLMs favor their own outputs when evaluating quality?
- **Correlation**: Is there a relationship between self-recognition ability and self-preference bias?

## 🎯 Key Findings

### Summary Generation
- **Self-preference exists**: LLMs show measurable bias toward their own outputs
- **Self-recognition capability**: Models demonstrate above-chance ability to identify their own text
- **Linear correlation**: Strong positive correlation between self-recognition and self-preference scores

### Story Generation
- **Inconsistent patterns**: Results vary significantly across models and evaluation methods
- **Gemini anomaly**: Shows extremely high self-preference in pairwise comparison but low individual scores
- **Model-dependent behavior**: Different architectures exhibit varying degrees of self-bias

## 🛠️ Methodology

### Models Tested
- **Qwen2.5-1.5B-Instruct**
- **Gemma-2-9b-instruct** 
- **Deepseek-llm-7b-chat**
- **Llama-3.1-8B-Instruct**
- **Gemini 2.0 Flash-Lite**

### Evaluation Approaches

#### Individual Assessment
- **Recognition**: Yes/No identification of self-generated text
- **Preference**: 1-5 Likert scale quality rating
- **Scoring**: Weighted average using token probabilities

#### Pairwise Comparison
- **Recognition**: Choose which of two texts was self-generated
- **Preference**: Select higher quality text between two options
- **Bias mitigation**: Order swapping to reduce position effects

### Tasks
1. **Summary Generation**: News article summarization using XSUM prompts
2. **Story Generation**: Creative writing based on instruction prompts

## 📊 Results Summary

### Self-Preference (Individual)
- Most models show scores around 0.50 (neutral)
- Notable exceptions in specific model-task combinations
- Story generation shows more variation than summarization

### Self-Recognition (Individual) 
- Models demonstrate above-chance recognition ability (~0.50-0.65)
- Consistent across different model architectures
- Higher confidence in story generation for some models

### Pairwise Results
- More pronounced self-preference bias in pairwise settings
- Gemini shows extreme behavior (0.739 preference score)
- Position effects successfully mitigated through order swapping

## 🔬 Technical Implementation

### Confidence Scoring
```python
# Likert Score Calculation
likert_score = sum(i * P(i) for i in range(1, 6))

# Recognition Confidence  
confidence = P(Yes) for self-recognition tasks
```

### Normalization
Scores normalized relative to comparison models:
```
normalized_score = model_score / (model_score + comparison_score)
```

### API Integration
- **Hugging Face models**: Direct logit access for probability calculation
- **API models (Gemini)**: Response log probability extraction

## 📈 Data Analysis

Results include comprehensive analysis across:
- Individual vs. pairwise evaluation methods
- Recognition vs. preference tasks
- Summary vs. story generation domains
- Multiple model architectures and sizes

## 🔄 Reproducibility

All experiments designed for reproducibility with:
- Standardized prompting templates
- Consistent evaluation metrics  
- Order randomization for bias reduction
- Multiple model architecture coverage

## 📚 Related Work

This research builds on findings about LLM evaluation biases, particularly:
- Anchoring effects in LLM scoring (Pearson r = 0.979 vs human r = 0.315)
- Improper use of scoring scales (overuse of 90, 95 in 1-100 scales)
- Inconsistency in automated evaluation

## 🚀 Usage

```bash
# Clone repository
git clone https://github.com/username/llm-self-recognition.git

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python evaluate_models.py --task summary --model gemini --method pairwise
```

## 📝 Citation

```bibtex
@inproceedings{yaicon2025_llm_self_recognition,
  title={Can LLMs Distinguish Themselves?},
  author={Yaicon},
  booktitle={Yaicon 2025},
  year={2025},
  month={May}
}
```

## ⚠️ Limitations

- Limited to specific model architectures and sizes
- Focused on English language tasks only
- Summary and story generation domains
- Future work needed on causal mechanisms

## 🔮 Future Work

- Expand to more diverse tasks and languages
- Investigate causal relationships through intervention studies
- Examine fine-tuning effects on self-recognition
- Explore mitigation strategies for self-preference bias

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

*"An LLM that is not finetuned cannot identify itself... or can it?"* 🦙🔍
