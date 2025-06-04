import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Raw data 입력
preference_data = {
    "Deepseek": {"Gemini": 0.4933, "Gemma": 0.4915, "Llama": 0.4980, "Qwen": 0.4946},
    "Gemini": {"Deepseek": 0.5742, "Gemma": 0.5428, "Llama": 0.5001, "Qwen": 0.5941},
    "Gemma": {"Deepseek": 0.5004, "Gemini": 0.4994, "Llama": 0.5004, "Qwen": 0.5056},
    "Llama": {"Deepseek": 0.5019, "Gemini": 0.5013, "Gemma": 0.5028, "Qwen": 0.5159},
    "Qwen": {"Deepseek": 0.4884, "Gemini": 0.4976, "Gemma": 0.5005, "Llama": 0.4887},
}

recognition_data = {
    "Deepseek": {"Gemini": 0.5037, "Gemma": 0.5057, "Llama": 0.4908, "Qwen": 0.5021},
    "Gemini": {"Deepseek": 0.6426, "Gemma": 0.6412, "Llama": 0.6425, "Qwen": 0.6048},
    "Gemma": {"Deepseek": 0.5019, "Gemini": 0.4657, "Llama": 0.5022, "Qwen": 0.4988},
    "Llama": {"Deepseek": 0.5008, "Gemini": 0.4992, "Gemma": 0.5007, "Qwen": 0.5129},
    "Qwen": {"Deepseek": 0.4509, "Gemini": 0.4661, "Gemma": 0.4817, "Llama": 0.4558},
}

def dict_to_df(data_dict, setting_name):
    rows = []
    for evaluator, comparisons in data_dict.items():
        for alt, score in comparisons.items():
            rows.append({
                "Evaluator": evaluator,
                "Alternative": alt,
                "Score": score,
                "Setting": setting_name
            })
    return pd.DataFrame(rows)

df_preference = dict_to_df(preference_data, "Self-Preference (individual)")
df_recognition = dict_to_df(recognition_data, "Self-Recognition (individual)")
df = pd.concat([df_preference, df_recognition])

models = ["Gemini", "Gemma", "Llama", "Qwen", "Deepseek"]  # 순서 변경

full_index = pd.MultiIndex.from_product([models, models, df["Setting"].unique()],
                                        names=["Evaluator", "Alternative", "Setting"])

df_full = pd.DataFrame(index=full_index).reset_index()
df = pd.merge(df_full, df, how="left", on=["Evaluator", "Alternative", "Setting"])

df.loc[df["Evaluator"] == df["Alternative"], "Score"] = np.nan

palette = {
    "Deepseek": "#9467bd",
    "Gemini": "#1f77b4",
    "Gemma": "#ff7f0e",
    "Llama": "#2ca02c",
    "Qwen": "#d62728"
}

sns.set(style="whitegrid", font_scale=0.9)
g = sns.catplot(
    data=df,
    kind="bar",
    x="Evaluator",
    y="Score",
    hue="Alternative",
    hue_order=models,
    palette=palette,
    col="Setting",
    height=4,
    aspect=1.3
)

g.set_axis_labels("Evaluator", "Score")
g.set_titles("{col_name}")
g.set(ylim=(0.4, 0.65))
g._legend.set_bbox_to_anchor((0.99, 0.85))
g._legend.set_title("Compared To")

plt.tight_layout()
plt.show()
