import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Raw data 입력
detection_data = {
    "Deepseek": {"Gemini": 0.4978, "Gemma": 0.5037, "Llama": 0.4922, "Qwen": 0.5072},
    "Gemini": {"Deepseek": 0.5175, "Gemma": 0.5543, "Llama": 0.4826, "Qwen": 0.5817},
    "Gemma": {"Deepseek": 0.4938, "Gemini": 0.4968, "Llama": 0.4902, "Qwen": 0.5083},
    "Llama": {"Deepseek": 0.5163, "Gemini": 0.5571, "Gemma": 0.5797, "Qwen": 0.5852},
    "Qwen": {"Deepseek": 0.4929, "Gemini": 0.5008, "Gemma": 0.5073, "Llama": 0.4739},
}

comparison_data = {
    "Deepseek": {"Gemini": 0.4972, "Gemma": 0.4996, "Llama": 0.4970, "Qwen": 0.5003},
    "Gemini": {"Deepseek": 0.4761, "Gemma": 0.5748, "Llama": 0.3942, "Qwen": 0.5875},
    "Gemma": {"Deepseek": 0.4966, "Gemini": 0.4967, "Llama": 0.4897, "Qwen": 0.5063},
    "Llama": {"Deepseek": 0.5437, "Gemini": 0.6614, "Gemma": 0.6927, "Qwen": 0.6771},
    "Qwen": {"Deepseek": 0.4941, "Gemini": 0.5002, "Gemma": 0.5079, "Llama": 0.4721},
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

df_preference = dict_to_df(comparison_data, "Self-Preference (pairwise)")
df_recognition = dict_to_df(detection_data, "Self-Recognition (pairwise)")
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
    col="Setting",
    hue_order=models,  # legend와 bar 순서 고정
    palette=palette,
    height=4,
    aspect=1.3
)

g.set_axis_labels("Evaluator", "Score")
g.set_titles("{col_name}")
g.set(ylim=(0.3, 0.75))
g._legend.set_bbox_to_anchor((0.99, 0.85))
g._legend.set_title("Compared To")

plt.tight_layout()
plt.show()
