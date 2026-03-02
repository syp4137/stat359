import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    needed = ["prompt_id", "temperature", "sample_id", "generated_text"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"JSONL is missing columns: {missing}")
    df["temperature"] = df["temperature"].astype(float)
    df["prompt_id"] = df["prompt_id"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["generated_text"] = df["generated_text"].astype(str)
    return df


def distinct_2(text: str) -> float:
    toks = [t for t in text.strip().split() if t]
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def mean_pairwise_cosine(mat) -> float:
    """
    mat: array-like (k, d)
    Returns mean cosine similarity over all pairs i<j.
    """
    k = mat.shape[0]
    if k < 2:
        return np.nan
    sims = cosine_similarity(mat)
    upper = sims[np.triu_indices(k, k=1)]
    return float(upper.mean())


def compute_metrics_for_group(texts, tfidf: TfidfVectorizer) -> dict:
    # Story-level cosine similarity using TF-IDF vectors (lexical semantic-ish baseline)
    X = tfidf.transform(texts)
    mean_cos = mean_pairwise_cosine(X.toarray())
    cosine_div = 1.0 - mean_cos if not np.isnan(mean_cos) else np.nan

    # Distinct-2 averaged over samples
    d2 = float(np.mean([distinct_2(t) for t in texts]))

    return {
        "mean_pairwise_cosine": mean_cos,
        "story_cosine_diversity": cosine_div,  # 1 - mean cosine
        "distinct_2": d2,
        "n_samples": len(texts),
        "avg_words": float(np.mean([len(t.split()) for t in texts])) if texts else np.nan,
    }


def plot_temp_curves(agg: pd.DataFrame, outpath: Path):
    # Two-panel plot: diversity & distinct-2 across temperatures
    temps = agg["temperature"].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: story cosine diversity
    axes[0].errorbar(
        temps,
        agg["story_cosine_div_mean"],
        yerr=agg["story_cosine_div_std"],
        fmt="o-",
        capsize=3,
    )
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Story cosine diversity (1 - mean cosine)")
    axes[0].set_title("Temperature vs Story Diversity (TF-IDF cosine)")

    # Panel 2: distinct-2
    axes[1].errorbar(
        temps,
        agg["distinct2_mean"],
        yerr=agg["distinct2_std"],
        fmt="o-",
        capsize=3,
    )
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Distinct-2 (avg)")
    axes[1].set_title("Temperature vs Lexical Diversity (Distinct-2)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_relationship(metrics: pd.DataFrame, outpath: Path):
    # Scatter: prompt-level points, colored by temperature
    fig = plt.figure(figsize=(6, 5))
    temps = sorted(metrics["temperature"].unique())
    for t in temps:
        sub = metrics[metrics["temperature"] == t]
        plt.scatter(sub["story_cosine_diversity"], sub["distinct_2"], label=f"T={t}", alpha=0.7)

    plt.xlabel("Story cosine diversity (1 - mean cosine)")
    plt.ylabel("Distinct-2")
    plt.title("Relationship between Diversity metrics (prompt-level)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to diversity_generations.jsonl")
    parser.add_argument("--outdir", type=str, default="outputs/diversity_2metrics")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(args.jsonl)

    # Fit TF-IDF across all generated texts once
    tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=args.min_df,
        max_features=args.max_features,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf.fit(df["generated_text"].tolist())

    # Compute metrics per (prompt_id, temperature)
    rows = []
    for (pid, temp), g in df.groupby(["prompt_id", "temperature"], sort=True):
        texts = g.sort_values("sample_id")["generated_text"].tolist()
        m = compute_metrics_for_group(texts, tfidf)
        m.update({"prompt_id": int(pid), "temperature": float(temp)})
        rows.append(m)

    metrics = pd.DataFrame(rows).sort_values(["temperature", "prompt_id"])
    metrics_path = outdir / "metrics_by_prompt_temp.csv"
    metrics.to_csv(metrics_path, index=False)

    # Aggregate across prompts (prompt-level averages)
    agg = metrics.groupby("temperature").agg(
        prompts=("prompt_id", "nunique"),
        story_cosine_div_mean=("story_cosine_diversity", "mean"),
        story_cosine_div_std=("story_cosine_diversity", "std"),
        distinct2_mean=("distinct_2", "mean"),
        distinct2_std=("distinct_2", "std"),
        avg_words_mean=("avg_words", "mean"),
        avg_words_std=("avg_words", "std"),
    ).reset_index()

    agg_path = outdir / "metrics_by_temperature.csv"
    agg.to_csv(agg_path, index=False)

    # Correlation analysis
    overall_corr = metrics[["story_cosine_diversity", "distinct_2"]].corr().iloc[0, 1]
    corr_by_temp = (
        metrics.groupby("temperature")[["story_cosine_diversity", "distinct_2"]]
        .corr()
        .iloc[0::2, -1]  # take corr between the two columns
        .reset_index()
        .rename(columns={"distinct_2": "corr_storycos_distinct2"})
    )

    corr_path = outdir / "correlations.csv"
    corr_by_temp.to_csv(corr_path, index=False)

    with open(outdir / "correlation_overall.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall Pearson corr(story_cosine_diversity, distinct_2) = {overall_corr:.4f}\n")

    # Plots
    plot_temp_curves(agg, outdir / "temp_vs_2metrics.png")
    plot_metric_relationship(metrics, outdir / "scatter_diversity_vs_distinct2.png")

    print(f"Saved:\n- {metrics_path}\n- {agg_path}\n- {corr_path}\n- {outdir/'correlation_overall.txt'}")
    print(f"- {outdir/'temp_vs_2metrics.png'}\n- {outdir/'scatter_diversity_vs_distinct2.png'}")
    print("Done.")


if __name__ == "__main__":
    main()