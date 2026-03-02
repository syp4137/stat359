import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# I/O
# -----------------------
def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    needed = ["prompt_id", "temperature", "sample_id", "generated_text"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"JSONL missing columns: {missing}")

    df["temperature"] = df["temperature"].astype(float)
    df["prompt_id"] = df["prompt_id"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["generated_text"] = df["generated_text"].astype(str)
    return df


# -----------------------
# Metrics helpers
# -----------------------
def distinct_2(text: str) -> float:
    toks = [t for t in text.strip().split() if t]
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def mean_pairwise_cosine(mat: np.ndarray) -> float:
    k = mat.shape[0]
    if k < 2:
        return np.nan
    sims = cosine_similarity(mat)
    upper = sims[np.triu_indices(k, k=1)]
    return float(upper.mean())


def split_sentences(text: str):
    """
    Lightweight sentence splitter (no extra deps).
    Good enough for TinyStories outputs.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on punctuation + space
    sents = re.split(r"(?<=[.!?])\s+", text)
    # Filter very short fragments
    sents = [s.strip() for s in sents if len(s.strip().split()) >= 3]
    return sents


def story_coherence_tfidf(text: str, sent_tfidf: TfidfVectorizer) -> float:
    """
    Sentence-level coherence proxy:
    - Split story into sentences
    - TF-IDF embed each sentence
    - Return mean pairwise cosine similarity among sentences
    """
    sents = split_sentences(text)
    if len(sents) < 2:
        return np.nan
    X = sent_tfidf.transform(sents).toarray()
    return mean_pairwise_cosine(X)


# -----------------------
# Group computation
# -----------------------
def compute_metrics_for_group(texts, story_tfidf: TfidfVectorizer, sent_tfidf: TfidfVectorizer) -> dict:
    # (1) Story-level diversity across samples (within same prompt/temp)
    X_story = story_tfidf.transform(texts).toarray()
    mean_cos = mean_pairwise_cosine(X_story)
    story_div = 1.0 - mean_cos if not np.isnan(mean_cos) else np.nan

    # (2) Lexical diversity: Distinct-2 across samples
    d2 = float(np.mean([distinct_2(t) for t in texts]))

    # (3) Coherence: average sentence-level coherence across samples
    coherences = [story_coherence_tfidf(t, sent_tfidf) for t in texts]
    coh = float(np.nanmean(coherences)) if np.any(~np.isnan(coherences)) else np.nan

    avg_words = float(np.mean([len(t.split()) for t in texts])) if texts else np.nan
    avg_sents = float(np.mean([len(split_sentences(t)) for t in texts])) if texts else np.nan

    return {
        "mean_pairwise_cosine": mean_cos,
        "story_cosine_diversity": story_div,
        "distinct_2": d2,
        "coherence_sent_tfidf": coh,
        "avg_words": avg_words,
        "avg_sentences": avg_sents,
        "n_samples": len(texts),
    }


# -----------------------
# Plotting
# -----------------------
def plot_temp_curves(agg: pd.DataFrame, outpath: Path):
    temps = agg["temperature"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Diversity
    axes[0].errorbar(
        temps,
        agg["story_div_mean"],
        yerr=agg["story_div_std"],
        fmt="o-",
        capsize=3,
    )
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Story diversity (1 - mean cosine)")
    axes[0].set_title("Temperature vs Story Diversity")

    # Distinct-2
    axes[1].errorbar(
        temps,
        agg["distinct2_mean"],
        yerr=agg["distinct2_std"],
        fmt="o-",
        capsize=3,
    )
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Distinct-2 (avg)")
    axes[1].set_title("Temperature vs Lexical Diversity")

    # Coherence
    axes[2].errorbar(
        temps,
        agg["coherence_mean"],
        yerr=agg["coherence_std"],
        fmt="o-",
        capsize=3,
    )
    axes[2].set_xlabel("Temperature")
    axes[2].set_ylabel("Sentence-level coherence (mean cosine)")
    axes[2].set_title("Temperature vs Coherence")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(metrics: pd.DataFrame, outpath: Path):
    """
    Prompt-level tradeoff plot:
    - x: story diversity
    - y: coherence
    color: temperature
    """
    fig = plt.figure(figsize=(6, 5))
    temps = sorted(metrics["temperature"].unique())
    for t in temps:
        sub = metrics[metrics["temperature"] == t]
        plt.scatter(
            sub["story_cosine_diversity"],
            sub["coherence_sent_tfidf"],
            label=f"T={t}",
            alpha=0.75,
        )

    plt.xlabel("Story diversity (1 - mean cosine)")
    plt.ylabel("Sentence-level coherence (mean cosine)")
    plt.title("Diversity–Coherence Tradeoff (prompt-level)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/diversity_with_coherence")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--sent_max_features", type=int, default=30000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(args.jsonl)

    # TF-IDF for story-level vectors (fit on full stories)
    story_tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=args.min_df,
        max_features=args.max_features,
        token_pattern=r"(?u)\b\w+\b",
    )
    story_tfidf.fit(df["generated_text"].tolist())

    # TF-IDF for sentence-level vectors (fit on all sentences across corpus)
    all_sents = []
    for txt in df["generated_text"].tolist():
        all_sents.extend(split_sentences(txt))
    if len(all_sents) < 10:
        raise ValueError("Too few sentences extracted. Check sentence splitter or data.")
    sent_tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=args.min_df,
        max_features=args.sent_max_features,
        token_pattern=r"(?u)\b\w+\b",
    )
    sent_tfidf.fit(all_sents)

    # Compute metrics per (prompt_id, temperature)
    rows = []
    for (pid, temp), g in df.groupby(["prompt_id", "temperature"], sort=True):
        texts = g.sort_values("sample_id")["generated_text"].tolist()
        m = compute_metrics_for_group(texts, story_tfidf, sent_tfidf)
        m.update({"prompt_id": int(pid), "temperature": float(temp)})
        rows.append(m)

    metrics = pd.DataFrame(rows).sort_values(["temperature", "prompt_id"])
    metrics.to_csv(outdir / "metrics_by_prompt_temp.csv", index=False)

    # Aggregate per temperature
    agg = metrics.groupby("temperature").agg(
        prompts=("prompt_id", "nunique"),
        story_div_mean=("story_cosine_diversity", "mean"),
        story_div_std=("story_cosine_diversity", "std"),
        distinct2_mean=("distinct_2", "mean"),
        distinct2_std=("distinct_2", "std"),
        coherence_mean=("coherence_sent_tfidf", "mean"),
        coherence_std=("coherence_sent_tfidf", "std"),
        avg_words_mean=("avg_words", "mean"),
        avg_words_std=("avg_words", "std"),
        avg_sents_mean=("avg_sentences", "mean"),
        avg_sents_std=("avg_sentences", "std"),
    ).reset_index()
    agg.to_csv(outdir / "metrics_by_temperature.csv", index=False)

    # Plots
    plot_temp_curves(agg, outdir / "temp_vs_diversity_distinct2_coherence.png")
    plot_tradeoff(metrics, outdir / "tradeoff_diversity_vs_coherence.png")

    # Quick correlations (prompt-level)
    corr = metrics[["story_cosine_diversity", "distinct_2", "coherence_sent_tfidf"]].corr()
    corr.to_csv(outdir / "correlation_matrix.csv")

    print("Saved:")
    print(f"- {outdir/'metrics_by_prompt_temp.csv'}")
    print(f"- {outdir/'metrics_by_temperature.csv'}")
    print(f"- {outdir/'temp_vs_diversity_distinct2_coherence.png'}")
    print(f"- {outdir/'tradeoff_diversity_vs_coherence.png'}")
    print(f"- {outdir/'correlation_matrix.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()