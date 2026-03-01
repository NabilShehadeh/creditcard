#!/usr/bin/env python3
"""
Generate a black-and-white correlation heatmap image for the README.
Run from repo root: python scripts/generate_heatmap_image.py
Output: docs/heatmap.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

def generate_sample_data(n=5000):
    np.random.seed(42)
    data = {
        "Time": np.cumsum(np.random.exponential(800, n)),
        "Amount": np.random.lognormal(3, 1.4, n),
        "Class": np.random.binomial(1, 0.002, n),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    return pd.DataFrame(data)

def main():
    try:
        from data.processor import DataProcessor
        proc = DataProcessor()
        df = proc.load_data(None)
    except Exception:
        df = generate_sample_data()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    heatmap_cols = [c for c in numeric if c in ("Time", "Amount") or (c.startswith("V") and c[1:].isdigit())]
    if len(heatmap_cols) > 14:
        heatmap_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 13)]
    corr = df[heatmap_cols].corr()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color="black")
    ax.set_yticklabels(corr.index, fontsize=8, color="black")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.colorbar(im, ax=ax, label="Correlation", cmap="Greys")
    plt.title("Feature correlation (credit card fraud dataset)", fontsize=12, color="black")
    plt.tight_layout()
    out = REPO_ROOT / "docs" / "heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
