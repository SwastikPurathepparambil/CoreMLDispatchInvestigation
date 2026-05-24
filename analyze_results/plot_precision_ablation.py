from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_model_name(model_name: str) -> str:
    return (
        model_name
        .replace("_fp16", "")
        .replace("_fp32", "")
    )


def extract_precision(model_name: str) -> str:
    if model_name.endswith("_fp16"):
        return "FP16"
    if model_name.endswith("_fp32"):
        return "FP32"
    return "unknown"


def clean_compute_units(name: str) -> str:
    mapping = {
        "all": "All",
        "cpuOnly": "CPU",
        "cpuAndGPU": "CPU+GPU",
        "cpuAndNeuralEngine": "CPU+ANE",
    }
    return mapping.get(name, name)


def main():
    project_root = Path(__file__).resolve().parents[1]

    fp16_csv = project_root / "results" / "full_baseline_fp16.csv"
    fp32_csv = project_root / "results" / "full_baseline_fp32.csv"

    output_dir = project_root / "plots"
    output_dir.mkdir(exist_ok=True)

    output_png = output_dir / "precision_ablation_median_latency.png"
    output_pdf = output_dir / "precision_ablation_median_latency.pdf"
    summary_csv = project_root / "results" / "precision_ablation_summary.csv"

    df16 = pd.read_csv(fp16_csv)
    df32 = pd.read_csv(fp32_csv)

    df = pd.concat([df16, df32], ignore_index=True)

    df["model_clean"] = df["model"].apply(clean_model_name)
    df["precision"] = df["model"].apply(extract_precision)
    df["compute_clean"] = df["compute_units"].apply(clean_compute_units)

    df["model_precision_backend"] = (
        df["model_clean"] + " " + df["precision"] + " " + df["compute_clean"]
    )

    model_order = ["MobileNetV2", "ResNet50", "SimpleMLP", "DistilBERT", "TinyGPT"]
    precision_order = ["FP16", "FP32"]
    compute_order = ["All", "CPU", "CPU+GPU", "CPU+ANE"]

    df["model_clean"] = pd.Categorical(df["model_clean"], model_order, ordered=True)
    df["precision"] = pd.Categorical(df["precision"], precision_order, ordered=True)
    df["compute_clean"] = pd.Categorical(df["compute_clean"], compute_order, ordered=True)

    df = df.sort_values(["model_clean", "precision", "compute_clean"])

    summary = df[
        ["model_clean", "precision", "compute_clean", "mean_ms", "median_ms"]
    ].copy()

    summary.to_csv(summary_csv, index=False)

    # Plot only .all and fastest manual path per model/precision to keep chart readable.
    fastest_manual = (
        df[df["compute_clean"] != "All"]
        .sort_values("median_ms")
        .groupby(["model_clean", "precision"], observed=False)
        .first()
        .reset_index()
    )
    fastest_manual["compute_clean"] = "Best Manual"

    automatic = df[df["compute_clean"] == "All"].copy()

    plot_df = pd.concat([automatic, fastest_manual], ignore_index=True)

    plot_df["label"] = (
        plot_df["model_clean"].astype(str)
        + " "
        + plot_df["precision"].astype(str)
        + " "
        + plot_df["compute_clean"].astype(str)
    )

    plot_df = plot_df.sort_values(["model_clean", "precision", "compute_clean"])

    ax = plot_df.plot(
        kind="bar",
        x="label",
        y="median_ms",
        figsize=(14, 7),
        legend=False,
    )

    ax.set_title("Precision Ablation: Automatic vs Best Manual Compute Setting")
    ax.set_xlabel("Model / precision / setting")
    ax.set_ylabel("Median latency (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)

    print(f"Saved plot to: {output_png}")
    print(f"Saved plot to: {output_pdf}")
    print(f"Saved summary CSV to: {summary_csv}")

    print("\nPrecision ablation summary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()