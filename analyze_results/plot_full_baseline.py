from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_model_name(model_name: str) -> str:
    return model_name.replace("_fp16", "")


def clean_compute_units(name: str) -> str:
    mapping = {
        "all": "All / Automatic",
        "cpuOnly": "CPU only",
        "cpuAndGPU": "CPU + GPU",
        "cpuAndNeuralEngine": "CPU + Neural Engine",
    }
    return mapping.get(name, name)


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_csv = project_root / "results" / "full_baseline_fp16.csv"
    output_dir = project_root / "plots"
    output_dir.mkdir(exist_ok=True)

    output_png = output_dir / "full_baseline_fp16_median_latency.png"
    output_pdf = output_dir / "full_baseline_fp16_median_latency.pdf"
    summary_csv = output_dir / "full_baseline_fp16_summary.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Could not find input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = {"model", "compute_units", "mean_ms", "median_ms"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["model_clean"] = df["model"].apply(clean_model_name)
    df["compute_units_clean"] = df["compute_units"].apply(clean_compute_units)

    model_order = [
        "MobileNetV2",
        "ResNet50",
        "SimpleMLP",
        "DistilBERT",
        "TinyGPT",
    ]

    compute_order = [
        "All / Automatic",
        "CPU only",
        "CPU + GPU",
        "CPU + Neural Engine",
    ]

    df["model_clean"] = pd.Categorical(
        df["model_clean"],
        categories=model_order,
        ordered=True,
    )

    df["compute_units_clean"] = pd.Categorical(
        df["compute_units_clean"],
        categories=compute_order,
        ordered=True,
    )

    df = df.sort_values(["model_clean", "compute_units_clean"])

    pivot = df.pivot(
        index="model_clean",
        columns="compute_units_clean",
        values="median_ms",
    )

    pivot.to_csv(summary_csv)

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )

    ax.set_title("Core ML FP16 Median Latency by Model and Compute Setting")
    ax.set_xlabel("Model")
    ax.set_ylabel("Median latency (ms)")
    ax.legend(title="Compute setting")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)

    print(f"Saved plot to: {output_png}")
    print(f"Saved plot to: {output_pdf}")
    print(f"Saved summary table to: {summary_csv}")

    print("\nSummary table:")
    print(pivot.round(4))


if __name__ == "__main__":
    main()