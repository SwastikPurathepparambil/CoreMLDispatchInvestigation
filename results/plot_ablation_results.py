import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/ablation_results.csv")

# Clean up ordering for consistent plots
compute_order = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"]
df["compute_units"] = pd.Categorical(df["compute_units"], categories=compute_order, ordered=True)

# Plot each model (backend comparison)

models = df["model"].unique()

for model in models:
    subset = df[df["model"] == model].sort_values("compute_units")

    plt.figure()
    plt.bar(subset["compute_units"], subset["mean_ms"])

    plt.title(f"{model} - Mean Latency by Compute Unit")
    plt.xlabel("Compute Unit")
    plt.ylabel("Mean Latency (ms)")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

# 2. Group models by architecture type

def get_family(name):
    if "MobileNet" in name or "ResNet" in name:
        return "CNN"
    elif "MLP" in name:
        return "MLP"
    elif "DistilBERT" in name or "TinyGPT" in name:
        return "Transformer"
    return "Other"

df["family"] = df["model"].apply(get_family)

# 3. Plot: CNNs only

cnn = df[df["family"] == "CNN"]

plt.figure()
for cu in compute_order:
    subset = cnn[cnn["compute_units"] == cu]
    if len(subset):
        plt.plot(subset["model"], subset["mean_ms"], marker="o", label=cu)

plt.title("CNN Models - Latency Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Latency (ms)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Plot: Transformers only

tfm = df[df["family"] == "Transformer"]

plt.figure()
for cu in compute_order:
    subset = tfm[tfm["compute_units"] == cu]
    if len(subset):
        plt.plot(subset["model"], subset["mean_ms"], marker="o", label=cu)

plt.title("Transformer Models - Latency Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Latency (ms)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()

# 5. Plot: MLP only

mlp = df[df["family"] == "MLP"]

plt.figure()
for cu in compute_order:
    subset = mlp[mlp["compute_units"] == cu]
    if len(subset):
        plt.plot(subset["model"], subset["mean_ms"], marker="o", label=cu)

plt.title("MLP Model - Latency Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Latency (ms)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()


# 6. Global comparison (compute unit averages)

avg = df.groupby("compute_units")["mean_ms"].mean().reindex(compute_order)

plt.figure()
plt.bar(avg.index.astype(str), avg.values)

plt.title("Average Latency Across All Models")
plt.xlabel("Compute Unit")
plt.ylabel("Mean Latency (ms)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
