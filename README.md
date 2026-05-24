# Setup Requirements

## Requirements

This project was built and tested on:

- macOS 14.6
- Apple Silicon Mac
- Xcode with Core ML support
- Python 3.11
- `uv` for Python environment management
- Core ML Tools
- PyTorch / torchvision
- transformers
- pandas
- matplotlib

`MLComputePlan` requires macOS 14.4 or newer.

---

## Python Setup with uv

From the project root, install and pin Python 3.11:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
uv add coremltools torch torchvision numpy pandas matplotlib tqdm transformers
```

If Core ML Tools warns that the installed PyTorch version is newer than the tested version, use the tested pair:
```bash
uv remove torch torchvision
uv add "torch==2.7.0" "torchvision==0.22.0"
```

Then commit it:

```bash
git add SETUP.md
git commit -m "Add setup requirements documentation"
git push
```

## Xcode Setup

This repo does **not** track generated Core ML model packages (`*.mlpackage`) because they are large. After cloning the repo, you must regenerate the models locally and manually add them to the Xcode targets.

### 1. Generate the Core ML models

From the project root, run the Python conversion scripts.

```bash
uv run python convert_models/convert_mobilenetv2.py
uv run python convert_models/convert_resnet50.py
uv run python convert_models/convert_mlp.py
uv run python convert_models/convert_distilbert.py
uv run python convert_models/convert_tinygpt.py
uv run python convert_models/convert_mobilenetv2_fp32.py
uv run python convert_models/convert_resnet50_fp32.py
uv run python convert_models/convert_mlp_fp32.py
uv run python convert_models/convert_distilbert_fp32.py
uv run python convert_models/convert_tinygpt_fp32.py
```

After this the model should generate:
```bash
MobileNetV2_fp16.mlpackage
MobileNetV2_fp32.mlpackage
ResNet50_fp16.mlpackage
ResNet50_fp32.mlpackage
SimpleMLP_fp16.mlpackage
SimpleMLP_fp32.mlpackage
DistilBERT_fp16.mlpackage
DistilBERT_fp32.mlpackage
TinyGPT_fp16.mlpackage
TinyGPT_fp32.mlpackage
```

Then add the mlpackage files into the appropriate swift code files and run the CoreMLBenchmark/ComputePlanInspector

# Results + Analysis

## Goal 1:
We benchmarked five FP16 Core ML models across four compute-unit configurations: automatic/all, CPU-only, CPU+GPU, and CPU+Neural Engine. The model suite included two CNNs, one MLP, one transformer encoder-style model, and one tiny GPT-style generative transformer.

The CNN models showed strong benefit from ANE-enabled execution. MobileNetV2 and ResNet50 achieved their best or near-best latency under either .all or .cpuAndNeuralEngine, suggesting that Core ML’s automatic dispatch selected an ANE-heavy path for these convolutional workloads.

The non-CNN models showed a different pattern. The SimpleMLP and DistilBERT-style model were fastest under CPU-only execution, and the TinyGPT-style model showed a large gap: CPU-only was about 6.25× faster than automatic dispatch. These results suggest that automatic dispatch is effective for FP16 CNNs but may be suboptimal for small dense or transformer-style workloads.

## Goal 2:

Questions to answer:
1. CNNs: Why does .all ≈ cpuAndNeuralEngine?
2. MLP: Why is CPU-only fastest?
3. DistilBERT: Why is CPU-only slightly fastest?
4. TinyGPT: Why is .all much worse than CPU-only?

Insights:

Using MLComputePlan, we inspected the estimated device usage of each Core ML model under the automatic `.all` configuration. The two CNN models showed a consistent ANE-preferred pattern: convolution, activation, pooling, residual addition, and linear operations were all predicted to prefer the Neural Engine. This matched the latency results, where `.all` and `.cpuAndNeuralEngine` were nearly identical.

The SimpleMLP model showed the opposite pattern. Its linear and ReLU operations were CPU-preferred, matching the benchmark result where CPU-only execution was fastest.

The transformer-style models were more complex. DistilBERT was mostly ANE-preferred, but its gather operation did not list ANE support and was CPU-preferred. TinyGPT was the most striking case: despite being fastest on CPU-only, MLComputePlan showed that nearly all non-constant operations were GPU-preferred under `.all`. This provides a plausible explanation for the large performance gap observed in Goal 1 and identifies TinyGPT as a strong candidate case of suboptimal automatic dispatch.

After doing an fp32 vs fp16 ablation study, precision significantly affects dispatch behavior. In FP16, CNNs behaved like ANE-dispatched models, with .all matching .cpuAndNeuralEngine. In FP32, the same CNNs behaved more like GPU-dispatched models, with .all matching .cpuAndGPU and .cpuAndNeuralEngine becoming much slower. This supports the hypothesis that Core ML dispatch is sensitive to numerical precision, not only architecture type.

Next Steps:
1. Do TinyGPT size ablation
2. Do SimpleMLP size ablation
3. MLComputePlan on the ablation models
4. Run Instruments on ResNet50 and TinyGPT
