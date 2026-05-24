Goal 1:
We benchmarked five FP16 Core ML models across four compute-unit configurations: automatic/all, CPU-only, CPU+GPU, and CPU+Neural Engine. The model suite included two CNNs, one MLP, one transformer encoder-style model, and one tiny GPT-style generative transformer.

The CNN models showed strong benefit from ANE-enabled execution. MobileNetV2 and ResNet50 achieved their best or near-best latency under either .all or .cpuAndNeuralEngine, suggesting that Core ML’s automatic dispatch selected an ANE-heavy path for these convolutional workloads.

The non-CNN models showed a different pattern. The SimpleMLP and DistilBERT-style model were fastest under CPU-only execution, and the TinyGPT-style model showed a large gap: CPU-only was about 6.25× faster than automatic dispatch. These results suggest that automatic dispatch is effective for FP16 CNNs but may be suboptimal for small dense or transformer-style workloads.

Goal 2:

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
3. Do TinyGPT size ablation
4. Do SimpleMLP size ablation
5. Run Instruments on ResNet50 and TinyGPT
