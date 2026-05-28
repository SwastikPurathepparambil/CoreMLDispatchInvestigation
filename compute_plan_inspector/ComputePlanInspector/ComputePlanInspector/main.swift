import Foundation
import CoreML

let modelNames = [
    "MobileNetV2_fp32",
    "ResNet50_fp32",
    "SimpleMLP_fp32",
    "DistilBERT_fp32",
    "TinyGPT_fp32"
]


func loadModel(modelName: String, computeUnits: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits

    guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
        throw NSError(
            domain: "CoreMLBenchmark",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Missing model \(modelName)"]
        )
    }

    return try MLModel(contentsOf: modelURL, configuration: config)
}


func makeInput(modelName: String) throws -> MLFeatureProvider {
    let shape: [NSNumber]
    let inputName: String
    let dataType: MLMultiArrayDataType

    if modelName.contains("SimpleMLP") {
        shape = [1, 1024]
        inputName = "input"
        dataType = .float32

    } else if modelName.contains("DistilBERT") {
        shape = [1, 128]
        inputName = "input_ids"
        dataType = .int32

    } else if modelName.contains("TinyGPT") {
        shape = [1, 64]
        inputName = "input_ids"
        dataType = .int32

    } else {
        shape = [1, 3, 224, 224]
        inputName = "input"
        dataType = .float32
    }

    let inputArray = try MLMultiArray(shape: shape, dataType: dataType)

    for i in 0..<inputArray.count {
        if dataType == .int32 {
            inputArray[i] = NSNumber(value: Int32(i % 100))
        } else {
            inputArray[i] = NSNumber(value: Float(0.5))
        }
    }

    return try MLDictionaryFeatureProvider(dictionary: [
        inputName: inputArray
    ])
}

func benchmark(modelName: String, computeUnits: MLComputeUnits, label: String) throws {

    let model = try loadModel(modelName: modelName, computeUnits: computeUnits)
    let input = try makeInput(modelName: modelName)

    // Warmup PER CONFIG
    for _ in 0..<10 {
        _ = try model.prediction(from: input)
    }

    var times: [Double] = []

    for _ in 0..<50 {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try model.prediction(from: input)
        let end = CFAbsoluteTimeGetCurrent()

        times.append((end - start) * 1000.0)
    }

    let sorted = times.sorted()
    let median = sorted[sorted.count / 2]
    let mean = times.reduce(0, +) / Double(times.count)

    print("\(modelName),\(label),\(mean),\(median)")
}

print("model,compute_units,mean_ms,median_ms")

for modelName in modelNames {
    try benchmark(modelName: modelName, computeUnits: .all, label: "all")
    try benchmark(modelName: modelName, computeUnits: .cpuOnly, label: "cpuOnly")
    try benchmark(modelName: modelName, computeUnits: .cpuAndGPU, label: "cpuAndGPU")
    try benchmark(modelName: modelName, computeUnits: .cpuAndNeuralEngine, label: "cpuAndNeuralEngine")
}
