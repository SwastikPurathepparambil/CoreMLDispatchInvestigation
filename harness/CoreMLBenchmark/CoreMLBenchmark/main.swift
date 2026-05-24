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
            userInfo: [NSLocalizedDescriptionKey: "Could not find compiled model \(modelName).mlmodelc"]
        )
    }

    return try MLModel(contentsOf: modelURL, configuration: config)
}

func makeInput(modelName: String) throws -> MLFeatureProvider {
    let shape: [NSNumber]
    let inputName: String
    let dataType: MLMultiArrayDataType

    switch modelName {
    case "SimpleMLP_fp16", "SimpleMLP_fp32":
        shape = [1, 1024]
        inputName = "input"
        dataType = .float32

    case "DistilBERT_fp16", "DistilBERT_fp32":
        shape = [1, 128]
        inputName = "input_ids"
        dataType = .int32

    case "TinyGPT_fp16", "TinyGPT_fp32":
        shape = [1, 64]
        inputName = "input_ids"
        dataType = .int32

    default:
        shape = [1, 3, 224, 224]
        inputName = "input"
        dataType = .float32
    }

    guard let inputArray = try? MLMultiArray(shape: shape, dataType: dataType) else {
        throw NSError(
            domain: "CoreMLBenchmark",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Could not create input MLMultiArray for \(modelName)"]
        )
    }

    for i in 0..<inputArray.count {
        if dataType == .int32 {
            inputArray[i] = NSNumber(value: Int32.random(in: 0...1000))
        } else {
            inputArray[i] = NSNumber(value: Float.random(in: 0...1))
        }
    }

    return try MLDictionaryFeatureProvider(dictionary: [
        inputName: inputArray
    ])
}

func benchmark(modelName: String, computeUnits: MLComputeUnits, label: String) throws {
    let model = try loadModel(modelName: modelName, computeUnits: computeUnits)
    let input = try makeInput(modelName: modelName)

    let warmupRuns = 10
    let measuredRuns = 50

    for _ in 0..<warmupRuns {
        _ = try model.prediction(from: input)
    }

    var times: [Double] = []

    for _ in 0..<measuredRuns {
        let start = DispatchTime.now()
        _ = try model.prediction(from: input)
        let end = DispatchTime.now()

        let elapsedNs = end.uptimeNanoseconds - start.uptimeNanoseconds
        let elapsedMs = Double(elapsedNs) / 1_000_000.0
        times.append(elapsedMs)
    }

    let sorted = times.sorted()
    let median = sorted[sorted.count / 2]
    let mean = times.reduce(0, +) / Double(times.count)

    print("\(modelName),\(label),\(mean),\(median)")
}

do {
    print("model,compute_units,mean_ms,median_ms")

    for modelName in modelNames {
        try benchmark(modelName: modelName, computeUnits: .all, label: "all")
        try benchmark(modelName: modelName, computeUnits: .cpuOnly, label: "cpuOnly")
        try benchmark(modelName: modelName, computeUnits: .cpuAndGPU, label: "cpuAndGPU")
        try benchmark(modelName: modelName, computeUnits: .cpuAndNeuralEngine, label: "cpuAndNeuralEngine")
    }
} catch {
    print("Error: \(error.localizedDescription)")
}
