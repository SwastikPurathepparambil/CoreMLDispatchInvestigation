import Foundation
import CoreML

let modelNames = [
    "MobileNetV2_fp16",
    "ResNet50_fp16",
    "SimpleMLP_fp16",
    "DistilBERT_fp16",
    "TinyGPT_fp16"
]

func modelURL(for modelName: String) throws -> URL {
    guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
        throw NSError(
            domain: "ComputePlanInspector",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "Could not find compiled model \(modelName).mlmodelc"
            ]
        )
    }

    return url
}

func collectOperations(
    from block: MLModelStructure.Program.Block,
    into operations: inout [MLModelStructure.Program.Operation]
) {
    for operation in block.operations {
        operations.append(operation)

        for nestedBlock in operation.blocks {
            collectOperations(from: nestedBlock, into: &operations)
        }
    }
}

func inspectModel(modelName: String) async throws {
    print("MODEL,\(modelName)")

    let url = try modelURL(for: modelName)

    let config = MLModelConfiguration()
    config.computeUnits = .all

    let computePlan = try await MLComputePlan.load(
        contentsOf: url,
        configuration: config
    )

    switch computePlan.modelStructure {

    case .program(let program):
        print("STRUCTURE,program")
        print("FUNCTION_COUNT,\(program.functions.count)")

        var allOperations: [MLModelStructure.Program.Operation] = []

        for (functionName, function) in program.functions {
            var functionOperations: [MLModelStructure.Program.Operation] = []
            collectOperations(from: function.block, into: &functionOperations)

            print("FUNCTION,\(functionName)")
            print("OPERATION_COUNT,\(functionOperations.count)")

            allOperations.append(contentsOf: functionOperations)
        }

        
        var operatorCounts: [String: Int] = [:]
        var operatorDeviceCounts: [String: Int] = [:]

        for operation in allOperations {
            let operatorName = operation.operatorName
            operatorCounts[operatorName, default: 0] += 1

            let usage = computePlan.deviceUsage(for: operation)
            let usageString = String(describing: usage)

            let key = "\(operatorName),\(usageString)"
            operatorDeviceCounts[key, default: 0] += 1
        }

        print("OPERATOR_SUMMARY_BEGIN")

        for (operatorName, count) in operatorCounts.sorted(by: { $0.key < $1.key }) {
            print("\(operatorName),\(count)")
        }

        print("OPERATOR_SUMMARY_END")

        print("OPERATOR_DEVICE_SUMMARY_BEGIN")

        for (key, count) in operatorDeviceCounts.sorted(by: { $0.key < $1.key }) {
            print("\(key),\(count)")
        }

        print("OPERATOR_DEVICE_SUMMARY_END")

        print("TOTAL_OPERATIONS,\(allOperations.count)")
        

    case .neuralNetwork(let network):
        print("STRUCTURE,neuralNetwork")
        print("LAYER_COUNT,\(network.layers.count)")

        var layerTypeCounts: [String: Int] = [:]

        for layer in network.layers {
            let layerDescription = String(describing: layer)
            layerTypeCounts[layerDescription, default: 0] += 1
        }

        print("LAYER_SUMMARY_BEGIN")

        for (layerType, count) in layerTypeCounts.sorted(by: { $0.key < $1.key }) {
            print("\(layerType),\(count)")
        }

        print("LAYER_SUMMARY_END")

    case .pipeline(let pipeline):
        print("STRUCTURE,pipeline")
        print("PIPELINE_MODELS,\(pipeline.subModels.count)")

    case .unsupported:
        print("STRUCTURE,unsupported")

    @unknown default:
        print("STRUCTURE,unknown")
    }

    print("END_MODEL,\(modelName)")
    print("")
}

let semaphore = DispatchSemaphore(value: 0)

print("model,field,value")

Task.detached {
    do {
        for modelName in modelNames {
            try await inspectModel(modelName: modelName)
        }

        print("COMPUTE_PLAN_INSPECTOR_DONE")
    } catch {
        print("ERROR,\(error)")
    }

    fflush(stdout)
    semaphore.signal()
}

semaphore.wait()
