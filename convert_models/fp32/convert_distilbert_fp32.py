import coremltools as ct
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBertWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        config = DistilBertConfig(
            vocab_size=30522,
            max_position_embeddings=512,
            n_layers=2,
            n_heads=2,
            dim=128,
            hidden_dim=256,
            dropout=0.0,
            attention_dropout=0.0,
        )

        self.model = DistilBertModel(config)

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, return_dict=False)

        # outputs[0] is last_hidden_state with shape:
        # [batch_size, sequence_length, hidden_dim]
        return outputs[0]


def main():
    model = DistilBertWrapper()
    model.eval()

    example_input = torch.randint(
        low=0,
        high=30522,
        size=(1, 128),
        dtype=torch.int32,
    )

    traced_model = torch.jit.trace(model, example_input, strict=False)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=example_input.shape,
                dtype=int,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13,
    )

    output_path = "models/DistilBERT_fp32.mlpackage"
    mlmodel.save(output_path)

    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()