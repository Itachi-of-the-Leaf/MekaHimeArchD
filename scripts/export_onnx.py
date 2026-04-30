import os
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

def export_ecapa():
    print("Loading ECAPA-TDNN PyTorch model...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

    # Dummy input for the embedding model: Batch size 1, 100 frames, 80 channels
    dummy_input = torch.randn(1, 100, 80, device=classifier.device)

    print("Tracing ECAPA...")
    traced_model = torch.jit.trace(classifier.mods.embedding_model, dummy_input, check_trace=False)

    print("Exporting ECAPA to ONNX...")
    torch.onnx.export(
        traced_model, # Export only the core embedding network
        dummy_input,
        "models/ecapa.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['embedding'],
        dynamic_axes={'input_features': {0: 'batch_size', 1: 'time'}, 
                      'embedding': {0: 'batch_size'}},
        dynamo=False
    )
    print("Successfully exported models/ecapa.onnx!")

def export_separator():
    print("Loading SepFormer PyTorch model...")
    separator = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer-wsj02mix")

    class SepWrapper(torch.nn.Module):
        def __init__(self, sep):
            super().__init__()
            self.sep = sep
        def forward(self, x):
            return self.sep(x)
            
    wrapper = SepWrapper(separator)
    wrapper.eval()

    # Dummy input: Batch size 1, 16000 samples
    dummy_input = torch.randn(1, 16000, device=separator.device)

    print("Exporting SepFormer to ONNX as models/bsrnn.onnx...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        "models/bsrnn.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_audio'],
        output_names=['separated_sources'],
        dynamic_axes={'input_audio': {0: 'batch_size', 1: 'time'}, 
                      'separated_sources': {0: 'batch_size', 2: 'time'}},
        dynamo=False
    )
    print("Successfully exported models/bsrnn.onnx!")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    export_ecapa()
    export_separator()
