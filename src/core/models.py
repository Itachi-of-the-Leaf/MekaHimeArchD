import onnxruntime as ort
import numpy as np

class BSRNNSeparator:
    def __init__(self, model_path="models/bsrnn.onnx"):
        # Use Vulkan/CoreML/DirectML based on availability
        providers = ['VulkanExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

    def separate(self, audio_chunk):
        # audio_chunk shape: (1, samples)
        # Expected model input: (1, samples)
        # Expected model output: (2, samples) - one for each speaker
        inputs = {self.session.get_inputs()[0].name: audio_chunk.astype(np.float32)}
        outputs = self.session.run(None, inputs)
        return outputs[0] # Should be [speaker1, speaker2]

class ECAPATDNNManager:
    def __init__(self, model_path="models/ecapa.onnx", ema_alpha=0.1):
        providers = ['VulkanExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.ema_alpha = ema_alpha
        self.identities = {} # id -> current_embedding (EMA)

    def get_embedding(self, audio_chunk):
        inputs = {self.session.get_inputs()[0].name: audio_chunk.astype(np.float32)}
        outputs = self.session.run(None, inputs)
        return outputs[0]

    def update_identity(self, identity_id, new_embedding):
        if identity_id not in self.identities:
            self.identities[identity_id] = new_embedding
        else:
            # EMA: current = (1 - alpha) * current + alpha * new
            self.identities[identity_id] = (1 - self.ema_alpha) * self.identities[identity_id] + self.ema_alpha * new_embedding
        return self.identities[identity_id]
