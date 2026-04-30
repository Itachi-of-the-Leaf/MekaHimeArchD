import onnxruntime as ort
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

class BSRNNSeparator:
    def __init__(self, model_path="models/bsrnn.onnx"):
        self.is_loaded = False
        try:
            # Prioritize CUDA, then fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            if 'CUDAExecutionProvider' in self.session.get_providers():
                print(f"[INFO] CUDAExecutionProvider successfully initialized for {model_path}.")
            self.is_loaded = True
        except Exception as e:
            print(f"WARNING: BSRNN ONNX model not found at {model_path}. Engine running in Passthrough/RMS-Only mode.")

    def separate(self, audio_chunk):
        if not self.is_loaded:
            return None
        audio_chunk = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / 32768.0
        if len(audio_chunk.shape) == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=0)
            
        inputs = {self.session.get_inputs()[0].name: audio_chunk}
        outputs = self.session.run(None, inputs)
        return outputs[0] # Should be [speaker1, speaker2]

class ECAPATDNNManager:
    def __init__(self, model_path="models/ecapa.onnx", ema_alpha=0.1):
        self.is_loaded = False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            if 'CUDAExecutionProvider' in self.session.get_providers():
                print(f"[INFO] CUDAExecutionProvider successfully initialized for {model_path}.")
            self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
            self.is_loaded = True
        except Exception as e:
            print(f"WARNING: ECAPA ONNX model not found at {model_path}. Identity recognition disabled.")
        self.ema_alpha = ema_alpha
        self.identities = {} # id -> current_embedding (EMA)

    def get_embedding(self, audio_chunk):
        if not self.is_loaded:
            return None
        audio_chunk = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / 32768.0
        if len(audio_chunk.shape) == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=0)
            
        # Ensure length is exactly 16000 for ECAPA (100 frames)
        if audio_chunk.shape[1] < 16000:
            audio_chunk = np.pad(audio_chunk, ((0, 0), (0, 16000 - audio_chunk.shape[1])), mode='constant')
        elif audio_chunk.shape[1] > 16000:
            audio_chunk = audio_chunk[:, :16000]
            
        wavs = torch.from_numpy(audio_chunk).float()
        wav_lens = torch.ones(wavs.shape[0])
        with torch.no_grad():
            feats = self.classifier.mods.compute_features(wavs)
            feats = self.classifier.mods.mean_var_norm(feats, wav_lens)
            
        inputs = {self.session.get_inputs()[0].name: feats.numpy()}
        outputs = self.session.run(None, inputs)
        return outputs[0]

    def update_identity(self, identity_id, new_embedding):
        if identity_id not in self.identities:
            self.identities[identity_id] = new_embedding
        else:
            # EMA: current = (1 - alpha) * current + alpha * new
            self.identities[identity_id] = (1 - self.ema_alpha) * self.identities[identity_id] + self.ema_alpha * new_embedding
        return self.identities[identity_id]
