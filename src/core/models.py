import onnxruntime as ort
import numpy as np
import torch
import scipy.signal
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
            return None, None
            
        chunk_f32 = audio_chunk.astype(np.float32)
        if np.max(np.abs(chunk_f32)) > 1.0:
            chunk_f32 = chunk_f32 / 32768.0
            
        # Resample from 48kHz to 16kHz (Decimation by 3)
        chunk_16k = chunk_f32[::3]
        
        if len(chunk_16k.shape) == 1:
            chunk_16k = np.expand_dims(chunk_16k, axis=0)
            
        inputs = {self.session.get_inputs()[0].name: chunk_16k}
        outputs = self.session.run(None, inputs)
        out_array = outputs[0]
        
        # Telemetry to verify exact ONNX shape
        print(f"[DEBUG] BSRNN Output Shape: {out_array.shape}")
        
        # Robust slicing based on ONNX rank (handles [batch, sources, time] OR [sources, time])
        if len(out_array.shape) == 3:
            spk1 = out_array[0, 0, :]
            spk2 = out_array[0, 1, :]
        elif len(out_array.shape) == 2:
            spk1 = out_array[0, :]
            spk2 = out_array[1, :]
        else:
            raise ValueError(f"Unexpected BSRNN output shape: {out_array.shape}")
            
        return spk1, spk2

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
        chunk_f32 = audio_chunk.astype(np.float32)
        if np.max(np.abs(chunk_f32)) > 1.0:
            chunk_f32 = chunk_f32 / 32768.0
        # Resample from 48kHz to 16kHz (Decimation by 3)
        chunk_16k = chunk_f32[::3]
        
        if len(chunk_16k.shape) == 1:
            chunk_16k = np.expand_dims(chunk_16k, axis=0)
            
        # Ensure length is exactly 16000 for ECAPA (100 frames)
        if chunk_16k.shape[1] < 16000:
            chunk_16k = np.pad(chunk_16k, ((0, 0), (0, 16000 - chunk_16k.shape[1])), mode='constant')
        elif chunk_16k.shape[1] > 16000:
            chunk_16k = chunk_16k[:, :16000]
            
        wavs = torch.from_numpy(chunk_16k).float()
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
