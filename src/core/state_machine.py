# Architecture D - Commercial Safe - MIT License
import time
import numpy as np

class PipelineStateMachine:
    def __init__(self, energy_threshold: float = 0.05, similarity_threshold: float = 0.75):
        self.state = "UNIFIED" # UNIFIED or SPLIT
        self.energy_threshold = energy_threshold
        self.similarity_threshold = similarity_threshold
        
    def _rms(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0: return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def update(self, chunk: np.ndarray, bsrnn_out1: np.ndarray = None, bsrnn_out2: np.ndarray = None, 
               ecapa_sim: float = 1.0) -> str:
        """
        Updates the state machine based on the current chunk and optionally model outputs.
        """
        current_rms = self._rms(chunk)
        
        previous_state = self.state

        if self.state == "UNIFIED":
            # Transition to SPLIT if energy is high AND ECAPA similarity between outputs is low
            if current_rms > self.energy_threshold and ecapa_sim < self.similarity_threshold:
                self.state = "SPLIT"
        
        elif self.state == "SPLIT":
            # Transition to UNIFIED if one stream drops below silence threshold
            if bsrnn_out1 is not None and bsrnn_out2 is not None:
                rms1 = self._rms(bsrnn_out1)
                rms2 = self._rms(bsrnn_out2)
                
                # If either stream is silent, we revert to UNIFIED
                if rms1 < self.energy_threshold / 2 or rms2 < self.energy_threshold / 2:
                    self.state = "UNIFIED"

        if previous_state != self.state:
            print(f"[STATE] {time.strftime('%H:%M:%S')} - Transition: {previous_state} -> {self.state}")

        return self.state
