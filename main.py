# main.py
import os
import sys
from granian import Granian

# Dynamically link local pip CUDA libraries
venv_lib = os.path.join(os.path.dirname(sys.executable), "../lib/python3.12/site-packages")
nvidia_cublas = os.path.join(venv_lib, "nvidia/cublas/lib")
nvidia_cudnn = os.path.join(venv_lib, "nvidia/cudnn/lib")
nvidia_cufft = os.path.join(venv_lib, "nvidia/cufft/lib")
nvidia_cuda_runtime = os.path.join(venv_lib, "nvidia/cuda_runtime/lib")
os.environ["LD_LIBRARY_PATH"] = f"{nvidia_cublas}:{nvidia_cudnn}:{nvidia_cufft}:{nvidia_cuda_runtime}:" + os.environ.get("LD_LIBRARY_PATH", "")

def main():
    print("Starting Granian Server...")
    server = Granian(
        target="src.server.app:create_app", 
        address="0.0.0.0", 
        port=8000, 
        workers=1,
        interface="asgi",
        factory=True # Crucial for Granian 2.x string targets
    )
    server.serve()

if __name__ == "__main__":
    main()
