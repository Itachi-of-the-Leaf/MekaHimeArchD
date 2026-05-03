# Architecture D - Commercial Safe - MIT License
import os
import time

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Error: onnx and onnxruntime are required for quantization.")
    print("Please install them: uv add onnx onnxruntime")
    exit(1)

def quantize_model(input_model_path: str, output_model_path: str) -> None:
    print(f"Quantizing {input_model_path} to {output_model_path} (INT8)...")
    if not os.path.exists(input_model_path):
        print(f"  Error: {input_model_path} not found.")
        return

    orig_size = os.path.getsize(input_model_path) / (1024 * 1024)

    t0 = time.perf_counter()
    try:
        # Standard dynamic quantization to INT8
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QInt8
        )
        t1 = time.perf_counter()
        
        # Verify
        sess = ort.InferenceSession(output_model_path, providers=['CPUExecutionProvider'])
        quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
        ratio = orig_size / quant_size
        
        print(f"  Success! Quantized in {t1-t0:.2f}s")
        print(f"  Original size: {orig_size:.2f} MB")
        print(f"  Quantized size: {quant_size:.2f} MB")
        print(f"  Compression ratio: {ratio:.2f}x (Estimated equivalent memory bandwidth speedup)")
    except Exception as e:
        print(f"  Failed to quantize: {e}")

def main():
    print("--- Architecture D: Model Quantization ---")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    bsrnn_in = os.path.join(models_dir, "bsrnn.onnx")
    bsrnn_out = os.path.join(models_dir, "bsrnn_int8.onnx")
    quantize_model(bsrnn_in, bsrnn_out)
    
    print("-" * 40)

    ecapa_in = os.path.join(models_dir, "ecapa.onnx")
    ecapa_out = os.path.join(models_dir, "ecapa_int8.onnx")
    quantize_model(ecapa_in, ecapa_out)

if __name__ == "__main__":
    main()