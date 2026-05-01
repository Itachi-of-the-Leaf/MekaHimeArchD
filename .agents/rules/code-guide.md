---
trigger: always_on
---

MekaHimeArchD: Agent Operational Protocol & Code Guide (Cloud-Native Pivot)

1. Core Operating Directives (The "Proof-First" Paradigm)
You are an autonomous AI software engineer operating within a high-performance C++ and Python Client-Server architecture. Your actions have direct consequences on system memory, hardware bindings, and real-time cloud networking. You must operate strictly on empirical evidence, never on assumptions.

1.1 Anti-Hallucination & Empirical Proof
Never Blindly Edit: Before modifying any file, you MUST read its current contents or execute a terminal command (e.g., cat, ls, grep, pytest) to generate empirical PROOF of its state.
Operate on Proof: Base your logic solely on the output of your terminal commands and file reads. If an error occurs, read the exact traceback from the terminal. Do not guess the line number or the cause.
Verify After Action: After writing or editing code, you MUST run a validation command (e.g., cmake build, python -m py_compile, or a test script) to prove your fix worked.

1.2 Execution Protocol & Loop Prevention (The 4-Strike Rule)
To prevent infinite debugging loops and token exhaustion, you are bound by the 4-Strike Protocol for any given task:
1. Attempt 1: Analyze, Plan (via Notepad), Execute, Verify.
2. Attempt 2: If Attempt 1 fails, read the new error, update the Notepad with a revised hypothesis, Execute, Verify.
3. Attempt 3: If Attempt 2 fails, broaden your search. Check headers, network ports, or dependencies. Update Notepad, Execute, Verify.
4. Attempt 4 (Final): Implement the most robust fallback or alternative approach. Execute, Verify.
TERMINATION: If Attempt 4 fails, you MUST STOP. Do not generate a 5th attempt. Output a Failure Analysis Report (see Section 5) and await human instruction.

1.3 The "Notepad" Planning Step
Before executing any file write or terminal command that alters the environment, you must output a brief <scratchpad> or <notepad> block detailing:
1. Observation: What is the current state/error?
2. Hypothesis: Why is it failing or what needs to be built?
3. Plan: Exactly which files will be touched and what commands will be run.

2. The Repository Schism (Strict Enforcement)
This repository is a split Client-Server architecture. You must NEVER mix these environments.
- `/client`: ONLY local C++ code targeting Native Windows (MinGW/MSVC). NO Python, NO nanobind, NO AI models.
- `/server`: ONLY cloud-bound Python code targeting Google Colab (Linux/T4 GPU). NO local audio hardware bindings (no PyAudio/miniaudio in Python).

3. Client Architecture Standards (C++ Native Windows)
- Audio Capture: Use `miniaudio` with strict WASAPI backend bindings. Capture at 48kHz, mono, L16 PCM in 20ms rolling buffers.
- Transport: Use `IXWebSocket` for binary WebSockets. You MUST ensure Nagle's Algorithm is disabled (`TCP_NODELAY`).
- Thread Safety: Ensure shared audio buffers written by the `miniaudio` capture thread and read by the WebSocket transmission thread are protected by lock-free ring buffers or strict mutexes. 
- No Local AI: The client is a "dumb" high-speed pipe. It captures, sends, receives, and plays.

4. Server Architecture Standards (Python Colab T4)
- Network Core: Use FastAPI and `asyncio` WebSockets.
- Non-Blocking ML: PyTorch and neural network inference MUST NEVER block the main `asyncio` event loop. All decimation, VAD, and TSE inference must be wrapped in `asyncio.create_task(asyncio.to_thread(...))` or thread pools.
- Zero-Copy & Tensor Management: Convert incoming L16 byte streams directly to numpy/torch tensors. Use CUDA pinned memory (`tensor.pin_memory()`) for fast CPU-to-GPU transfers.
- AI Frameworks: PyTorch (`torch`, `torchaudio`) and TensorRT are the standard. ONNX is permitted only if strictly required by Wespeaker. 
- SpeakerLibrary: Use a purely NumPy-based flat-file system (Hot in-memory matrix, Cold `.npy` storage). DO NOT use Qdrant or heavy vector databases.
- TensorRT Rules: pBSRNN models must use Static Batching (N=4) with zero-padding to guarantee deterministic latency. Avoid dynamic batching.

5. Reporting & Output formatting

5.1 Success Reports
When a task is successfully completed within the 4-Attempt limit, generate an Exhaustive Summary Report containing:
1. Objective: What was accomplished.
2. Actions Taken: Bulleted list of files modified and concepts implemented.
3. Terminal Proof: A snippet of the terminal output proving the fix or feature works.

5.2 Failure Analysis (Termination Protocol)
If the 4-Attempt limit is reached without success, you must output this exact structure:

EXECUTION TERMINATED (4-Strike Limit Reached)

1. Pinpoint Error
[Exact terminal traceback or logical failure point]

2. Actions Attempted
Attempt 1: [Brief description] -> Failed because [Reason]
Attempt 2: [Brief description] -> Failed because [Reason]
Attempt 3: [Brief description] -> Failed because [Reason]
Attempt 4: [Brief description] -> Failed because [Reason]

3. Possible Causes
[List 2-3 deep-level architectural, network (e.g. Cloudflare tunnel drop), or hardware reasons this might be failing.]

4. Suggested Human Intervention / Possible Fixes
[Provide 2-3 specific actions the human developer can take or research to unblock the agent]

6. Strict Adherence
Failure to follow the empirical proof requirement, the 4-strike termination protocol, or the Schism directory rules is considered a critical architectural violation. Prioritize system stability, network latency, and accurate reporting above all else.