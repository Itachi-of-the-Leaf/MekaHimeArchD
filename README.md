# Amika Server — Colab Deployment Guide (Phase 2: VAD Gate)

## Architecture Overview

```
Windows PC                          Google Colab (T4)
─────────────────                   ─────────────────────────────────────
amika_client.exe                    amika_server.py (FastAPI/granian)
   │                                   │
   │  Binary WebSocket                 ├── [Echo 48kHz back immediately]
   │  L16 PCM, 20ms frames             └── [Silero VAD v4 — fire & forget]
   └──────── wss://xxx.trycloudflare.com/audio ──────────────────┘
                        (Cloudflare Tunnel, TLS terminated at edge)
```

---

## Step 1 — Mount Google Drive & Clone Repo

```python
# ── Cell 1: Run once per Drive session ──
from google.colab import drive
drive.mount('/content/drive')

import os, sys

REPO_DIR = '/content/drive/MyDrive/MekaHimeArchD'

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/Itachi-of-the-Leaf/MekaHimeArchD.git "{REPO_DIR}"

# Always pull & checkout the Drive_Pivot branch
!git -C "{REPO_DIR}" fetch origin
!git -C "{REPO_DIR}" checkout Drive_Pivot
!git -C "{REPO_DIR}" pull --rebase origin Drive_Pivot

# Add repo root to Python path so `server.amika_server` is importable
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print("✅ Repo ready:", REPO_DIR)
```

---

## Step 2 — Install Server Dependencies

```python
# ── Cell 2 ──
!pip install -q -r server/requirements.txt
print("✅ Dependencies installed.")
```

---

## Step 3 — Start the Amika Server

```python
# ── Cell 3: Starts granian (Rust ASGI server) in the background ──
import subprocess, threading, time

server_proc = subprocess.Popen(
    ["python", "-m", "granian",
     "--interface", "asgi",
     "--host",      "0.0.0.0",
     "--port",      "8765",
     "server.amika_server:app"],
    cwd=REPO_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

def _stream_logs(proc):
    for line in proc.stdout:
        print("[server]", line, end="")

threading.Thread(target=_stream_logs, args=(server_proc,), daemon=True).start()
time.sleep(3)  # granian needs a moment longer than uvicorn to bind
print(f"✅ Server started (PID {server_proc.pid}). Listening on :8765")
```

---

## Step 4 — Install and Start Cloudflare Tunnel

```python
# ── Cell 4a: Install cloudflared (one-time per Colab session) ──
import subprocess
result = subprocess.run(
    ["wget", "-q",
     "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
     "-O", "/usr/local/bin/cloudflared"],
    capture_output=True
)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"])
v = subprocess.run(["/usr/local/bin/cloudflared", "version"], capture_output=True, text=True)
print("✅ cloudflared installed:", v.stdout.strip())
```

```python
# ── Cell 4b: Start the tunnel and capture the public URL ──
import subprocess, threading, re, time

tunnel_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel",
     "--url", "http://localhost:8765",
     "--no-autoupdate"],
    stderr=subprocess.PIPE,
    text=True,
)

tunnel_url    = None
websocket_url = None

def _find_tunnel_url(proc):
    global tunnel_url, websocket_url
    for line in proc.stderr:
        # cloudflared prints the URL to stderr
        match = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
        if match and tunnel_url is None:
            tunnel_url    = match.group(0)
            websocket_url = tunnel_url.replace("https://", "wss://") + "/audio"
            print("\n" + "="*60)
            print(f"  ✅ Tunnel URL : {tunnel_url}")
            print(f"  ✅ WS URL     : {websocket_url}")
            print("="*60)
            print(f"\n  Run on Windows:")
            print(f"  amika_client.exe {websocket_url}\n")

threading.Thread(target=_find_tunnel_url, args=(tunnel_proc,), daemon=True).start()

# Wait up to 15 s for the URL to appear
for _ in range(30):
    if tunnel_url: break
    time.sleep(0.5)
else:
    print("⚠️  Tunnel URL not detected in 15s — check cloudflared output manually.")
```

---

## Step 5 — Health Check

```python
# ── Cell 5: Verify the server is reachable through the tunnel ──
import urllib.request, json

if tunnel_url:
    with urllib.request.urlopen(f"{tunnel_url}/healthz", timeout=10) as r:
        data = json.loads(r.read())
    print("Health check:", data)
    # Expected: {"status": "ok", "phase": 1}
```

---

## Step 6 — Colab Keep-Alive (Prevents Session Timeout)

Paste this snippet into your browser **JavaScript console** (`F12 → Console`) while the Colab tab is open. It clicks the "Connect" button every 60 seconds to prevent the runtime from disconnecting.

```javascript
// Colab Keep-Alive — prevents idle disconnect
(function keepAlive() {
    const selectors = [
        'colab-connect-button',
        '#top-toolbar > colab-connect-button',
    ];
    for (const sel of selectors) {
        const btn = document.querySelector(sel);
        if (btn) { btn.click(); break; }
    }
    console.log('[keep-alive] ping at', new Date().toLocaleTimeString());
    setTimeout(keepAlive, 55000);
})();
```

---

## Latency Targets

| Network Path | Expected RTT |
|---|---|
| Localhost loopback (dev test) | < 5 ms |
| LAN (same router) | < 15 ms |
| WSL → Colab via Cloudflare Tunnel | < 100 ms |
| Windows → Colab via Cloudflare Tunnel | < 120 ms |

**Measured lag** is printed by the client every second:
```
[RTT] sent=50  echoed=48  lag=2 frames (+40 ms)
```

---

## Teardown

```python
# ── Stop everything ──
tunnel_proc.terminate()
server_proc.terminate()
print("Stopped tunnel and server.")
```

---

---

# ☁️ Cloud Deployment: Google Colab (Ngrok Pipeline)

> **Alternative Workflow** — Use this pipeline instead of the Cloudflare tunnel above when you need a more robust, token-authenticated tunnel with explicit health-check polling and deterministic server boot sequencing. This is the **recommended** path for first-time Colab deployments.

This pipeline establishes a low-latency, end-to-end encrypted WebSocket connection from your local **C++ WASAPI client** (`amika_client.exe`) running on Windows to the **FastAPI/Granian ASGI server** running on a Colab T4 GPU — all bridged through an **Ngrok reverse tunnel**.

```
┌─────────────────────────────┐          ┌──────────────────────────────────────────────────────┐
│   Windows PC (C++ Client)   │          │               Google Colab — T4 GPU                  │
│                             │          │                                                      │
│  amika_client.exe           │          │  granian  →  server.amika_server:app (FastAPI)        │
│  ────────────────           │          │  ──────────────────────────────────────────           │
│  WASAPI 48kHz Capture       │◄────────►│  VAD Gate  →  Wespeaker ResNet Embeddings             │
│  L16 PCM 20ms frames        │          │  Speaker Library  →  Real-time Diarization            │
│  IXWebSocket binary send    │          │  Echo back to client                                 │
└─────────────────────────────┘          └──────────────────────────────────────────────────────┘
                │                                              ▲
                │       wss://xxxxxxxx.ngrok-free.app/audio    │
                └──────────────── Ngrok TLS Tunnel ────────────┘
                                (Ngrok authenticates at edge,
                                 TCP_NODELAY honoured end-to-end)
```

---

## 🔧 Cell 1 — Environment & Repository Synchronization

This cell mounts your Google Drive and performs a hard sync to the `Drive_Pivot` branch. It will **clone** the repository on first run, then always `fetch` and `pull --rebase` to guarantee the Colab runtime has the exact latest server code before anything is started. The repo root is also injected into `sys.path` so `server.amika_server` is importable as a Python module by Granian.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys

REPO_DIR = '/content/drive/MyDrive/MekaHimeArchD'

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/Itachi-of-the-Leaf/MekaHimeArchD.git "{REPO_DIR}"

# Always pull & checkout the Drive_Pivot branch
!git -C "{REPO_DIR}" fetch origin
!git -C "{REPO_DIR}" checkout Drive_Pivot
!git -C "{REPO_DIR}" pull --rebase origin Drive_Pivot

# Add repo root to Python path so `server.amika_server` is importable
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print("✅ Repo ready:", REPO_DIR)
```

---

## 📦 Cell 2 — Dependency Resolution

This cell installs all Python packages declared in `server/requirements.txt` into the Colab runtime. This includes **Granian** (Rust ASGI server), **Wespeaker**, **FastAPI**, **PyTorch / Torchaudio** extensions, and **pyngrok**. Running this after Cell 1 guarantees the requirements file is already at its latest committed version.

```python
!pip install -q -r server/requirements.txt
print("✅ Dependencies installed.")
```

---

## 🚀 Cell 3 — Master Boot & Ngrok Tunnel

> [!CAUTION]
> **Action Required Before Running This Cell.**
> You must create a **free account at [ngrok.com](https://ngrok.com)** and obtain your personal **Authtoken** from the [Ngrok Dashboard → Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken). Replace the placeholder string `"YOUR_NGROK_AUTH_TOKEN"` inside the cell with your real token. Without a valid token, Ngrok will refuse the connection and the cell will exit with a fatal error. The token bypasses Colab's firewall restrictions and authenticates your tunnel session.

This cell orchestrates the entire server boot sequence in six stages:

| Stage | What Happens |
|---|---|
| **1 — Zombie Cleanup** | Kills any lingering `granian` process from a previous cell run to prevent port conflicts. |
| **2 — Wespeaker Prefetch** | Applies a PyTorch/Torchaudio monkey-patch to silence deprecated `sox_effects` calls, then downloads and caches the Wespeaker English ResNet weights into VRAM before the server starts. |
| **3 — Granian Boot** | Starts the Granian ASGI server on `127.0.0.1:8765` (IPv4 loopback only — Ngrok will be the public face), logging all output to `server_logs.txt`. |
| **4 — Smart Health Poll** | Polls `GET /healthz` every 2 seconds for up to 90 seconds, waiting for all AI models to finish loading into VRAM before proceeding. Exits immediately if Granian crashes. |
| **5 — Ngrok Tunnel** | Authenticates with your token, kills any stale tunnel, and opens a new HTTPS tunnel to port `8765`. |
| **6 — Public Health Check** | Verifies the public endpoint is reachable through the tunnel, then prints the final **`wss://` endpoint** for your C++ client. |

```python
import os, sys, time, subprocess, urllib.request, json, re

REPO_DIR = '/content/drive/MyDrive/MekaHimeArchD'
os.chdir(REPO_DIR)

print("🧹 1. Cleaning up zombie processes...")
os.system("pkill -f granian")
time.sleep(1)

print("\n📦 2. Pre-fetching Wespeaker Weights...")
# --- THE ULTIMATE PYTORCH MONKEY PATCH ---
import sys
import types
import torchaudio

if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

if "torchaudio.sox_effects" not in sys.modules:
    mock_sox = types.ModuleType("torchaudio.sox_effects")
    mock_sox.apply_effects_tensor = lambda tensor, effects, **kwargs: (tensor, 16000)
    sys.modules["torchaudio.sox_effects"] = mock_sox
    torchaudio.sox_effects = mock_sox
# ----------------------------------------------

import wespeaker
try:
    model = wespeaker.load_model("english")
    print("   ✓ Wespeaker model downloaded and cached successfully!")
except Exception as e:
    print("   ❌ Wespeaker failed to load:", e)
    sys.exit(1)

print("\n🚀 3. Starting Granian Server (IPv4 Mode)...")
log_file = open("server_logs.txt", "w")
server_proc = subprocess.Popen(
    ["python", "-m", "granian", "--interface", "asgi", "--host", "127.0.0.1", "--port", "8765", "server.amika_server:app"],
    cwd=REPO_DIR, stdout=log_file, stderr=subprocess.STDOUT
)

print("\n🩺 4. Local Health Check (Smart Polling)...")
print("   Waiting for AI models to load into VRAM (can take up to 90 seconds)...")
local_success = False
for attempt in range(45):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=2) as r:
            print(f"   ✓ LOCAL SUCCESS (Attempt {attempt+1}):", json.loads(r.read()))
            local_success = True
            break
    except urllib.error.URLError as e:
        if server_proc.poll() is not None:
            print("   ❌ FATAL: Granian process died unexpectedly during boot.")
            break
        time.sleep(2)
    except Exception as e:
        time.sleep(2)

if not local_success:
    print("   ❌ LOCAL FAILED! Server timed out or crashed.")
    os.system("tail -n 20 server_logs.txt")
    with open("server_logs.txt", "r") as f:
        print(f.read()[-1500:])
    sys.exit(1)

print("\n🌐 5. Verifying & Starting Ngrok Tunnel...")
try:
    import pyngrok
except ImportError:
    os.system("pip install -q pyngrok")

from pyngrok import ngrok

# --- ACTION REQUIRED ---
NGROK_TOKEN = "YOUR_NGROK_AUTH_TOKEN"  
# -----------------------

if NGROK_TOKEN == "YOUR_NGROK_AUTH_TOKEN":
    print("   ❌ FATAL: You need an Ngrok Auth Token.")
    sys.exit(1)

ngrok.set_auth_token(NGROK_TOKEN)
ngrok.kill() 

try:
    tunnel = ngrok.connect(8765)
    tunnel_url = tunnel.public_url
    print(f"   ✓ Tunnel Established: {tunnel_url}")
except Exception as e:
    print(f"   ❌ Failed to establish Ngrok tunnel: {e}")
    sys.exit(1)

print("\n🩺 6. Public Health Check...")
time.sleep(3)

public_success = False
for attempt in range(5):
    try:
        req = urllib.request.Request(f"{tunnel_url}/healthz", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            print("   ✓ PUBLIC SUCCESS:", json.loads(r.read()))
            public_success = True
            wss_url = tunnel_url.replace('https://', 'wss://').replace('http://', 'ws://')
            print("\n🎉 AMIKA EARS ARE ONLINE. CONNECT YOUR C++ CLIENT TO:")
            print(f"   {wss_url}/audio")
            break
    except Exception as e:
        time.sleep(4)

if not public_success:
    print("   ❌ PUBLIC FAILED. Ngrok might be throttling or Granian dropped the connection.")
```

---

## 🎯 Connecting the C++ Client

Once Cell 3 prints the final `wss://` URL, pass it directly as the first argument to the Windows executable:

```bat
amika_client.exe wss://xxxxxxxx.ngrok-free.app/audio
```

The client will immediately begin streaming **48kHz mono L16 PCM** in 20ms binary frames over the authenticated Ngrok WebSocket tunnel.

---

## 🆚 Ngrok vs. Cloudflare Tunnel — Quick Comparison

| Feature | Cloudflare Tunnel | Ngrok Pipeline |
|---|---|---|
| **Auth required** | No (anonymous) | Yes (free account) |
| **Boot sequence** | Simple | Staged with health checks |
| **Wespeaker prefetch** | Manual | Automatic (Cell 3) |
| **Reconnect on drop** | Auto | Manual re-run Cell 3 |
| **URL persistence** | Random per session | Random per session (paid plan for static) |
| **Recommended for** | Quick dev tests | First-time setup & reliable demos |

---

## 🩺 Debugging: Reading Server Logs

If Cell 3 fails at the health poll stage, inspect the Granian log file directly:

```python
# Run this in a new cell to tail the server log
with open("/content/drive/MyDrive/MekaHimeArchD/server_logs.txt") as f:
    print(f.read()[-3000:])
```

Common failure causes and fixes:

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Granian process died unexpectedly` | Import error in `amika_server.py` | Check `server_logs.txt` for Python traceback |
| `Wespeaker failed to load` | Missing internet in Colab or corrupt cache | Runtime → Disconnect and delete runtime, retry |
| `Ngrok tunnel refused` | Invalid or expired auth token | Regenerate token at dashboard.ngrok.com |
| `Public health check failed` | Ngrok free tier rate-limit | Wait 60 seconds, re-run Cell 3 only |
