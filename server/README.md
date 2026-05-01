# Amika Server — Colab Deployment Guide (Phase 1)

## Architecture Overview

```
Windows PC                          Google Colab (T4)
─────────────────                   ─────────────────────────────────────
amika_client.exe                    echo_server.py (FastAPI/uvicorn)
   │                                   │
   │  Binary WebSocket                 │
   │  L16 PCM, 20ms frames             │
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
else:
    !git -C "{REPO_DIR}" pull --rebase origin main

# Add repo root to Python path so `server.echo_server` is importable
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

## Step 3 — Start the Echo Server

```python
# ── Cell 3: Starts uvicorn in the background ──
import subprocess, threading, time

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "server.echo_server:app",
     "--host", "0.0.0.0", "--port", "8765",
     "--log-level", "info"],
    cwd=REPO_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

def _stream_logs(proc):
    for line in proc.stdout:
        print("[server]", line, end="")

threading.Thread(target=_stream_logs, args=(server_proc,), daemon=True).start()
time.sleep(2)
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
