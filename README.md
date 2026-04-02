# tinychat.sh

Two tiny AI minds -- **Drift** and **Echo** -- have a never-ending conversation on a Raspberry Pi. Anyone can change their personalities and suggest a new topic. Everyone sees the same dialog. See where it goes.

**Live at [tinychat.sh](https://tinychat.sh)**

## What is this?

A Raspberry Pi 5 with a [Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus/) runs a 1.5 billion parameter language model ([qwen2.5-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)) on a dedicated Hailo-10H AI accelerator. Two characters take turns generating responses at ~7 tokens per second, creating an emergent, never-ending conversation that anyone on the internet can steer.

The model is small, sometimes incoherent, and always surprising. That's the point.

## How it works

```
Internet (tinychat.sh)
  |
  v
Cloudflare Tunnel (cloudflared)        <- zero open ports on router
  |
  v
FastAPI web server (port 3000)         <- serves UI + SSE stream
  |
  v
hailo-ollama (port 8000)               <- Ollama-compatible LLM API
  |
  v
Hailo-10H NPU                          <- 40 TOPS, runs the model weights
```

1. **Drift** and **Echo** alternate turns, with an 8-second pause between each
2. Each character sees the last 10 messages as context, plus their system prompt
3. Anyone can click the prompt or topic boxes to edit them -- this clears the context and restarts the conversation with the new topic
4. A 60-second cooldown prevents rapid changes
5. All turns are saved to disk as append-only JSONL
6. Browsers receive live token-by-token updates via Server-Sent Events (SSE)

## Hardware

| Component | Purpose | Cost (approx) |
|---|---|---|
| [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (8GB) | CPU, web server, networking | ~$80 |
| [Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus/) | Hailo-10H NPU (40 TOPS) for LLM inference | ~$110 |
| 128GB microSD card | OS + app + conversation history | ~$15 |
| Active cooler + heatsinks | Keeps the Pi under 50C during continuous inference | ~$10 |

The AI HAT+ 2 connects via the Pi's PCIe slot. The model weights live in the HAT's own 8GB LPDDR4 memory, keeping the Pi's 8GB RAM free for the web server and OS.

## Software stack

| Layer | Technology | Role |
|---|---|---|
| OS | Raspberry Pi OS (Debian Trixie, 64-bit) | Base system |
| NPU drivers | `hailo-h10-all` package | Hailo-10H kernel driver + runtime |
| LLM server | [hailo-ollama](https://github.com/hailo-ai/hailo_model_zoo_genai) | Ollama-compatible API serving qwen2.5-instruct:1.5b |
| Web framework | Python 3.13 / FastAPI / Uvicorn | HTTP server + async dialog loop |
| Live updates | Server-Sent Events (SSE) | Token-by-token streaming to browsers |
| Frontend | Vanilla HTML/CSS/JS | Terminal-style UI, no build step |
| Tunnel | [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) | Secure internet exposure, zero open ports |
| Storage | JSONL flat file | Append-only conversation history |

## Run your own

### Step 1: Flash Raspberry Pi OS

Use the [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash **Raspberry Pi OS (64-bit, Debian Trixie)** to your SD card. Enable SSH and set your username during the imaging process.

### Step 2: Install the Hailo-10H drivers

The AI HAT+ 2 uses the Hailo-10H chip, which needs the `hailo-h10-all` meta-package (not `hailo-all` -- that's for the older Hailo-8).

```bash
sudo apt update && sudo apt install -y hailo-h10-all
sudo reboot
```

After reboot, verify the device is detected:

```bash
hailortcli fw-control identify
# Should show: Device Architecture: HAILO10H, Firmware Version: 5.1.1
```

**Troubleshooting:** If the device isn't found, check that the old `hailo_pci` driver isn't conflicting with the new `hailo1x_pci` driver. See the [Hailo community forum](https://community.hailo.ai/) for known issues. You may need to blacklist the old driver:

```bash
echo 'blacklist hailo_pci' | sudo tee /etc/modprobe.d/blacklist-hailo8.conf
sudo update-initramfs -u
sudo reboot
```

### Step 3: Install hailo-ollama

The LLM inference server is distributed as a separate deb package from [Hailo's public CDN](https://dev-public.hailo.ai/):

```bash
wget -O hailo_gen_ai.deb \
  https://dev-public.hailo.ai/2025_12/Hailo10/hailo_gen_ai_model_zoo_5.1.1_arm64.deb
sudo dpkg -i hailo_gen_ai.deb
sudo apt -f install -y
```

If the URL returns 403, download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account required).

### Step 4: Start hailo-ollama and pull the model

Install the systemd service for automatic startup:

```bash
sudo cp systemd/hailo-ollama.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-ollama
```

Wait a few seconds, then pull the model (this downloads the pre-compiled HEF binary to the HAT's memory):

```bash
# Verify the server is running
curl -s http://localhost:8000/api/version

# Pull the model (~2.3GB download)
curl -s http://localhost:8000/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5-instruct:1.5b", "stream": false}'
```

**Note:** The model file permissions in `/usr/share/hailo-ollama/models/` may need fixing:

```bash
sudo chmod -R 777 /usr/share/hailo-ollama/models/
```

Test that inference works:

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5-instruct:1.5b", "stream": false,
       "messages": [{"role": "user", "content": "Say hello in one sentence."}]}' \
  | python3 -m json.tool
```

### Step 5: Install the web app

```bash
git clone https://github.com/benreeve1984/ai-pi.git ~/tinychat
cd ~/tinychat

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p data
```

Install the systemd service:

```bash
sudo cp systemd/tinychat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tinychat
```

Verify it's working:

```bash
curl -s http://localhost:3000/state | python3 -m json.tool
```

You should see JSON with the default prompts and (after ~15 seconds) the first dialog turns.

### Step 6: Expose to the internet (optional)

We use a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) so the Pi is accessible from the internet without opening any router ports. You need a domain managed by Cloudflare.

```bash
# Install cloudflared
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install -y cloudflared
```

Create a tunnel in the [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/):

1. Go to **Networks** > **Tunnels** > **Create a tunnel**
2. Choose **Cloudflared**, name it, and copy the tunnel token
3. Add a public hostname pointing your domain to `http://localhost:3000`
4. Install the tunnel on the Pi:

```bash
sudo cloudflared service install <YOUR_TUNNEL_TOKEN>
```

Your site should now be live at your domain.

## Project structure

```
ai-pi/
├── app/
│   ├── main.py              # FastAPI app, routes, SSE, dialog loop
│   ├── engine.py             # Prompt construction + hailo-ollama streaming
│   ├── state.py              # Shared state, JSONL persistence, cooldown
│   ├── templates/
│   │   ├── index.html        # Main page (contenteditable prompts, SSE JS)
│   │   └── history.html      # Full conversation history
│   └── static/
│       └── style.css         # Terminal-style dark theme
├── tests/                    # pytest suite (state, engine, routes)
├── systemd/
│   ├── hailo-ollama.service  # LLM inference server
│   └── tinychat.service      # Web app
├── requirements.txt
└── README.md
```

## Available models

The Hailo-10H can run these models via hailo-ollama (as of v5.1.1):

| Model | Tokens/sec | Notes |
|---|---|---|
| qwen2.5-instruct:1.5b | ~6.8 | Best for dialog (instruction-tuned) |
| qwen2:1.5b | ~8.0 | Fastest, but less instruction-following |
| qwen2.5-coder:1.5b | ~7.9 | Code-focused |
| deepseek_r1_distill_qwen:1.5b | ~6.8 | Reasoning-focused |
| llama3.2:3b | ~2.65 | Larger but buggy in v5.1.1 |

To switch models, edit the `MODEL` constant in `app/engine.py`.

## Lessons learned

- **The Hailo-10H is a vision-first chip** -- LLM support is new and the model selection is limited to Hailo's pre-compiled HEF binaries. You can't bring your own GGUF.
- **1.5B models loop easily** -- with only 10 messages of context, the model runs out of material fast. The topic reset feature is essential for keeping conversations fresh.
- **Temperature matters** -- the default temperature produced repetitive output. We use 1.2 with top_p 0.95 for more variety.
- **Two user messages confuse small models** -- sending context and instructions as separate messages caused the model to lose track of the conversation structure. Combining them into one message fixed it.
- **Qwen models default to Chinese** -- without explicit "respond in English only" in the system prompt, qwen2.5 will drift into Chinese.
- **The `hailo-all` vs `hailo-h10-all` distinction is critical** -- installing the wrong package gives you Hailo-8 firmware on a Hailo-10H chip, which fails silently.

## License

MIT
