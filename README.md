# Voice Cloning Engine

Self-hosted, real-time voice cloning: train a voice model on a few hours of
audio, then convert speech into that voice either offline (file-to-file) or
live (microphone in, converted voice out, streamed).

- **Problem**: Third-party voice synthesis APIs are expensive and introduce
  network latency that breaks real-time conversational pipelines.
- **Stack**: PyTorch, RVC v2, HiFi-GAN vocoders, ContentVec, RMVPE pitch
  extraction. Built on top of [Applio](https://github.com/IAHispano/Applio)
  (MIT licensed).
- **Hard part**: training and running a usable model on a single 4GB-VRAM
  consumer GPU, and getting real-time conversion latency low enough that a
  live conversation doesn't fall apart.
- **Result**: a self-hosted, high-fidelity voice cloning pipeline trained on
  as little as 2 hours of clean audio, running entirely offline.

## Pipeline

```
Source audio ─▶ Preprocess ─▶ Extract features ─▶ Train (RVC v2)
                (slice/norm)   (RMVPE + ContentVec)      │
                                                          ▼
                                                    trained model (.pth + .index)
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                     convert.py (offline)                                  realtime_monitor.py
                   file in → converted file out                    mic in → converted audio out, live
```

## Engineering notes

- **VRAM-constrained training** (`train.py`): tuned for a single RTX 3050
  (4GB VRAM) — batch size 4, gradient/optimizer checkpointing enabled every
  epoch so a killed or crashed run resumes from the last checkpoint instead
  of restarting from scratch. Home training rigs get interrupted; the
  pipeline assumes that as normal, not exceptional.
- **Real-time inference** (`realtime_monitor.py`): the whole mic-to-speaker
  path has to complete inside one buffer window or the audio breaks up.
  Default buffer is 64 samples (~171ms) with a 0.15s cross-fade between
  chunks to smooth the seams between converted segments — tuned by ear on
  the same 4GB card the training ran on.
- **Parameter sweeps** (`sweep_settings.py`): pitch shift, index rate
  (how strongly the model pulls timbre toward the trained voice vs. the
  source), and post-processing "cleaning" all trade off against each other.
  Rather than guess, the same source clip gets run through a fixed grid of
  settings so the artifacts are directly comparable side by side.

## Setup

1. Clone and set up [Applio](https://github.com/IAHispano/Applio) per its
   own instructions (Python env, model weights, CUDA).
2. Set `APPLIO_DIR` to your Applio install path, or pass `--applio-dir`
   explicitly to every script below.
3. `pip install -r requirements.txt`

## Usage

```bash
# Train a new voice model
python train.py --model-name my_voice --dataset-path ./data/my_voice --epochs 300

# Convert a file offline
python convert.py --input in.wav --model ./weights/my_voice.pth \
  --index ./logs/my_voice/my_voice.index --output out.wav --pitch 0

# Live mic -> converted voice, streamed
python realtime_monitor.py --model ./weights/my_voice.pth \
  --index ./logs/my_voice/my_voice.index --pitch 0

# Compare pitch/index/cleaning settings side by side
python sweep_settings.py --input in.wav --model ./weights/my_voice.pth \
  --index ./logs/my_voice/my_voice.index --output-dir ./sweep_output
```

## What's not in this repo

Training datasets, trained model weights (`.pth`/`.index`), and converted
audio outputs are intentionally excluded — they're derived from private
voice recordings and aren't published here. Bring your own dataset to
train a model with `train.py`.

## License

All rights reserved. Viewing permitted; commercial use, redistribution, and
derivative works are prohibited without written permission. Applio itself
remains MIT licensed under its own repository.
