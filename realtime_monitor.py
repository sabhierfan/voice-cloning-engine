"""
Real-time voice changer: streams your microphone through a trained RVC model
and plays the converted voice back to your speakers live.

This is the latency-critical path -- everything downstream of the mic has to
happen inside one buffer window or the conversation breaks up. Buffer size,
cross-fade overlap, and mic gain are exposed as flags specifically so they
can be tuned per-machine; the defaults below (64-sample blocks, ~171ms
buffer, 0.15s cross-fade) are what worked on an RTX 3050 4GB without
audible stutter.

Wraps Applio's realtime inference core (https://github.com/IAHispano/Applio,
MIT licensed). Requires an Applio install reachable via --applio-dir or the
APPLIO_DIR environment variable.
"""

import argparse
import os
import sys

import numpy as np
import sounddevice as sd


def run(applio_dir: str, model_path: str, index_path: str, pitch: int,
        index_rate: float, chunk: int, gain: float):
    os.chdir(applio_dir)
    sys.path.append(applio_dir)
    from rvc.realtime.core import VoiceChanger

    print("=" * 50)
    print(" Real-Time Voice Monitor")
    print(f" Model: {model_path}")
    print(f" Pitch: {pitch:+d} semitones | Gain: {gain}x | Index rate: {index_rate}")
    print("=" * 50)
    print("[INFO] Loading model on GPU...")

    vc = VoiceChanger(
        read_chunk_size=chunk,
        cross_fade_overlap_size=0.15,
        extra_convert_size=0.06,
        model_path=model_path,
        index_path=index_path,
        f0_method="rmvpe",
        embedder_model="contentvec",
        vad_enabled=False,
    )
    print("[OK] Ready -- speak into your mic. Ctrl+C to stop.\n")

    def callback(indata, outdata, frames, time_info, status):
        audio_in = np.clip(indata[:, 0] * gain, -1.0, 1.0)
        try:
            converted, _ = vc.process_audio(
                audio_input=audio_in,
                f0_up_key=pitch,
                index_rate=index_rate,
                protect=0.33,
                volume_envelope=1.0,
            )
            n = min(len(converted), frames)
            outdata[:n, 0] = converted[:n]
            if n < frames:
                outdata[n:, 0] = 0
        except Exception as e:
            print(f"[err] {e}", file=sys.stderr)
            outdata.fill(0)

    try:
        with sd.Stream(
            samplerate=48000,
            blocksize=chunk * 128,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n[Stopped]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--index", required=True, help="Path to the model's .index file")
    parser.add_argument("--pitch", type=int, default=0, help="Semitone shift, e.g. +12 male->female")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--chunk", type=int, default=64, help="Buffer size in samples (~171ms at default)")
    parser.add_argument("--gain", type=float, default=8.0, help="Mic input gain multiplier")
    parser.add_argument("--applio-dir", default=os.environ.get("APPLIO_DIR"), required="APPLIO_DIR" not in os.environ)
    args = parser.parse_args()
    run(args.applio_dir, args.model, args.index, args.pitch, args.index_rate, args.chunk, args.gain)
