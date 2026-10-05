"""
Parameter sweep: run the same source file through a trained model under
several pitch / index-rate / cleaning combinations, to diagnose artifacts
and pick the best-sounding settings before locking in defaults.

Wraps Applio's inference pipeline (https://github.com/IAHispano/Applio, MIT
licensed). Requires an Applio install reachable via --applio-dir or the
APPLIO_DIR environment variable.
"""

import argparse
import os
import sys

CONFIGS = [
    {"name": "no_cleaning_medium_index", "pitch": 0, "index_rate": 0.5, "clean_audio": False, "clean_strength": 0.5},
    {"name": "no_cleaning_no_index", "pitch": 0, "index_rate": 0.0, "clean_audio": False, "clean_strength": 0.5},
    {"name": "pitch_down_5", "pitch": -5, "index_rate": 0.5, "clean_audio": False, "clean_strength": 0.5},
    {"name": "pitch_up_5", "pitch": 5, "index_rate": 0.5, "clean_audio": False, "clean_strength": 0.5},
]


def sweep(applio_dir: str, input_path: str, model_path: str, index_path: str, output_dir: str):
    # Resolve user paths before chdir-ing into Applio, otherwise relative
    # paths (and the ./sweep_output default) resolve inside the Applio dir.
    input_path = os.path.abspath(input_path)
    model_path = os.path.abspath(model_path)
    index_path = os.path.abspath(index_path)
    output_dir = os.path.abspath(output_dir)

    os.chdir(applio_dir)
    sys.path.append(applio_dir)
    from core import run_infer_script

    os.makedirs(output_dir, exist_ok=True)
    for c in CONFIGS:
        output_path = os.path.join(output_dir, f"{c['name']}.wav")
        print(f"\nRunning: {c['name']} (pitch={c['pitch']}, index_rate={c['index_rate']}, clean={c['clean_audio']})")
        try:
            _, out_path = run_infer_script(
                pitch=c["pitch"],
                index_rate=c["index_rate"],
                volume_envelope=1.0,
                protect=0.33,
                f0_method="rmvpe",
                input_path=input_path,
                output_path=output_path,
                pth_path=model_path,
                index_path=index_path,
                split_audio=True,
                f0_autotune=False,
                f0_autotune_strength=0.25,
                proposed_pitch=False,
                proposed_pitch_threshold=0.5,
                clean_audio=c["clean_audio"],
                clean_strength=c["clean_strength"],
                export_format="WAV",
                embedder_model="contentvec",
            )
            print(f"[SUCCESS] Saved to {out_path}")
        except Exception as e:
            print(f"[ERROR] {c['name']} failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", default="./sweep_output")
    parser.add_argument("--applio-dir", default=os.environ.get("APPLIO_DIR"), required="APPLIO_DIR" not in os.environ)
    args = parser.parse_args()
    sweep(args.applio_dir, args.input, args.model, args.index, args.output_dir)
