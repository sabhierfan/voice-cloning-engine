"""
Offline file-to-file voice conversion: run a source audio file through a
trained RVC model and write the converted result.

Wraps Applio's inference pipeline (https://github.com/IAHispano/Applio, MIT
licensed). Requires an Applio install reachable via --applio-dir or the
APPLIO_DIR environment variable.
"""

import argparse
import os
import sys
import time


def convert(applio_dir: str, input_path: str, model_path: str, index_path: str,
            output_path: str, pitch: int, index_rate: float):
    os.chdir(applio_dir)
    sys.path.append(applio_dir)
    from core import run_infer_script

    print(f"Input:  {input_path}")
    print(f"Model:  {model_path}")
    print(f"Index:  {index_path or '(none)'}")
    print(f"Output: {output_path}")

    start = time.time()
    result, out_path = run_infer_script(
        pitch=pitch,
        index_rate=index_rate if index_path else 0.0,
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
        clean_audio=True,
        clean_strength=0.7,
        export_format="WAV",
        embedder_model="contentvec",
    )
    print(f"[SUCCESS] {result} ({time.time() - start:.2f}s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source audio file to convert")
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--index", default="", help="Path to the model's .index file (optional)")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--pitch", type=int, default=0, help="Semitone shift, e.g. +12 male->female")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--applio-dir", default=os.environ.get("APPLIO_DIR"), required="APPLIO_DIR" not in os.environ)
    args = parser.parse_args()
    convert(args.applio_dir, args.input, args.model, args.index, args.output, args.pitch, args.index_rate)
