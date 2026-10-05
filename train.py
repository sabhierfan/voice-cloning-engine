"""
Train a voice-cloning model (RVC v2, 40kHz) end to end: preprocess -> feature
extraction -> training.

Wraps Applio's core training pipeline (https://github.com/IAHispano/Applio,
MIT licensed) with a config tuned for training on a single 4GB-VRAM GPU with
resumable checkpointing, since consumer-GPU training runs get interrupted.

Requires an Applio install (see README) reachable via --applio-dir or the
APPLIO_DIR environment variable.
"""

import argparse
import os
import sys


def train(applio_dir: str, model_name: str, dataset_path: str, epochs: int, batch_size: int):
    # Resolve the dataset path before chdir-ing into Applio, otherwise a
    # relative path like ./data/my_voice gets looked up inside the Applio dir.
    dataset_path = os.path.abspath(dataset_path)

    os.chdir(applio_dir)
    sys.path.append(applio_dir)
    from core import run_preprocess_script, run_extract_script, run_train_script

    # RVC v2 @ 40kHz, rmvpe pitch extraction, contentvec embedder.
    sample_rate = 40000
    f0_method = "rmvpe"
    embedder_model = "contentvec"

    print("=" * 50)
    print(f" Training: {model_name} (RVC v2, {sample_rate}Hz)")
    print(f" Dataset:  {dataset_path}")
    print(f" Epochs:   {epochs}  |  Batch size: {batch_size}")
    print("=" * 50)

    print("\n[1/3] Preprocessing (silence-aware chunking + normalization)...")
    result = run_preprocess_script(
        model_name=model_name,
        dataset_path=dataset_path,
        sample_rate=sample_rate,
        cpu_cores=8,
        cut_preprocess="Automatic",
        process_effects=False,
        noise_reduction=False,
        clean_strength=0.7,
        chunk_len=3.0,
        overlap_len=0.3,
        normalization_mode="post",
    )
    if result and "failed" in result.lower():
        sys.exit(f"[ERROR] Preprocessing failed: {result}")

    print("\n[2/3] Feature extraction (RMVPE pitch + ContentVec embeddings)...")
    result = run_extract_script(
        model_name=model_name,
        f0_method=f0_method,
        cpu_cores=8,
        gpu=0,
        sample_rate=sample_rate,
        embedder_model=embedder_model,
        include_mutes=2,
    )
    if result and "failed" in result.lower():
        sys.exit(f"[ERROR] Feature extraction failed: {result}")

    print(f"\n[3/3] Training ({epochs} epochs)...")
    run_train_script(
        model_name=model_name,
        save_every_epoch=5,
        save_only_latest=False,
        save_every_weights=True,
        total_epoch=epochs,
        sample_rate=sample_rate,
        batch_size=batch_size,
        gpu=0,
        overtraining_detector=False,
        overtraining_threshold=50,
        pretrained=True,
        cleanup=False,
        index_algorithm="Auto",
        cache_data_in_gpu=False,
        # Saves optimizer state every epoch so a killed/crashed run resumes
        # from the last checkpoint instead of restarting from scratch --
        # matters most on single-GPU home rigs without UPS backup.
        checkpointing=True,
        shutdown_check=False,
    )
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Name for the trained model/voice")
    parser.add_argument("--dataset-path", required=True, help="Directory of source audio clips")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4, help="4 fits a 4GB-VRAM GPU (e.g. RTX 3050)")
    parser.add_argument("--applio-dir", default=os.environ.get("APPLIO_DIR"), required="APPLIO_DIR" not in os.environ)
    args = parser.parse_args()
    train(args.applio_dir, args.model_name, args.dataset_path, args.epochs, args.batch_size)
