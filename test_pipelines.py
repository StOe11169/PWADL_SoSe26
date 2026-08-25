import argparse
import shutil
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data import get_all_data_paths
from src.evaluation import predict_logits
from src.experiment import build_dataset, build_model, evaluate_multimodal, get_input_key
from src.utils import get_device

TEST_CFG = {
    "num_frames": 4,
    #audio
    "audio_sample_rate": 16000,
    "audio_clip_seconds": 1.0,
    "audio_mono": True,
    "audio_normalize": True,
    "num_audio_clips": 4,
    #DataLoader 
    "batch_size": 2,
    "num_workers": 0,
    #model
    "dropout": 0.3}


#hardcoded test files
EXPECTED_FILES = {"001-driver-yawning.mp4": 1.0,
    "002-driver-normal.mp4": 0.0,
    "003-driver-yawning.mp4": 1.0}

def load_test_dataframe(data_dir):
    #load the three dumy and verify labels
    df = get_all_data_paths(str(data_dir))

    #use fixed order for readable test output
    df = df.sort_values("filepath").reset_index(drop=True)

    found_files = {Path(filepath).name for filepath in df["filepath"]}

    #ensure dummys are found
    assert found_files == set(EXPECTED_FILES), (f"Unexpected files.\n"f"Expected: {set(EXPECTED_FILES)}\n"f"Found:{found_files}")

    #check filename parsing generated correct labels
    for _, row in df.iterrows():
        filename = Path(row["filepath"]).name
        expected_label = EXPECTED_FILES[filename]

        assert row["yawning"] == expected_label, (f"Wrong label for {filename}: "f"expected {expected_label}, "f"got {row['yawning']}")

    print(f"[OK] Found {len(df)} correctly labelled test videos.")

    return df


def check_dataset_sample(dataset, mode):
    #check one sample before runng the whole dataset
    sample = dataset[0]
    input_key = get_input_key(mode)

    assert input_key in sample
    assert "labels" in sample
    assert "filepath" in sample

    tensor = sample[input_key]

    if mode == "visual":
        #expected: [frames, channels, height, width]
        assert tensor.ndim == 4
        assert tensor.shape[0] == TEST_CFG["num_frames"]
        assert tensor.shape[1] == 3

    elif mode == "audio":
        #expected: fixed-length mono waveform, [num_audio_clips, samples_per_clip]
        assert tensor.ndim == 2
        expected_samples = int(TEST_CFG["audio_sample_rate"]* TEST_CFG["audio_clip_seconds"])
        assert tensor.shape == (TEST_CFG["num_audio_clips"], expected_samples)

    print(f"[OK] {mode} dataset sample:{input_key} shape={tuple(tensor.shape)}")


def run_unimodal_test(df, mode, device):
    #test visual or audio:video -> dataset -> model -> raw logits
    
    print(f"\n===== TESTING {mode.upper()} PIPELINE =====")

    dataset = build_dataset(df, TEST_CFG, mode)

    if mode == "visual":
    #verify different videos actually produce different frame tensors
        frames_0 = dataset[0]["frames"]
        frames_1 = dataset[1]["frames"]
        frames_2 = dataset[2]["frames"]

        diff_01 = torch.mean(torch.abs(frames_0 - frames_1)).item()
        diff_02 = torch.mean(torch.abs(frames_0 - frames_2)).item()

        print(f"[CHECK] Visual input differences: " f"001/002={diff_01:.6f}, "f"001/003={diff_02:.6f}")
        assert not torch.allclose(frames_0, frames_1), ("Videos 001 and 002 produced identical frame tensors")
        assert not torch.allclose(frames_0, frames_2),("Videos 001 and 003 produced identical frame tensors")

    #vheck decoding/preprocessing 
    check_dataset_sample(dataset, mode)

    loader = DataLoader(dataset, batch_size=TEST_CFG["batch_size"], num_workers=TEST_CFG["num_workers"], shuffle=False)

    model = build_model(TEST_CFG, mode, device)

    input_key = get_input_key(mode)

    #test raw-logits used for late fusion
    predictions = predict_logits( loader, model, device, input_key=input_key)

    #ensure every video has a prediction
    assert len(predictions) == len(df)

    #make sure logit is real and finite
    assert np.isfinite(predictions["logit"].to_numpy()).all()

    #ensure predictions still correspond to the same videos
    input_files = set(df["filepath"])
    predicted_files = set(predictions["filepath"])
    assert input_files == predicted_files
    print(f"[OK] {mode} model produced {len(predictions)} logits.")

    #print predictions
    print(predictions[["filepath", "label", "logit"]].to_string(index=False))
    print(f"[PASS] {mode.upper()} pipeline")

    return predictions


def run_multimodal_test(df, device, output_dir, visual_weight):
    #Test: visual logits + audio logits -> late fusion -> saved results.
    print("\n===== TESTING MULTIMODAL PIPELINE =====")

    #start with clean test output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    #build pipelines
    visual_model = build_model(TEST_CFG, "visual", device)

    audio_model = build_model(TEST_CFG, "audio", device)

    #evaluate models
    metrics, contributions = evaluate_multimodal(
        test_df=df,
        visual_model=visual_model,
        audio_model=audio_model,
        visual_cfg=TEST_CFG,
        audio_cfg=TEST_CFG,
        device=device,
        fold_dir=str(output_dir),
        visual_weight=visual_weight)

    predictions_path = (output_dir / "fusion_predictions.csv")

    summary_path = (output_dir / "fusion_summary.json")

    #fusion logging should create both files
    assert predictions_path.exists()
    assert summary_path.exists()

    fused = pd.read_csv(predictions_path)

    #make sure only one fused prediction exists per video
    assert len(fused) == len(df)

    required_columns = {"logit_visual", "logit_audio", "visual_contribution", "audio_contribution", "fused_logit", "fused_probability", "y_pred"}
    missing = required_columns - set(fused.columns)

    assert not missing, (f"Missing fusion columns: {missing}")

    #verify  fusion equation:
    #fused_logit =visual_contribution + audio_contribution

    assert np.allclose(fused["fused_logit"], fused["visual_contribution"]+ fused["audio_contribution"])

    #probs must be valid sigmoid outputs
    assert ((fused["fused_probability"] >= 0.0) & (fused["fused_probability"] <= 1.0)).all()

    print("[OK] Visual and audio logits were aligned.")
    print("[OK] Fused logits equal summed contributions.")
    print("[OK] Fusion result files were created.")

    print("\nFusion results:")

    print(fused[["filepath",
                "logit_visual",
                "logit_audio",
                "visual_contribution",
                "audio_contribution",
                "fused_logit"]].to_string(index=False))

    print("\nContribution summary:")
    print(f"  visual:{contributions["mean_visual_abs_share"]:.3f}")
    print(f"  audio:{contributions["mean_audio_abs_share"]:.3f}")
    print(f"\n[PASS] MULTIMODAL pipeline (F1={metrics['f1']:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="test yawning detection pipelines")

    parser.add_argument("--mode",required=True,choices=[
            "visual",
            "audio",
            "multimodal",
            "all",],help="Pipeline to test")

    parser.add_argument("--data-dir", default="tests", help="Directory containing dummy videos")

    parser.add_argument("--visual-weight", type=float, default=0.5, help="Visual logit weight for multimodal fusion.")

    args = parser.parse_args()

    if not 0.0 <= args.visual_weight <= 1.0:
        raise ValueError("--visual-weight must be between 0 and 1.")

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {data_dir}")

    df = load_test_dataframe(data_dir)
    device = get_device()

    print(f"[INFO] Device: {device}")

    output_dir = (data_dir / "_test_output" / "multimodal")

    if args.mode in ("visual", "all"):
        run_unimodal_test( df, "visual", device)

    if args.mode in ("audio", "all"):
        run_unimodal_test( df, "audio", device)

    if args.mode in ("multimodal", "all"):
        run_multimodal_test( df, device, output_dir, args.visual_weight)

    print("\nAll requested tests passed.")


if __name__ == "__main__":
    main()