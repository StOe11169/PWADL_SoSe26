import os
import torch
from datetime import datetime
import argparse, time
from src.utils import setup_env, start_tensorboard, get_device
from src.data import  get_all_data_paths
from src.experiment import run_experiment

if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    #set seed and precision before doing anything else
    setup_env(seed=0)    
    device = get_device()

    # get client args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=10)

    #testing audio/fusion pipeline
    parser.add_argument("--mode", type=str, default="visual", choices=["visual", "audio", "multimodal"])
    parser.add_argument("--audio_exclude_path_parts", nargs="*", default=["Mirror"]) #exclude folders without audio
    parser.add_argument("--visual_weight", type=float, default=0.5, help="Visual logit weight for late fusion.")

    args = parser.parse_args()

    #Load dataset
    df = get_all_data_paths("data")
    #Fail early if any file cant be accessed at multiple positions
    #validate_video_decoding(df, num_frames=2)

    #Create unique folder for study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"study_{args.mode}_{timestamp}"
    study_dir = os.path.join("logs", study_name) #replace vision with audio or multimodal later
    os.makedirs(study_dir, exist_ok=True)

    tb_process = start_tensorboard(study_dir)

    try:
        run_experiment(df, args, study_dir)
    except Exception:
        print("[ERROR] Training failed.")
        raise

    else:
        print("Training completed successfully.")

    finally:
        tb_process.terminate() #always terminate tensorboard
        time_passed = time.time() - start_timestamp
        print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')

    
    