import os
import torch
import sys
import traceback
from datetime import datetime
import argparse, time
from src.utils import setup_env, start_tensorboard, get_device
from src.data import  get_all_data_paths
from src.experiment import run_experiment

class Tee:
    """Write output to both the terminal and a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    #set seed and precision before doing anything else
    setup_env(seed=0)    
    #device = get_device()

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
    #df = get_all_data_paths("data")
    #Fail early if any file cant be accessed at multiple positions
    #validate_video_decoding(df, num_frames=2)

    #Create unique folder for study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"study_{args.mode}_{timestamp}"
    study_dir = os.path.join("logs", study_name) #replace vision with audio or multimodal later

    console_path = os.path.join(study_dir, "console.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    tb_process = None
    exit_code = 0

    with open(console_path, "w", encoding="utf-8", buffering=1,) as console_file:

        sys.stdout = Tee( original_stdout, console_file)

        # tqdm, warnings and tracebacks normally use stderr.
        sys.stderr = Tee( original_stderr, console_file)

        try:
            # These checks are now included in console.log.
            get_device()
            df = get_all_data_paths("data")

            tb_process = start_tensorboard(study_dir)

            run_experiment(df, args, study_dir)

        except Exception:
            exit_code = 1

            print("[ERROR] Training failed.", file=sys.stderr )

            # Save the complete traceback before restoring stderr.
            traceback.print_exc()

        else:
            print("Training completed successfully.")

        finally:
            if tb_process is not None:
                tb_process.terminate()

            time_passed = (time.time() - start_timestamp)

            print(
                f"\nTraining finished in "
                f"{time_passed // 3600}h "
                f"{(time_passed % 3600) // 60}min "
                f"{time_passed % 60:.0f}s\n")

            # Restore normal console streams before closing the file.
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    if exit_code:
        raise SystemExit(exit_code)

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

    
    