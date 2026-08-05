import os
import cv2
import pandas as pd
import numpy as np
import subprocess   
import tempfile
from tqdm import tqdm
from src.data import get_all_data_paths


"""
This scripts augments all of the video in the data folder, conserving audio tracks, if there are any.
Note: This is only used once to double the size of the samples. Running it again as is, would then double the size again, but
also re-augment the already augmented videos. Doubling IDs, as they are in the original yawdd, are augmented identically and they 
also get the same incremented id. This leads to eg. 001-male and 001-female always being in either train, test or validation set
together. While not perfect, the effects while shuffling are considered to be neglible, as long as they are always only ever 
in train, test or val
"""

def augment_video(input_path, output_path):

    #temp file to hold audio
    #ext = os.path.splitext(input_path)[1]
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    #Load Video
    cap = cv2.VideoCapture(input_path)

    #Preserve format
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    #Sample augment params once per video for temporal consistency
    flip = np.random.rand() < 0.5
    alpha = 0.9 + 0.2 *np.random.rand() #rnd contrast in rage [0.9, 1.1]
    beta = np.random.randint(-20,20) #rnd brightness 
    #rotation
    angle = np.random.uniform(-5,5)
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    

    #noise and blur
    noise_strength = np.random.uniform(0, 10)
    blur_prob = np.random.rand() < 0.3

    #Process Video frame by frame
    while True:
        ret, frame = cap.read() #returns bool for ret, matlike for frame

        if not ret: #stop when video ends
            break

        #Flip
        if flip:
            frame = cv2.flip(frame,1)

        #Rotation
        frame = cv2.warpAffine(frame, rot_mat, (width, height), borderMode=cv2.BORDER_REFLECT)

        #Brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        #Noise
        noise = np.random.normal(0, noise_strength, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        #Blur
        if blur_prob:
            frame = cv2.GaussianBlur(frame, (3,3),0)

        #write frame to output video
        out.write(frame)

    #Release ressources
    cap.release()
    out.release()

    #Merge aug vid and audio back together
    command = [
        "ffmpeg",       
        "-y",       #overwrite iutput
        "-i",temp_video,    #aug vid (no audio)
        "-i",input_path,    #og vid (has audio)
        "-c:v", "copy",     #copy vid
        "-c:a", "aac",      #encode audio
        "-map", "0:v:0",    #take vid from temp
        "-map", "1:a:0?",    #take audio from og vid
        output_path   
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #clean up temp file
    os.remove(temp_video)

def create_augmented_dataset(input_root="data", output_root="data_augmented"):
    #create output_dir if not already done
    os.makedirs(output_root, exist_ok=True)

    #Load dataset
    df = get_all_data_paths(input_root)

    #determine ID range
    unique_ids = sorted(df["id"].unique())
    max_id = max(int(i) for i in unique_ids)
    next_id = max_id +1

    print(F"Starting augment from ID: {next_id}")

    #group by id for augment
    grouped = df.groupby("id")

    for old_id, group in tqdm(grouped, desc="Augmenting subjects"):
        #assign new id to augmented subject
        new_id = f"{next_id:03d}" #pad with leading zeros, min 3 digits long
        next_id += 1

        #process all vids belonging to this id
        for _, row in group.iterrows():
            old_path = row["filepath"]

            #extract name components
            filename = os.path.splitext(os.path.basename(old_path))[0]
            parts = filename.split("-")

            _, info, activity = parts

            #construct new file name
            new_name = f"{new_id}-{info}-{activity}.mp4"
            new_path = os.path.join(output_root, new_name)

            #apply augment and save
            augment_video(old_path, new_path)

    print("Augment complete")

if __name__ == "__main__":
    create_augmented_dataset()