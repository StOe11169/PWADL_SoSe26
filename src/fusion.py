import json
import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

def fuse_logits(visual_predictions, audio_predictions, visual_weight=0.5):
    #combining logits of both pipelines
    #using weights for ease of use instead of a neural network for now

    audio_weight = 1.0 - visual_weight

    #match predictions per video
    fused = pd.merge(visual_predictions, audio_predictions, on="filepath", suffixes=("_visual", "_audio"), validate="one_to_one") #one_to_one avoids duplicate predictions for same video

    #ensure labels describe the same video, after multimod alignment
    if not np.array_equal(fused["label_visual"].to_numpy(), fused["label_audio"].to_numpy()):
        raise ValueError("visual and audo labels dont match")

    fused["label"] = fused["label_visual"]

    #show how much each mode add to the final logit
    fused["visual_contribution"] = (visual_weight * fused["logit_visual"])
    fused["audio_contribution"] = (audio_weight * fused["logit_audio"])

    #fuse logits through addition
    fused["fused_logit"] = (fused["visual_contribution"] + fused["audio_contribution"])

    #convert fused logits to prob with sigmoid
    #Note: raw logits of each mode may have different scales -> ToDo: research logit calibration or using probabilities instead.
    fused_logits = torch.tensor(fused["fused_logit"].to_numpy(), dtype=torch.float32)
    fused["fused_probability"] = (torch.sigmoid(fused_logits).numpy())

    # sigmoid(logit) > 0.5 is equal to logit > 0
    fused["y_pred"] = (fused["fused_logit"] > 0.0).astype(int)

    #which mode contributed more?
    total_magnitude = (fused["audio_contribution"].abs() +fused["visual_contribution"].abs())
    #avoiding dividing by zero
    total_magnitude = total_magnitude.replace(0, np.nan)
    fused["visual_abs_share"] = (fused["visual_contribution"].abs() / total_magnitude).fillna(0.5)
    fused["audio_abs_share"] = (fused["audio_contribution"].abs() / total_magnitude).fillna(0.5)

    return fused

def get_fusion_metrics(fused):
    #metrics for late fusion, similar to evaluate()

    y_true = fused["label"].to_numpy()
    y_pred = fused["y_pred"].to_numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score( y_true, y_pred, zero_division=0),
        "f1": f1_score( y_true, y_pred, zero_division=0)}

def get_contribution_summary(fused,visual_weight,):
    #note: bsolute share measures magnitude, not causal importance
    # + -> more confident its yawning, - -> more confident its not-yawning

    audio_weight = 1.0 - visual_weight

    return {"visual_weight": visual_weight, "audio_weight": audio_weight,
        #mean shows average in-/decrease in confidence of decision
        "mean_visual_contribution": float(fused["visual_contribution"].mean()),
        "mean_audio_contribution": float(fused["audio_contribution"].mean()),
        #abs-values show average contribution strength
        "mean_abs_visual_contribution": float(fused["visual_contribution"].abs().mean()),
        "mean_abs_audio_contribution": float(fused["audio_contribution"].abs().mean()),
        #relative magnitude
        "mean_visual_abs_share": float(fused["visual_abs_share"].mean()), "mean_audio_abs_share": float(fused["audio_abs_share"].mean())}

def save_fusion_results(fused, metrics, contribution_summary, fold_dir,):
    #result for every video, saves per video metrics seperately from summarised metrics
    fused.to_csv(os.path.join(fold_dir, "fusion_predictions.csv"), index=False)
    summary = {"metrics": metrics, "contributions": contribution_summary}
    with open(os.path.join(fold_dir, "fusion_summary.json"), "w") as f: json.dump(summary, f, indent=4)

