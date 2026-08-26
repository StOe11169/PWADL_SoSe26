# src/config.py
class Config:
    # -----------------
    # Dataset
    # -----------------
    DATA_FOLDER = "data"
    NUM_FRAMES = 64
    RANDOM_STATE = 42
    # -----------------
    # Splitting
    # -----------------
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    # -----------------
    # Training
    # -----------------
    EPOCHS = 32
    BATCH_SIZE = 4
    LEARNING_RATE = 1.02997397467e-4
    WEIGHT_DECAY = 1e-2
    FREEZE_BACKBONE = False
    # -----------------
    # Model
    # -----------------
    DROPOUT = 0.2
    ATTENTION_HIDDEN = 128
    # -----------------
    # Loss
    # -----------------
    POS_WEIGHT = 2.0
    # -----------------
    # Gradient Clipping
    # -----------------
    GRAD_CLIP_NORM = 1.0
    # -----------------
    # DataLoader
    # -----------------
    NUM_WORKERS = 0
    # -----------------
    # Optuna
    # -----------------
    N_TRIALS = 1