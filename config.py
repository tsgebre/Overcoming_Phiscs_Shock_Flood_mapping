class Config:
    PROJECT_NAME = "Flood_Physics_Guided_DL"
    MODE = "SOTA_Probabilistic_Attention"

    # Data Dimensions
    IMG_SIZE = 256
    CHANNELS_SAR = 2
    CHANNELS_OPT = 3
    CHANNELS_DEM = 3

    # Physics Parameters
    GRAVITY = 9.81
    MANNING_N = 0.03

    # Training Hyperparameters
    LR = 1e-3
    EPOCHS = 120
    BATCH_SIZE = 8

    # Model Capacity
    MODEL_WIDTH = 64
    MODEL_MODES = 16

    # Loss Weights & Strategy
    LAMBDA_DATA = 1.0
    LAMBDA_PHYS = 1e-4
    WARMUP_EPOCHS = 5
    LOSS_REDUCTION = 'mean'

    # Normalization
    PIXEL_SIZE = 10.0
    NORMALIZE_GLOBAL = False

    # Checkpointing
    RESUME = False

    # Added for main.py / Trainer enhancements
    OUT_DIR = "./checkpoints"
    USE_TENSORBOARD = True
