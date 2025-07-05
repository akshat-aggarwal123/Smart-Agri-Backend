import numpy as np

# ACTUAL EMBEDDINGS FROM TRAINING (replace with your real embeddings)
CROP_EMBEDDINGS = {
    "rice": np.array([...]),   # 128-dim vector
    "wheat": np.array([...]),  # 128-dim vector
    "corn": np.array([...]),   # 128-dim vector
    # Add all crops used in training
}

def get_crop_embeddings():
    return CROP_EMBEDDINGS