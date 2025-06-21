import numpy as np
import pickle
import os
import torch
from deepface import DeepFace
import cv2

def preprocess_vggface(img):
    img = img.astype(np.float32)
    mean = [91.4923, 103.8827, 131.0912] # B, G, R
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = img[..., ::-1]
    return img
MODEL_INPUT_SHAPE = (224, 224)
DEEPFACE_MODEL = DeepFace.build_model("VGG-Face")
MODEL_NAME = "VGG-Face"
def get_target_embedding(rep_path, target_id):
    if not os.path.exists(rep_path):
        raise FileNotFoundError(f"特征库文件未找到: {rep_path}")
    with open(rep_path, "rb") as f:
        representations = pickle.load(f)
    target_embeddings = []
    for rep in representations:
        identity = rep["identity"]
        folder = os.path.basename(os.path.dirname(identity))
        if folder == target_id:
            target_embeddings.append(rep["embedding"])

    if not target_embeddings:
        raise ValueError(f"在特征库中未找到目标ID: {target_id}")

    mean_embedding = np.mean(target_embeddings, axis=0)
    return torch.from_numpy(mean_embedding).float()

def get_embedding_from_tensor(image_tensor, device):
    image_tensor = (image_tensor + 1) / 2.0 * 255.0
    image_numpy_batch = image_tensor.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
    processed_batch = []
    for i in range(image_numpy_batch.shape[0]):
        img = image_numpy_batch[i]
        img_resized = cv2.resize(img, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
        img_processed = preprocess_vggface(img_resized)
        processed_batch.append(img_processed)
    processed_batch_np = np.array(processed_batch)
    embeddings = DEEPFACE_MODEL.model.predict(processed_batch_np, verbose=0)
    return torch.from_numpy(embeddings).float().to(device)