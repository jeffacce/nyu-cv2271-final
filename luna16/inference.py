import pandas as pd
import numpy as np
import torch
import torch.nn
from models import Classifier3D, TransferClassifierHead
from UNet3D import Encoder
from datasets import LUNA16DatasetFromIso
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path
import torch.nn.functional as F


# linearly transform [-1000, 400] to [0, 1]
def linear_transform_to_0_1(X, min=-1000, max=400):
    result = torch.clamp(X, min=min, max=max)
    result = result - min
    result = result / (max - min)
    return result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

baseline_path = '/scratch/zc2357/cv/final/nyu-cv2271-final/luna16/runs/Dec10_23-51-34_gr056.nyu.cluster_baseline_UNet3D3x3_randomFlipsPosNeg_focalLossAlpha0.5_weightDecay1e-2_lr1e-4_AdamW/epoch_5_2431.pth'
encoder_path = '/scratch/zc2357/cv/final/nyu-cv2271-final/luna16/runs/Dec12_21-09-20_gr009.nyu.cluster_transfer_randomFlipsPosNeg_focalLossAlpha0.5_weightDecay1e-2_lr1e-4_AdamW/epoch_5_2431_encoder.pth'
classifier_head_path = '/scratch/zc2357/cv/final/nyu-cv2271-final/luna16/runs/Dec12_21-09-20_gr009.nyu.cluster_transfer_randomFlipsPosNeg_focalLossAlpha0.5_weightDecay1e-2_lr1e-4_AdamW/epoch_5_2431_classifier.pth'

test_subsets = [
    'subset8',
    'subset9',
]

test_dataset = LUNA16DatasetFromIso(
    iso_root_path='/scratch/zc2357/cv/final/datasets/luna16_iso/',
    candidates_file='candidates_V2.csv',
    subsets=test_subsets,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # uses cached arrays to save disk hits
    num_workers=1,
)


baseline = Classifier3D(in_channels=1, img_size=48).to(device)
baseline.load_state_dict(torch.load(baseline_path, map_location=device))
baseline.eval()

encoder = Encoder(in_channels=1, base_n_filter=8).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.eval()

classifier_head = TransferClassifierHead(base_n_filter=8).to(device)
classifier_head.load_state_dict(torch.load(classifier_head_path, map_location=device))
classifier_head.eval()


ys = []
yhat_baselines = []
yhat_transfers = []

for X, y in tqdm(test_dataloader):
    X = X.reshape(-1, 1, 48, 48, 48)
    X = linear_transform_to_0_1(X, min=-1000, max=400)
    X = X.to(device)
    with torch.no_grad():
        yhat_baseline = baseline(X)
    with torch.no_grad():
        out, _, _, _, _ = encoder(X)
        yhat_transfer = classifier_head(out)
    ys.append(y)
    yhat_baselines.append(yhat_baseline.detach().cpu())
    yhat_transfers.append(yhat_transfer.detach().cpu())

ys = torch.cat(ys)
yhat_baselines = torch.cat(yhat_baselines)
yhat_transfers = torch.cat(yhat_transfers)

ys = ys.cpu().numpy()
yhat_baselines = yhat_baselines.cpu().numpy().flatten()
yhat_transfers = yhat_transfers.cpu().numpy().flatten()

df = pd.DataFrame({
    'y': ys,
    'baseline': yhat_baselines,
    'transfer': yhat_transfers,
})

df.to_csv('test_inference.csv', index=False)
