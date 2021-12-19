import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import itertools
import sys

from datasets import lits17_no_slice, brats20_no_slice, kits21_no_slice
from losses import MultiTverskyLoss
from UNet3D import Encoder, Decoder
from scipy.stats import entropy
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from evaluations import eval_uncertainty, eval_dice


def forward(encoder, decoder_mean, decoder_var, x):
    out, context_1, context_2, context_3, context_4 = encoder(x)
    mean = decoder_mean(out, context_1, context_2, context_3, context_4)
    var = decoder_var(out, context_1, context_2, context_3, context_4)
    return mean, var


def predict(encoder, decoder_mean, decoder_var, x, T=10):
    mean, var = forward(encoder, decoder_mean, decoder_var, x)
    running_x = torch.zeros(var.shape).to(x.device)
    noise = torch.randn(var.shape).to(x.device)
    for i in range(T):
        x = mean + var * noise
        running_x += F.softmax(x, dim=1)
    return running_x / T


# returns aleatoric uncertainty map, yhat
def aleatoric_uncertainty_density_model(encoder, decoder_mean, decoder_var, x, T=10):
    encoder = encoder.eval()
    decoder_mean = decoder_mean.eval()
    decoder_var = decoder_var.eval()
    
    with torch.no_grad():
        yhat = predict(encoder, decoder_mean, decoder_var, x, T=T)
    return entropy(yhat.cpu().numpy()[0], axis=0), yhat[0, 1].cpu().numpy()


# returns epistemic uncertainty map, yhat
def epistemic_uncertainty_mc_dropout(encoder, decoder_mean, decoder_var, x, K=10):
    encoder = encoder.eval()
    decoder_mean = decoder_mean.eval()
    decoder_var = decoder_var.eval()
    enable_dropout(encoder)
    enable_dropout(decoder_mean)

    yhat = []
    for k in range(K):
        with torch.no_grad():
            yhat_mean, yhat_var = forward(encoder, decoder_mean, decoder_var, x)
            yhat_mean = F.softmax(yhat_mean, dim=1).cpu().numpy()
            yhat.append(yhat_mean)
    yhat = np.stack(yhat).mean(axis=0)
    epis = entropy(yhat, axis=1)
    return epis[0], yhat[0, 1]


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


ROOT = Path('/scratch/zc2357/cv/final/nyu-cv2271-final/baseline/runs/')

tasks = [
    {
        'name': 'lits17_baseline',
        'dataset': lits17_no_slice,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec12_04-00-51_gr017.nyu.cluster_lits17_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'lits17_epoch_55_step_5824_encoder.pth',
        'decoder_mean_path': 'lits17_epoch_55_epoch_55_loss_0.04442_decoder_mean.pth',
        'decoder_var_path': 'lits17_epoch_55_epoch_55_loss_0.04442_decoder_var.pth',
    },
    {
        'name': 'lits17_cotraining',
        'dataset': lits17_no_slice,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec11_19-20-25_gr011.nyu.cluster_lits17_brats20_kits21_cotraining_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'lits17_epoch_59_step_6241_encoder.pth',
        'decoder_mean_path': 'lits17_epoch_59_epoch_59_loss_0.04083_decoder_mean.pth',
        'decoder_var_path': 'lits17_epoch_59_epoch_59_loss_0.04083_decoder_var.pth',
    },
    {
        'name': 'brats20_baseline',
        'dataset': brats20_no_slice,
        'in_channels': 4,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec11_14-53-08_gr011.nyu.cluster_brats20_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'brats20_epoch_21_step_6490_encoder.pth',
        'decoder_mean_path': 'brats20_epoch_21_epoch_21_loss_0.09565_decoder_mean.pth',
        'decoder_var_path': 'brats20_epoch_21_epoch_21_loss_0.09565_decoder_var.pth',
    },
    {
        'name': 'brats20_cotraining',
        'dataset': brats20_no_slice,
        'in_channels': 4,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec11_19-20-25_gr011.nyu.cluster_lits17_brats20_kits21_cotraining_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'brats20_epoch_14_step_4596_encoder.pth',
        'decoder_mean_path': 'brats20_epoch_14_epoch_14_loss_0.09117_decoder_mean.pth',
        'decoder_var_path': 'brats20_epoch_14_epoch_14_loss_0.09117_decoder_var.pth',
    },
    {
        'name': 'kits21_baseline',
        'dataset': kits21_no_slice,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec11_05-24-43_gr038.nyu.cluster_kits21_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'kits21_epoch_25_step_6240_encoder.pth',
        'decoder_mean_path': 'kits21_epoch_25_epoch_25_loss_0.05592_decoder_mean.pth',
        'decoder_var_path': 'kits21_epoch_25_epoch_25_loss_0.05592_decoder_var.pth',
    },
    {
        'name': 'kits21_cotraining',
        'dataset': kits21_no_slice,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'root': ROOT / 'Dec11_19-20-25_gr011.nyu.cluster_lits17_brats20_kits21_cotraining_baseline_lr1.0e-04_weightDecay1.0e-02',
        'encoder_path': 'kits21_epoch_25_step_6241_encoder.pth',
        'decoder_mean_path': 'kits21_epoch_25_epoch_25_loss_0.05474_decoder_mean.pth',
        'decoder_var_path': 'kits21_epoch_25_epoch_25_loss_0.05474_decoder_var.pth',
    },
]

# DO NOT CHANGE; CHANGING THESE BREAKS REPLICATION
SEED = 42
TRAIN_VAL_SPLIT = 0.8  # 80% training
# /DO NOT CHANGE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(len(tasks)):
    task = tasks[i]
    if task['enabled']:
        n = len(task['dataset'])
        idxrange = np.arange(n)
        rng = np.random.RandomState(SEED)
        rng.shuffle(idxrange)
        n_train = int(n * TRAIN_VAL_SPLIT)
        task['train_idx'] = idxrange[:n_train]
        task['val_idx'] = idxrange[n_train:]

        task['val_dataloader'] = DataLoader(
            task['dataset'],
            batch_size=1,
            sampler=task['val_idx'],
            num_workers=1,
        )
        task['decoder_mean'] = Decoder(
            task['in_channels'],
            task['n_classes'],
        ).to(device)
        task['decoder_var'] = Decoder(
            task['in_channels'],
            task['n_classes'],
        ).to(device)
        task['encoder'] = Encoder(task['in_channels'], base_n_filter=8, drop_p=0.6).to(device)
        
        task['decoder_mean'].load_state_dict(torch.load(task['root'] / task['decoder_mean_path'], map_location=device))
        task['decoder_var'].load_state_dict(torch.load(task['root'] / task['decoder_var_path'], map_location=device))
        task['encoder'].load_state_dict(torch.load(task['root'] / task['encoder_path'], map_location=device))

SAVEROOT = Path('/scratch/zc2357/cv/final/nyu-cv2271-final/baseline/inference_alea_yhat')
if not SAVEROOT.exists():
    SAVEROOT.mkdir()

for task in tasks:
    if task['enabled']:
        savepath = SAVEROOT / task['name']
        if not savepath.exists():
            savepath.mkdir()
        
        task['encoder'].eval()
        task['decoder_mean'].eval()
        task['decoder_var'].eval()

        with torch.no_grad():
            for i, (X, y) in enumerate(task['val_dataloader']):
                print(task['name'], i, '/', len(task['val_dataloader']))
                X, y = X.to(device), y.to(device)
                alea, yhat_alea = aleatoric_uncertainty_density_model(
                    task['encoder'],
                    task['decoder_mean'],
                    task['decoder_var'],
                    X,
                    T=50,
                )
                epis, yhat_epis = epistemic_uncertainty_mc_dropout(
                    task['encoder'],
                    task['decoder_mean'],
                    task['decoder_var'],
                    X,
                    K=50,
                )
                np.save(savepath / ('alea_%03d.npy' % i), alea)
                np.save(savepath / ('epis_%03d.npy' % i), epis)
                np.save(savepath / ('yhat_alea_%03d.npy' % i), yhat_alea)
                np.save(savepath / ('yhat_epis_%03d.npy' % i), yhat_epis)
                print(alea.shape, yhat_alea.shape)
                sys.stdout.flush()
