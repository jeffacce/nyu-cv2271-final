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

from datasets import lits17, brats20, kits21
from losses import MultiTverskyLoss
from UNet3D import Encoder, Decoder


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


tasks = [
    {
        'name': 'lits17',
        'dataset': lits17,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'resume_decoder_mean_weight': 'lits17_epoch_29_epoch_29_loss_0.04831_decoder_mean.pth',
        'resume_decoder_var_weight': 'lits17_epoch_29_epoch_29_loss_0.04831_decoder_var.pth',
        'resume_decoder_optim_state': 'lits17_epoch_29_epoch_29_loss_0.04831_decoder_optim.pth',
        'resume_epoch': 30,
    },
    {
        'name': 'brats20',
        'dataset': brats20,
        'in_channels': 4,
        'n_classes': 2,
        'enabled': True,
        'resume_decoder_mean_weight': 'brats20_epoch_9_epoch_9_loss_0.13591_decoder_mean.pth',
        'resume_decoder_var_weight': 'brats20_epoch_9_epoch_9_loss_0.13591_decoder_var.pth',
        'resume_decoder_optim_state': 'brats20_epoch_9_epoch_9_loss_0.13591_decoder_optim.pth',
        'resume_epoch': 10,
    },
    {
        'name': 'kits21',
        'dataset': kits21,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': True,
        'resume_decoder_mean_weight': 'kits21_epoch_12_epoch_12_loss_0.07526_decoder_mean.pth',
        'resume_decoder_var_weight': 'kits21_epoch_12_epoch_12_loss_0.07526_decoder_var.pth',
        'resume_decoder_optim_state': 'kits21_epoch_12_epoch_12_loss_0.07526_decoder_optim.pth',
        'resume_epoch': 13,
    }
]


SEED = 42
TRAIN_VAL_SPLIT = 0.8  # 80% training
COMMENT = 'lits17_brats20_kits21_cotraining_baseline_encoder_decoder_both_update'
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_STEPS = 10000
DROPOUT_P = 0.6
RESUME = True
resume_path = Path('/scratch/zc2357/cv/final/nyu-cv2271-final/baseline/runs/Dec11_19-20-25_gr011.nyu.cluster_lits17_brats20_kits21_cotraining_baseline_lr1.0e-04_weightDecay1.0e-02')
resume_encoder_weight = 'kits21_epoch_12_step_3120_encoder.pth'
resume_encoder_optim_state = 'kits21_epoch_12_step_3120_encoder_optim.pth'
resume_step = 3121

# initialize training tasks

if RESUME:
    writer = SummaryWriter(log_dir=resume_path.as_posix(), purge_step=resume_step)
else:
    writer = SummaryWriter(comment='_' + COMMENT + f'_lr{LR:.1e}_weightDecay{WEIGHT_DECAY:.1e}_dropout{DROPOUT_P:.2f}')
logdir = Path(writer.log_dir)
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

        task['train_dataloader'] = DataLoader(
            task['dataset'],
            batch_size=1,
            sampler=SubsetRandomSampler(task['train_idx']),
            num_workers=1,
        )
        task['train_dataiter'] = iter(task['train_dataloader'])
        task['val_dataloader'] = DataLoader(
            task['dataset'],
            batch_size=1,
            sampler=SubsetRandomSampler(task['val_idx']),
            num_workers=1,
        )

        task['decoder_mean'] = Decoder(
            task['in_channels'],
            task['n_classes'],
            drop_p=DROPOUT_P,
        ).to(device)
        task['decoder_var'] = Decoder(
            task['in_channels'],
            task['n_classes'],
            drop_p=DROPOUT_P,
        ).to(device)
        task['decoder_optim'] = optim.AdamW(
            list(task['decoder_mean'].parameters()) + list(task['decoder_var'].parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        task['epoch'] = 0
        if RESUME:
            task['decoder_mean'].load_state_dict(torch.load(resume_path / task['resume_decoder_mean_weight']))
            task['decoder_var'].load_state_dict(torch.load(resume_path / task['resume_decoder_var_weight']))
            task['decoder_optim'].load_state_dict(torch.load(resume_path / task['resume_decoder_optim_state']))
            task['epoch'] = task['resume_epoch']

encoder = Encoder(4, drop_p=DROPOUT_P).to(device)
encoder_optim = optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
if RESUME:
    encoder.load_state_dict(torch.load(resume_path / resume_encoder_weight))
    encoder_optim.load_state_dict(torch.load(resume_path / resume_encoder_optim_state))
    step = resume_step
loss_f = MultiTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)

if RESUME:
    start_step = resume_step
else:
    start_step = 0

for step in range(start_step, MAX_STEPS):
    encoder_optim.zero_grad()
    print(f'==== Step {step} ====')
    
    for task in tasks:
        if task['enabled']:
            print(f"  Task: {task['name']}")
            try:
                X, y = next(task['train_dataiter'])
            except StopIteration:
                # ==== VALIDATION ====
                print(f"    Validation")
                loss_mean = 0
                encoder.eval()
                task['decoder_mean'].eval()
                task['decoder_var'].eval()
                
                with torch.no_grad():
                    for X, y in task['val_dataloader']:
                        X, y = X.to(device), y.to(device)
                        yhat = predict(encoder, task['decoder_mean'], task['decoder_var'], X, T=10)
                        loss = loss_f(yhat, y)
                        loss_mean += loss.item()
                    loss_mean /= len(task['val_dataloader'])
                
                encoder.train()
                task['decoder_mean'].train()
                task['decoder_var'].train()
                
                # checkpoint and reset train dataiter
                print(f"    Validation loss: {loss_mean:.5f}")
                print(f"    Saving models.")
                writer.add_scalar(f"{task['name']}/loss/val", loss_mean, global_step=step)
                
                saveroot = Path(writer.get_logdir())
                savename = f"{task['name']}_epoch_{task['epoch']}"
                
                encoder_savepath = (saveroot / f"{savename}_step_{step}_encoder.pth").as_posix()
                encoder_optim_savepath = (saveroot / f"{savename}_step_{step}_encoder_optim.pth").as_posix()
                torch.save(encoder.state_dict(), encoder_savepath)
                torch.save(encoder_optim.state_dict(), encoder_optim_savepath)
                
                decoder_mean_savepath = (saveroot / f"{savename}_epoch_{task['epoch']}_loss_{loss_mean:.5f}_decoder_mean.pth").as_posix()
                decoder_var_savepath = (saveroot / f"{savename}_epoch_{task['epoch']}_loss_{loss_mean:.5f}_decoder_var.pth").as_posix()
                decoder_optim_savepath = (saveroot / f"{savename}_epoch_{task['epoch']}_loss_{loss_mean:.5f}_decoder_optim.pth").as_posix()
                torch.save(task['decoder_mean'].state_dict(), decoder_mean_savepath)
                torch.save(task['decoder_var'].state_dict(), decoder_var_savepath)
                torch.save(task['decoder_optim'].state_dict(), decoder_optim_savepath)
                # ==== VALIDATION ====
                
                task['epoch'] += 1
                task['train_dataiter'] = iter(task['train_dataloader'])
                X, y = next(task['train_dataiter'])
            
            # encoder_optim.zero_grad()
            task['decoder_optim'].zero_grad()
            X, y = X.to(device), y.to(device)
            
            yhat = predict(encoder, task['decoder_mean'], task['decoder_var'], X, T=10)
            loss = loss_f(yhat, y)
            loss.backward()
            # encoder_optim.step()
            task['decoder_optim'].step()
            
            print(f"    Train loss: {loss.item():.5f}")
            writer.add_scalar(f"{task['name']}/loss/train", loss.item(), global_step=step)
    encoder_optim.step()
