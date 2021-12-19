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
        'enabled': False,
    },
    {
        'name': 'brats20',
        'dataset': brats20,
        'in_channels': 4,
        'n_classes': 2,
        'enabled': True,
    },
    {
        'name': 'kits21',
        'dataset': kits21,
        'in_channels': 1,
        'n_classes': 2,
        'enabled': False,
    }
]


SEED = 42
TRAIN_VAL_SPLIT = 0.8  # 80% training
COMMENT = 'brats20_baseline'
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_STEPS = 10000
ADVERSARIAL_TRAINING = True
EPSILON_FGSM = 1e-2  # for adversarial training


# initialize training tasks
final_comment = (
    '_' + COMMENT +
    f'_lr{LR:.1e}_weightDecay{WEIGHT_DECAY:.1e}'
)
if ADVERSARIAL_TRAINING:
    final_comment += f'_fgsm{EPSILON_FGSM:.1e}'
writer = SummaryWriter(comment=final_comment)
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
        ).to(device)
        task['decoder_var'] = Decoder(
            task['in_channels'],
            task['n_classes'],
        ).to(device)
        task['decoder_optim'] = optim.AdamW(
            list(task['decoder_mean'].parameters()) + list(task['decoder_var'].parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        task['epoch'] = 0

encoder = Encoder(task['in_channels']).to(device)
encoder_optim = optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_f = MultiTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)


for step in range(MAX_STEPS):
    print(f'==== Step {step} ====')
    encoder_optim.zero_grad()
    
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
            
            task['decoder_optim'].zero_grad()
            X, y = X.to(device), y.to(device)
            if ADVERSARIAL_TRAINING:
                X.requires_grad = True
            
            yhat = predict(encoder, task['decoder_mean'], task['decoder_var'], X, T=10)
            loss = loss_f(yhat, y)
            loss.backward()
            train_loss = loss.item()
            task['decoder_optim'].step()

            if ADVERSARIAL_TRAINING:
                sign_X_grad = X.grad.data.detach().sign()
                X_perturbed = X + EPSILON_FGSM * sign_X_grad
                X_perturbed = torch.clamp(X_perturbed, 0, 1).detach()
                yhat = predict(encoder, task['decoder_mean'], task['decoder_var'], X_perturbed, T=10)
                task['decoder_optim'].zero_grad()
                loss = loss_f(yhat, y)
                loss.backward()
                train_loss = (train_loss + loss.item()) / 2
                task['decoder_optim'].step()

            print(f"    Train loss: {train_loss:.5f}")
            writer.add_scalar(f"{task['name']}/loss/train", loss.item(), global_step=step)
    encoder_optim.step()
