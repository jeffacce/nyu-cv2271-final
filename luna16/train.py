import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import LUNA16DatasetFromIso, LUNA16DatasetFromCubes
from models import Classifier3D, TinyClassifier
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from losses import binary_focal_loss_with_logits
import itertools


# linearly transform [-1000, 400] to [0, 1]
def linear_transform_to_0_1(X, min=-1000, max=400):
    result = torch.clamp(X, min=min, max=max)
    result = result - min
    result = result / (max - min)
    return result


def calc_confusion(pred_bool, target_bool):
    tp = (pred_bool & target_bool).sum().item()
    tn = ((~pred_bool) & (~target_bool)).sum().item()
    fp = (pred_bool & (~target_bool)).sum().item()
    fn = ((~pred_bool) & target_bool).sum().item()
    return tp, tn, fp, fn


def shuffle_wrapper(x):
    np.random.shuffle(x)
    return x

###################### CONFIG ########################
BATCH_SIZE = 64
# batch size must be even since we sample half positives, half negatives
assert BATCH_SIZE % 2 == 0

# one epoch is defined as one pass through all negative samples;
# positive samples are reused
EPOCHS = 50
MOMENTUM = 0.9
LR = 1e-4
WEIGHT_DECAY = 1e-5
LOG_INTERVAL = 100
SHOULD_AUGMENT = True
RESUME = True
CHECKPOINT = {
    'logdir': 'runs/Dec05_02-17-28_gr032.nyu.cluster_baseline_UNet3D3x3_randomFlipsPosNeg_focalLossAlpha0.5_weightDecay1e-5_lr1e-4_AdamW',
    'optimizer_state': 'epoch_8_optimizer.pth',
    'model_state': 'epoch_8.pth',
    'resume_epoch': 9,
    'resume_pos_epoch': 3888,
}
RUN_COMMENT = '_baseline_UNet3D3x3_randomFlipsPosNeg_focalLossAlpha0.5_weightDecay1e-5_lr1e-4_AdamW_run2'
###################### /CONFIG ########################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Classifier3D(in_channels=1, img_size=48).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
if RESUME:
    logdir = Path(CHECKPOINT['logdir'])
    model.load_state_dict(torch.load(logdir / CHECKPOINT['model_state']))
    optimizer.load_state_dict(torch.load(logdir / CHECKPOINT['optimizer_state']))
    start_epoch = CHECKPOINT['resume_epoch']
    start_pos_epoch = CHECKPOINT['resume_pos_epoch']
    writer = SummaryWriter(log_dir=CHECKPOINT['logdir'], purge_step=CHECKPOINT['resume_pos_epoch'])  # should be global step, but val epoch always < pos epoch, doesn't matter
else:
    start_epoch = 1
    start_pos_epoch = 1
    writer = SummaryWriter(comment=RUN_COMMENT)


train_subsets = [
    'subset0',
    'subset1',
    'subset2',
    'subset3',
    'subset4',
    'subset5',
]
val_subsets = [
    'subset6',
    'subset7',
]
test_subsets = [
    'subset8',
    'subset9',
]

train_neg_dataset = LUNA16DatasetFromIso(
    iso_root_path='/scratch/zc2357/cv/final/datasets/luna16_iso/',
    candidates_file='candidates_V2.csv',
    subsets=train_subsets,
)
train_pos_dataset = LUNA16DatasetFromCubes(
    cube_root_path='/scratch/zc2357/cv/final/datasets/luna16_cubes',
    candidates_file='candidates_V2_subindexed.csv',
    subsets=train_subsets,
)
val_dataset = LUNA16DatasetFromIso(
    iso_root_path='/scratch/zc2357/cv/final/datasets/luna16_iso/',
    candidates_file='candidates_V2.csv',
    subsets=val_subsets,
)

train_pos_dataloader = DataLoader(
    train_pos_dataset,
    batch_size=BATCH_SIZE//2,
    sampler=SubsetRandomSampler(train_pos_dataset.pos_sample_idx),
    num_workers=1,
    drop_last=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # validation set uses cached arrays to save disk hits
    num_workers=1,
)




# =========== TRAINING ===========
pos_epoch = start_pos_epoch  # how many times we've gone through the positive samples
for epoch in range(start_epoch, EPOCHS+1):
    # reshuffle negative dataloader every epoch
    neg_idx_shuffled = (
        train_neg_dataset.candidates.loc[train_neg_dataset.neg_sample_idx]
        .copy().reset_index()
        .groupby('seriesuid')['index'].unique()
        .apply(shuffle_wrapper)  # shuffle within cases
        .apply(list)
    )
    neg_idx_shuffled = neg_idx_shuffled.sample(len(neg_idx_shuffled))      # shuffle case order
    neg_idx_shuffled = list(itertools.chain.from_iterable(neg_idx_shuffled.values))  # flatten

    train_neg_dataloader = DataLoader(
        train_neg_dataset,
        batch_size=BATCH_SIZE//2,
        shuffle=False,
        sampler=neg_idx_shuffled,
        num_workers=1,
    )

    model.train()
    print('Epoch %s' % epoch)
    train_pos_dataiter = iter(train_pos_dataloader)
    train_loss_mean = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for batch_idx, (neg_X, neg_y) in enumerate(train_neg_dataloader):
        optimizer.zero_grad()
        try:
            should_write = False
            pos_X, pos_y = next(train_pos_dataiter)
        except StopIteration:
            train_pos_dataiter = iter(train_pos_dataloader)
            pos_X, pos_y = next(train_pos_dataiter)
            
            pos_epoch += 1
            should_write = True

        neg_X = neg_X.reshape(-1, 1, 48, 48, 48)
        train_X = torch.cat([pos_X, neg_X])
        train_y = torch.cat([pos_y, neg_y]).reshape(-1, 1).float()

        train_X = linear_transform_to_0_1(train_X, min=-1000, max=400)
        if SHOULD_AUGMENT:
            flip_dims = []
            flip_x = (np.random.randint(0, 2) == 1)
            flip_y = (np.random.randint(0, 2) == 1)
            flip_z = (np.random.randint(0, 2) == 1)

            if flip_x:
                flip_dims.append(2)
            if flip_y:
                flip_dims.append(3)
            if flip_z:
                flip_dims.append(4)
            if len(flip_dims) > 0:
                train_X= torch.flip(train_X, flip_dims)
        
        train_X = train_X.to(device)
        train_y = train_y.to(device)
        
        pred_y = model(train_X)

        loss = binary_focal_loss_with_logits(pred_y, train_y, alpha=0.5, gamma=2.0, reduction='mean')
        loss.backward()
        optimizer.step()
        
        train_loss_mean += loss.cpu().sum().item()
        pred_y_bool = torch.sigmoid(pred_y) > 0.5
        train_y_bool = (train_y == 1)
        this_tp, this_tn, this_fp, this_fn = calc_confusion(pred_y_bool, train_y_bool)
        
        tp += this_tp
        tn += this_tn
        fp += this_fp
        fn += this_fn
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{:.0f}\t{:.0f}\t{:.0f}\t{:.0f}'.format(
                epoch, batch_idx * len(neg_X), len(train_neg_dataloader.dataset),
                100. * batch_idx / len(train_neg_dataloader), loss.item(), this_tp, this_tn, this_fp, this_fn))

        if should_write:
            writer.add_scalar('loss/train', train_loss_mean / len(train_pos_dataloader), pos_epoch)
            writer.add_scalar('accuracy/train', 100. * (tp + tn) / (tp + fp + tn + fn), pos_epoch)
            try:
                writer.add_scalar('precision/train', 100. * (tp / (tp + fp)), pos_epoch)
            except ZeroDivisionError:
                writer.add_scalar('precision/train', -1, pos_epoch)
            try:
                writer.add_scalar('recall/train', 100. * (tp / (tp + fn)), pos_epoch)
            except ZeroDivisionError:
                writer.add_scalar('recall/train', -1, pos_epoch)
            train_loss_mean = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
    
    # VALIDATION
    print('Validation')
    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for batch_idx, (X, y) in enumerate(val_dataloader):
        X = X.reshape(-1, 1, 48, 48, 48).float()
        y = y.reshape(-1, 1).float()
        X = X.to(device)
        y = y.to(device)
        pred_y = model(X)
        
        pred_y_bool = torch.sigmoid(pred_y) > 0.5
        y_bool = (y == 1)
        this_tp, this_tn, this_fp, this_fn = calc_confusion(pred_y_bool, y_bool)
        
        tp += this_tp
        tn += this_tn
        fp += this_fp
        fn += this_fn
    
    writer.add_scalar('accuracy/val', 100. * (tp + tn) / (tp + fp + tn + fn), epoch)
    writer.add_scalar('tp/val', tp, epoch)
    writer.add_scalar('fp/val', fp, epoch)
    writer.add_scalar('tn/val', tn, epoch)
    writer.add_scalar('fn/val', fn, epoch)
    try:
        writer.add_scalar('precision/val', 100. * (tp / (tp + fp)), epoch)
    except ZeroDivisionError:
        writer.add_scalar('precision/val', -1, epoch)
    try:
        writer.add_scalar('recall/val', 100. * (tp / (tp + fn)), epoch)
    except ZeroDivisionError:
        writer.add_scalar('recall/val', -1, epoch)
    
    model_savepath = (Path(writer.get_logdir()) / f'epoch_{epoch}_{pos_epoch}.pth').as_posix()
    optimizer_savepath = (Path(writer.get_logdir()) / f'epoch_{epoch}_{pos_epoch}_optimizer.pth').as_posix()
    torch.save(model.state_dict(), model_savepath)
    torch.save(optimizer.state_dict(), optimizer_savepath)

