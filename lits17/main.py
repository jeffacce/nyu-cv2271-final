import os, sys, random, time, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
sys.path.append(os.path.split(sys.path[0])[0])
import parameter as params
from dataset import Dataset
from loss import TverskyLoss, MultiTverskyLoss, BinaryTverskyLossV2, FocalBinaryTverskyLoss
from UNet3D import UNet3D
from train_ds import train_model, val_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load preprocessed data.
def load_data_to_dictionaries(path):
    """
    output dictionaries that record the input file paths
    """
    data = {}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            idx = int(re.findall(r'\d+', filename)[0])
            data[int(idx)] = os.path.join(dirname, filename)
    return data

def get_data_list(vol_dir, seg_dir):
    vol_dict = load_data_to_dictionaries(vol_dir)
    seg_dict = load_data_to_dictionaries(seg_dir)

    return [[vol_dict[key],seg_dict[key]] for key in sorted(vol_dict.keys())]

data = get_data_list(params.train_vol_path, params.train_seg_path)
random.shuffle(data)
train_size = int(0.8 * len(data))

train_data = data[:train_size]
val_data = data[train_size:]

train_loader = torch.utils.data.DataLoader(dataset = Dataset(train_data), 
                          batch_size = params.batch_size, 
                          shuffle = True, 
                          num_workers = params.num_workers,
                          pin_memory = params.pin_memory)

val_loader = torch.utils.data.DataLoader(dataset = Dataset(val_data),
                        batch_size = params.batch_size, 
                        shuffle = False, 
                        num_workers = params.num_workers, 
                        pin_memory = params.pin_memory)

loss_func = TverskyLoss()

# # Define network
net = UNet3D(4,1)
net = torch.nn.DataParallel(net).to(device)
# net.load_state_dict(torch.load('/scratch/ec2684/cv/lits17/net50-0.020-0.027.pth'))
print('net total parameters:', sum(param.numel() for param in net.parameters()))

# Define optimizer
opt = torch.optim.AdamW(net.parameters(), lr=params.learning_rate)

# Learning rate decay
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, params.learning_rate_decay, gamma = 0.5)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# # Set graphics card related
os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
cudnn.benchmark = params.cudnn_benchmark

val_acc_history = []

early_stopping = False
best_val_loss = 1E5
val_no_improve = 0
patience = params.early_stopping
best_acc = 0.0
swa_begin = False
num_epochs = params.epochs
writer = SummaryWriter()
print()

since = time.time()
######## Model Train/Validation Phase ########
for epoch in range(num_epochs):
    if(early_stopping):
        break
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 15)
    print('Training:')
    # Training phase
    train_epoch_loss = train_model(net, train_loader, loss_func, opt, device, epoch, num_epochs, writer, since)
    print('\nTraining epoch {} Mean Loss: {:.4f}\n'.format(epoch, train_epoch_loss))
    print('Validation:')
    # Validation phase
    val_epoch_loss = val_model(net, val_loader, loss_func, device, epoch, num_epochs, writer, since)
    print('\nValidation epoch {} Mean Loss: {:.4f}\n'.format(epoch, val_epoch_loss))

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        model_file = os.getcwd()+'/checkpoints/net{}-{:.3f}.pth'.format(epoch, val_epoch_loss)
        torch.save(net.state_dict(), model_file)
        print('Saved model to ' + model_file + '.')
        val_no_improve = 0
    else:
        val_no_improve += 1
    if (patience <= val_no_improve):
        print(f'\nEarly Stopping at epoch {epoch}.')
        early_stopping = True
        break
    scheduler.step()
    print('\n')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Loss: {:4f}'.format(best_val_loss))
print('\n')
writer.close()