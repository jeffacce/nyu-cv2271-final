"""
Training script
"""
import time, torch

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

############ Model Training ########
def train_model(net, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, start):
    net.train()    
    running_loss = 0.0
    for index, (vol, seg) in enumerate(dataloader):
        vol = vol.to(device)
        seg = seg.to(device)
        
        outputs = net(vol)
        loss = criterion(outputs, seg)

        running_loss += loss.item() * vol.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('index:{:3}, loss:{:6.3f}, time:{:8.3f} min'.format(index, loss.item(), (time.time() - start) / 60))
        if index % 5 == 0:
            writer.add_scalar("Loss", loss.item(), epoch)
            writer.flush()
    epoch_loss = running_loss / len(dataloader.dataset)
    writer.add_scalar("Train_poch_loss", epoch_loss, epoch)
    writer.flush()
    return epoch_loss



############ Model Validating ########
def val_model(net, dataloader, criterion, device, epoch, num_epochs, writer, start):
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for index, (vol, seg) in enumerate(dataloader):
            vol = vol.to(device)
            seg = seg.to(device)

            outputs = net(vol)
            loss = criterion(outputs, seg)

            running_loss += loss.item() * vol.size(0)

            print('index:{:3}, loss:{:6.3f}, time:{:8.3f} min'.format(index, loss.item(), (time.time() - start) / 60))
    epoch_loss = running_loss / len(dataloader.dataset)
    writer.add_scalar("Val_poch_loss", epoch_loss, epoch)
    writer.flush()
    return epoch_loss


