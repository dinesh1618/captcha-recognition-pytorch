from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    for data in tqdm(data_loader):
        for images, targets in data.items():
            data[images] = targets.to(config.DEVICE)
        optimizer.zero_grad()
        # print(data['images'], data['targets'])
        _, loss = model(data['images'], data['targets'])
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss/len(data_loader)

def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_pred = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            for images, targets in data.items():
                data[images] = targets.to(config.DEVICE)
            batch_pred, loss = model(data['images'], data['targets'])
            fin_loss += loss.item()
            fin_pred.append(batch_pred)
        return fin_pred, fin_loss / len(data_loader)