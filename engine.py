import torch
import config
from tqdm import tqdm
import numpy as np

def train_fn(model, data_loader, optimizer, scheduler, device, loss_fn):
    mini_batch_loss = []
    model.train()
    for batch in tqdm(data_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        logits = model(images)
        target_labels = torch.nn.functional.log_softmax(logits, 2).permute(1, 0, 2)

        input_lengths = torch.full(size=(target_labels.shape[1],),
                                   fill_value=config.input_seq_len,
                                   dtype=torch.int32)

        target_lengths = torch.full(size=(target_labels.shape[1],),
                                   fill_value=config.target_seq_len,
                                   dtype=torch.int32)

        loss = loss_fn(target_labels, labels, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        mini_batch_loss.append(loss.cpu().detach().numpy())
    print(input_lengths, input_lengths.shape)
    print(target_lengths, target_lengths.shape)
    return np.mean(mini_batch_loss)


def valid_fn(model, data_loader, device, loss_fn, encoder):
    mini_batch_loss = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(images)
            target_labels = torch.nn.functional.log_softmax(logits, 2)
            target_labels_reshaped = target_labels.permute(1, 0, 2)
            input_lengths = torch.full(size=(target_labels_reshaped.shape[1],),
                                       fill_value=config.input_seq_len,
                                       dtype=torch.long)
            target_lengths = torch.full(size=(target_labels_reshaped.shape[1],),
                                        fill_value=config.target_seq_len,
                                        dtype=torch.int32)

            loss = loss_fn(target_labels_reshaped, labels, input_lengths, target_lengths)
            mini_batch_loss.append(loss.cpu().detach().numpy())
            output = np.argmax(target_labels.cpu().detach().numpy(), 2) - 1
            print(labels, output)
    return np.mean(mini_batch_loss)

