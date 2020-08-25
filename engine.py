import torch
import config
from tqdm import tqdm
import numpy as np
from utils import decode


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
        mini_batch_loss.append(loss.cpu().detach().numpy())
    return np.mean(mini_batch_loss)


def valid_fn(model, data_loader, device, loss_fn, scheduler, encoder):
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
            results = decode(output, encoder)
            print([''.join(encoder.inverse_transform(r-1)) for r in labels.cpu().detach().numpy()], results)
        mean_loss = np.mean(mini_batch_loss)
        scheduler.step(mean_loss)
    return np.mean(mean_loss)
