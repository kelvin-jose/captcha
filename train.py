from dataset import CaptchaDataset
from glob import glob
import numpy as np
import torch
import config
from engine import train_fn
from engine import valid_fn
from model import CaptchaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

files = glob('input/captcha_images_v2/*.png')
labels = [name.split('/')[-1][:-4] for name in files]
chars = [list(file) for file in labels]
flat_list = [c for char in chars for c in char]
label_encoder = LabelEncoder().fit(flat_list)
label_encoded = np.array([label_encoder.transform(c) for c in chars]) + 1

x_train, x_test, y_train, y_test = train_test_split(files, label_encoded, test_size=0.1, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataset = torch.utils.data.DataLoader(CaptchaDataset(x_train, y_train),
                                            batch_size=config.train_batch,
                                            num_workers=4)
test_dataset = torch.utils.data.DataLoader(CaptchaDataset(x_test, y_test),
                                           batch_size=config.test_batch,
                                           num_workers=4)
model = CaptchaModel().to(device)
loss_fn = torch.nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)

for epoch in range(config.epochs):
    train_loss = train_fn(model, train_dataset, optimizer, scheduler, device, loss_fn)
    val_loss = valid_fn(model, test_dataset, device, loss_fn, scheduler, label_encoder)
    print(f'epoch: {epoch} train loss: {train_loss} valid loss: {val_loss}')
