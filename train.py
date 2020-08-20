from dataset import CaptchaDataset
from glob import glob
import numpy as np
from model import CaptchaModel
from sklearn.preprocessing import LabelEncoder

files = glob('input/captcha_images_v2/*.png')
chars = [list(file.split('/')[-1][:-4]) for file in files]
flat_list = [c for char in chars for c in char]
label_encoder = LabelEncoder().fit(flat_list)
label_encoded = np.array([label_encoder.transform(c) for c in chars]) + 1

dataset = CaptchaDataset(files, label_encoded)
model = CaptchaModel()

# Todo
# 1. loss
# 2. optimizer (adam)
# 3. scheduler
# 4. train script
# 5. test script
# 6. evaluation script