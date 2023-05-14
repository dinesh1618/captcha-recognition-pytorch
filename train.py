import torch
from glob import glob
import os, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

import config
import engine
from dataset import ClassificationDataset
from model import CaptchModel

def run_training():
    image_files = glob(os.path.join(config.DATA_DIR, '*.png'))
    list_targets = [re.split(r"\\|\.", img)[-2] for img in image_files]
    targets = [[c for c in target] for target in list_targets]
    flat_targets = [c for t in targets for c in t]
    le = LabelEncoder()
    le.fit(flat_targets)
    le_enc = [le.transform(t) for t in targets]
    le_enc = np.array(le_enc) + 1
    (
        train_imgs,
        test_imgs,
        train_target,
        test_target,
        _,
        test_target_origin) = train_test_split(image_files, le_enc, list_targets, train_size=0.1, random_state=42)

    train_dataset = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_target,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    train_load = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_dataset = ClassificationDataset(
        image_paths=test_imgs,
        targets=test_target,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    test_load = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = CaptchModel(num_char=len(le.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-1)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_load, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_load)
        print(f"Epochs: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

if __name__ == "__main__":
    run_training()
