import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path("dataset")
IMG_DIR  = BASE / "images"
MASK_DIR = BASE / "masks"

# جيب الصور اللي عندها ماسك بس
all_images = os.listdir(str(IMG_DIR))
valid_pairs = []

for img_file in all_images:
    stem = os.path.splitext(img_file)[0]  # ISIC_0030912
    mask_file = f'{stem}_segmentation.png'
    if os.path.exists(str(MASK_DIR / mask_file)):
        valid_pairs.append(img_file)

print(f'✅ Valid pairs found: {len(valid_pairs)}')

# قسّم 80% train / 20% val
train_imgs, val_imgs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
print(f'Train: {len(train_imgs)} | Val: {len(val_imgs)}')

# إنشاء المجلدات
for split in ['train', 'val']:
    os.makedirs(f'{split}/images', exist_ok=True)
    os.makedirs(f'{split}/masks', exist_ok=True)

# نسخ الملفات
for img_file in train_imgs:
    stem = os.path.splitext(img_file)[0]
    shutil.copy(str(IMG_DIR / img_file),                          f'train/images/{img_file}')
    shutil.copy(str(MASK_DIR / f'{stem}_segmentation.png'),       f'train/masks/{img_file}')  # نفس اسم الصورة

for img_file in val_imgs:
    stem = os.path.splitext(img_file)[0]
    shutil.copy(str(IMG_DIR / img_file),                          f'val/images/{img_file}')
    shutil.copy(str(MASK_DIR / f'{stem}_segmentation.png'),       f'val/masks/{img_file}')

print('✅ Dataset split done!')

class CFG:
    IMAGE_DIR_TRAIN = "train/images"
    MASK_DIR_TRAIN = "train/masks"
    IMAGE_DIR_VAL = "val/images"
    MASK_DIR_VAL = "val/masks"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-4
    MODEL_PATH = "best_model.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Using device: {CFG.DEVICE}")

# ============================================================
# 🧠 3. Dataset Class
# ============================================================
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = np.where(mask > 127, 1, 0).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"].unsqueeze(0)
        return image, mask

# ============================================================
# 🎨 4. Data Augmentation
# ============================================================
train_transform = A.Compose([
    A.Resize(*CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(*CFG.IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ============================================================
# 🧩 5. Model Definition (UNet++ + EfficientNetB3)
# ============================================================
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model.to(CFG.DEVICE)

# ============================================================
# ⚔️ 6. Custom Loss Function (Dice + BCE)
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return self.bce(inputs, targets) + 1 - dice

# ============================================================
# 🚀 7. Training & Validation Functions
# ============================================================
def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# ✅ Dice صح على مستوى كل صورة لوحدها
@torch.no_grad()
def validate(loader, model, criterion):
    model.eval()
    val_loss = 0
    all_dice = []

    for imgs, masks in tqdm(loader, desc="Validating", leave=False):
        imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)
        preds = model(imgs)
        loss = criterion(preds, masks)
        val_loss += loss.item()

        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()

        # حساب Dice لكل صورة لوحدها
        batch_size = imgs.shape[0]
        for i in range(batch_size):
            pred_i = preds_bin[i].view(-1)
            mask_i = masks[i].view(-1)
            intersection = (pred_i * mask_i).sum()
            dice_i = (2. * intersection + 1) / (pred_i.sum() + mask_i.sum() + 1)
            all_dice.append(dice_i.item())

    avg_dice = sum(all_dice) / len(all_dice)
    return val_loss / len(loader), avg_dice

# ============================================================
# 🧭 8. Training Loop
# ============================================================
def fit(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = DiceBCELoss()

    best_dice = 0
    for epoch in range(CFG.EPOCHS):
        print(f"\n📆 Epoch [{epoch+1}/{CFG.EPOCHS}]")
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion)
        val_loss, val_dice = validate(val_loader, model, criterion)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice*100:.2f}%")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CFG.MODEL_PATH)
            print("✅ Model Saved (New Best Dice Score)")

    print(f"\n🏁 Training Finished!")
    print(f"🏆 Best Dice Score: {best_dice*100:.2f}%")
    print(f"Best model saved as: {CFG.MODEL_PATH}")
# ============================================================
# 🧩 9. Run Everything
# ============================================================
if __name__ == "__main__":
    train_ds = SegDataset(CFG.IMAGE_DIR_TRAIN, CFG.MASK_DIR_TRAIN, transform=train_transform)
    val_ds = SegDataset(CFG.IMAGE_DIR_VAL, CFG.MASK_DIR_VAL, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE)

    model = build_model()
    fit(model, train_loader, val_loader)