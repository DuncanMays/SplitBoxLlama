import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ResNetStages import ResNetStage0, ResNetStage1, ResNetStage2
from CIFAR10_data import x_train, y_train, x_test, y_test

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Transforms ────────────────────────────────────────────────────────────────
# CIFAR-10 per-channel mean/std (computed over the training set)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# Training: pad→random crop→random flip→normalize (applied per sample)
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize(MEAN, STD),
])

# Evaluation: normalise only
eval_transform = T.Normalize(MEAN, STD)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CIFAR10Tensors(Dataset):
    """Wraps pre-loaded tensors and applies a per-sample transform."""
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]

train_loader = DataLoader(
    CIFAR10Tensors(x_train, y_train, transform=train_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    CIFAR10Tensors(x_test, y_test, transform=eval_transform),
    batch_size=512, shuffle=False, num_workers=4, pin_memory=True,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    ResNetStage0(num_blocks=18),
    ResNetStage1(num_blocks=18),
    ResNetStage2(num_blocks=18, num_classes=10),
).to(device)

# torch.compile gives a free speed-up on PyTorch 2.x
if hasattr(torch, 'compile'):
    model = torch.compile(model)

# ── Optimiser & schedule ──────────────────────────────────────────────────────
# SGD + Nesterov + cosine annealing is the standard recipe for ResNets on CIFAR-10
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Label smoothing regularises the output distribution and consistently improves accuracy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Accuracy helper ───────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return 100.0 * correct / total

# ── Training loop ─────────────────────────────────────────────────────────────
print('Starting training...')
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in tqdm(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    scheduler.step()

    train_acc = evaluate(train_loader)
    test_acc  = evaluate(test_loader)
    lr_now    = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch:3d}/{EPOCHS}  |  train: {train_acc:.2f}%  |  test: {test_acc:.2f}%  |  lr: {lr_now:.6f}')
