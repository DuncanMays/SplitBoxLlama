import torch
import torch.nn as nn
import torchvision.transforms as T
import axon
import asyncio
import pickle

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ResNetStages import ResNetStage0, ResNetStage1, ResNetStage2
from CIFAR10_data import x_train, y_train, x_test, y_test

from SplitBox.worker import NeuralBlock, get_arg
from SplitBox.multi_stub import get_multi_stub
from SplitBox.pipeline_client import get_training_flow, get_eval_flow

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS           = 200
BATCH_SIZE       = 128
NUM_MINI_BATCHES = 8     # pipeline depth; mini-batch size = BATCH_SIZE // NUM_MINI_BATCHES = 16
LR               = 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4

# ── Transforms ────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize(MEAN, STD),
])
eval_transform = T.Normalize(MEAN, STD)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CIFAR10Tensors(Dataset):
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

# ── Block factories ───────────────────────────────────────────────────────────
def make_optimizer(params):
    return torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

stage_fns = [
    lambda: ResNetStage0(num_blocks=18),
    lambda: ResNetStage1(num_blocks=18),
    lambda: ResNetStage2(num_blocks=18, num_classes=10),
]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Training coroutine ────────────────────────────────────────────────────────
async def train(stubs, block_stubs, urls):

    global_stub      = get_multi_stub(stubs)
    multi_block_stub = get_multi_stub(block_stubs)

    print("creating and uploading blocks")
    local_blocks = [NeuralBlock(fn, make_optimizer, make_scheduler) for fn in stage_fns]
    block_states = [block.get_state() for block in local_blocks]

    await multi_block_stub.load_blocks([[[]] for _ in block_stubs])
    for block_stub, state in zip(block_stubs, block_states):
        print(f"  loading block: {len(pickle.dumps(state))} bytes")
        await block_stub.push_block(state)

    # drop_last so every batch divides evenly into NUM_MINI_BATCHES
    train_loader = DataLoader(
        CIFAR10Tensors(x_train, y_train, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        CIFAR10Tensors(x_test, y_test, transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True,
    )

    async def evaluate(loader):
        await global_stub.train_mode([(False,) for _ in stubs])
        correct = total = 0
        for xb, yb in loader:
            xb_mini = xb.reshape([NUM_MINI_BATCHES, BATCH_SIZE // NUM_MINI_BATCHES, *xb.shape[1:]])
            flow, outputs = get_eval_flow(stubs, urls, xb_mini)
            await flow.start()
            await global_stub.clear_cache()
            logits = torch.cat([o[0] for o in outputs], dim=0)
            preds  = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
        await global_stub.train_mode([(True,) for _ in stubs])
        return 100.0 * correct / total

    print("Starting training!")
    for epoch in range(1, EPOCHS + 1):
        await global_stub.train_mode([(True,) for _ in stubs])

        epoch_losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb = xb.reshape([NUM_MINI_BATCHES, BATCH_SIZE // NUM_MINI_BATCHES, *xb.shape[1:]])
            yb = yb.reshape([NUM_MINI_BATCHES, BATCH_SIZE // NUM_MINI_BATCHES])

            flow, losses = get_training_flow(stubs, urls, xb, yb, criterion)
            await flow.start()
            await multi_block_stub.step([{"zero_grad": True} for _ in stubs])
            await global_stub.clear_cache()
            epoch_losses.extend(losses)

        await multi_block_stub.scheduler_step()

        train_acc = await evaluate(train_loader)
        test_acc  = await evaluate(test_loader)
        avg_loss  = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch:3d}/{EPOCHS}  |  loss: {avg_loss:.4f}  |  train: {train_acc:.2f}%  |  test: {test_acc:.2f}%")


# ── Entry points ──────────────────────────────────────────────────────────────
async def local_mode():
    print("instantiating network")

    urls = [
        "localhost:8001/llama_worker",
        "localhost:8002/llama_worker",
        "localhost:8003/llama_worker",
    ]
    stubs = [axon.client.get_stub(url) for url in urls]

    socket_tl   = axon.socket_transport.client()
    socket_urls = [
        "localhost:9001/block_stack",
        "localhost:9002/block_stack",
        "localhost:9003/block_stack",
    ]
    block_stubs = [axon.client.get_stub(url, tl=socket_tl) for url in socket_urls]

    await train(stubs, block_stubs, urls)


if __name__ == "__main__":
    mode = get_arg("local", "-mode")

    if mode == "local":
        asyncio.run(local_mode())
    else:
        raise ValueError(f"unrecognised mode: {mode}")
