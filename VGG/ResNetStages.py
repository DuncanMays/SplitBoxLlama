import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


def _make_layer(in_channels, out_channels, num_blocks, stride):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNetStage0(nn.Module):
    """
    Initial 3x3 conv (16ch) + n BasicBlocks at 16 channels.
    Input:  B x  3 x 32 x 32
    Output: B x 16 x 32 x 32
    """
    def __init__(self, num_blocks: int = 18):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.blocks = _make_layer(16, 16, num_blocks, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.entry(x))


class ResNetStage1(nn.Module):
    """
    n BasicBlocks at 32 channels; first block uses stride 2 to halve spatial dims.
    Input:  B x 16 x 32 x 32
    Output: B x 32 x 16 x 16
    """
    def __init__(self, num_blocks: int = 18):
        super().__init__()
        self.blocks = _make_layer(16, 32, num_blocks, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResNetStage2(nn.Module):
    """
    n BasicBlocks at 64 channels (stride-2 entry) + GlobalAvgPool + FC classifier.
    Input:  B x 32 x 16 x 16
    Output: B x num_classes
    """
    def __init__(self, num_blocks: int = 18, num_classes: int = 10):
        super().__init__()
        self.blocks = _make_layer(32, 64, num_blocks, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        out = self.pool(out).flatten(1)
        return self.fc(out)


if __name__ == "__main__":
    batch_size = 4
    num_classes = 10

    stage0 = ResNetStage0(num_blocks=18)
    stage1 = ResNetStage1(num_blocks=18)
    stage2 = ResNetStage2(num_blocks=18, num_classes=num_classes)

    x = torch.randn(batch_size, 3, 32, 32)

    out0 = stage0(x)
    assert out0.shape == (batch_size, 16, 32, 32), f"Unexpected shape: {out0.shape}"
    print(f"Stage 0 output: {tuple(out0.shape)}")

    out1 = stage1(out0)
    assert out1.shape == (batch_size, 32, 16, 16), f"Unexpected shape: {out1.shape}"
    print(f"Stage 1 output: {tuple(out1.shape)}")

    out2 = stage2(out1)
    assert out2.shape == (batch_size, num_classes), f"Unexpected shape: {out2.shape}"
    print(f"Stage 2 output: {tuple(out2.shape)}")

    # Verify the three stages compose cleanly as nn.Sequential (local inference path)
    model = nn.Sequential(stage0, stage1, stage2)
    output = model(x)
    assert output.shape == (batch_size, num_classes), f"Unexpected shape: {output.shape}"
    print(f"nn.Sequential output: {tuple(output.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("All forward passes succeeded.")
