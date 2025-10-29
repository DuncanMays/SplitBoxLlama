import torch
from torch import nn
F = torch.nn.functional

class VGGBlock_1(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.BATCH_SIZE = None

        self.conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1))
        self.conv2 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1))

        self.maxPool1 = torch.nn.MaxPool2d((2,2))
        self.bn1 = torch.nn.BatchNorm2d(128)

        self.conv3 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.conv4 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1))

        self.maxPool2 = torch.nn.MaxPool2d((2,2))
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.do2 = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = x.to(self.device)

        if (self.BATCH_SIZE == None):
            self.BATCH_SIZE = x.shape[0]

            self.pad1 = torch.zeros([self.BATCH_SIZE, 64, 32, 32]).to(self.device)
            self.pad2 = torch.zeros([self.BATCH_SIZE, 128, 16, 16]).to(self.device)
            self.pad3 = torch.zeros([self.BATCH_SIZE, 256, 8, 8]).to(self.device)
            self.pad4 = torch.zeros([self.BATCH_SIZE, 512, 4, 4]).to(self.device)
            self.pad5 = torch.zeros([self.BATCH_SIZE, 1024, 2, 2]).to(self.device)

        # 64*64*3
        x = self.conv1(x)
        # 64*64*64
        skip = x
        x = F.relu(self.conv2(x))
        # 64*64*128
        # skip padded to be 128, 32, 32
        skip = torch.cat([skip, self.pad1], dim=1)
        # doing the skip connection stuff (what skip connections are supposed to do)
        x = x + skip
        skip = torch.clone(x)
        x = self.maxPool1(x)
        # 32*32*128

        x = self.conv3(x)
        # 32*32*128
        x = F.relu(self.conv4(x))
        # 32*32*256

        # skip fed through the same maxpool as x and then padded with zeros to be the same shape
        skip = self.maxPool1(skip)
        skip = torch.cat([skip, self.pad2], dim=1)
        # doing the skip connection stuff (what skip connections are supposed to do)
        x = x + skip
        skip = torch.clone(x)

        x = self.do2(self.maxPool2(x))
        # 16*16*256

        return x

class VGGBlock_2(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.BATCH_SIZE = None

        self.maxPool2 = torch.nn.MaxPool2d((2,2))

        self.conv5 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1))
        self.conv6 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1))
        self.conv7 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1))
        self.conv8 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1))

        self.maxPool3 = torch.nn.MaxPool2d((2,2))
        self.bn3 = torch.nn.BatchNorm2d(512)
        self.do3 = torch.nn.Dropout(p=0.4)

        self.conv9 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1))
        self.conv10 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1))
        self.conv11 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1))
        self.conv12 = torch.nn.Conv2d(512, 1024, (3,3), padding=(1,1))

        self.maxPool4 = torch.nn.MaxPool2d((2,2))
        self.bn4 = torch.nn.BatchNorm2d(1024)
        self.do4 = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = x.to(self.device)

        if (self.BATCH_SIZE == None):
            self.BATCH_SIZE = x.shape[0]

            self.pad1 = torch.zeros([self.BATCH_SIZE, 64, 32, 32]).to(self.device)
            self.pad2 = torch.zeros([self.BATCH_SIZE, 128, 16, 16]).to(self.device)
            self.pad3 = torch.zeros([self.BATCH_SIZE, 256, 8, 8]).to(self.device)
            self.pad4 = torch.zeros([self.BATCH_SIZE, 512, 4, 4]).to(self.device)
            self.pad5 = torch.zeros([self.BATCH_SIZE, 1024, 2, 2]).to(self.device)

        x = self.conv5(x)
        # 16*16*256
        x = self.conv6(x)
        # 16*16*256
        x = self.conv7(x)
        # 16*16*256
        x = F.relu(self.conv8(x))
        # 16*16*512

        # skip fed through the same maxpool as x and then padded with zeros to be the same shape
        skip = self.maxPool2(skip)
        skip = torch.cat([skip, self.pad3], dim=1)
        # doing the skip connection stuff (what skip connections are supposed to do)
        x = x + skip
        skip = torch.clone(x)

        x = self.do3(self.maxPool3(x))
        # 8*8*512

        x = self.conv9(x)
        # 8*8*512
        x = self.conv10(x)
        # 8*8*512
        x = self.conv11(x)
        # 8*8*512
        x = F.relu(self.conv12(x))
        # 8*8*1024

        # skip fed through the same maxpool as x and then padded with zeros to be the same shape
        skip = self.maxPool3(skip)
        skip = torch.cat([skip, self.pad4], dim=1)
        # doing the skip connection stuff (what skip connections are supposed to do)
        x = x + skip
        skip = torch.clone(x)

        x = self.do4(self.maxPool4(x))
        # 4*4*1024

        return x

class VGGBlock_3(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.BATCH_SIZE = None

        self.conv13 = torch.nn.Conv2d(1024, 1024, (3,3), padding=(1,1))
        self.conv14 = torch.nn.Conv2d(1024, 1024, (3,3), padding=(1,1))
        self.conv15 = torch.nn.Conv2d(1024, 1024, (3,3), padding=(1,1))
        self.conv16 = torch.nn.Conv2d(1024, 2048, (3,3), padding=(1,1))

        self.maxPool5 = torch.nn.MaxPool2d((2,2))
        self.bn5 = torch.nn.BatchNorm2d(2048)
        self.do5 = torch.nn.Dropout(p=0.4)

        self.flatten = torch.nn.Flatten()
        
        self.dense1 = torch.nn.Linear(2048, 2048)
        self.dense2 = torch.nn.Linear(2048, 1000)
        self.dense3 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = x.to(self.device)

        if (self.BATCH_SIZE == None):
            self.BATCH_SIZE = x.shape[0]

            self.pad1 = torch.zeros([self.BATCH_SIZE, 64, 32, 32]).to(self.device)
            self.pad2 = torch.zeros([self.BATCH_SIZE, 128, 16, 16]).to(self.device)
            self.pad3 = torch.zeros([self.BATCH_SIZE, 256, 8, 8]).to(self.device)
            self.pad4 = torch.zeros([self.BATCH_SIZE, 512, 4, 4]).to(self.device)
            self.pad5 = torch.zeros([self.BATCH_SIZE, 1024, 2, 2]).to(self.device)

        x = self.conv13(x)
        # 4*4*1024
        x = self.conv14(x)
        # 4*4*1024
        x = self.conv15(x)
        # 4*4*1024
        x = F.relu(self.conv16(x))
        # 4*4*2048

        # skip fed through the same maxpool as x and then padded with zeros to be the same shape
        skip = self.maxPool4(skip)
        skip = torch.cat([skip, self.pad5], dim=1)
        # doing the skip connection stuff (what skip connections are supposed to do)
        x = x + skip

        x = self.do5(self.maxPool5(x))
        # 2*2*2048

        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.softmax(self.dense3(x), dim=1)

        return x

if (__name__ == "__main__"):

    block_1 = VGGBlock_1()
    block_2 = VGGBlock_2()
    block_3 = VGGBlock_3()

    x = torch.randn([64, 3, 32, 32])

    x = block_1(x)
    x = block_2(x)
    x = block_3(x)

    print(x.shape)