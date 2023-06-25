import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional
import segmentation_models_pytorch as smp


class Dataset(BaseDataset):
  def __init__(
      self,
      df,
      transform = None,
      classes = None,
      augmentation = None
      ):
    self.imgpath_list = df

  def __getitem__(self, i):
    imgpath = self.imgpath_list[i]
    img = cv2.imread(imgpath)
    img = cv2.resize(img, dsize = (256, 256))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.permute(2, 0, 1)


    data = {"img": img, "label": img}
    return data
  
  def __len__(self):
    return len(self.imgpath_list)


class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        size = 64
        self.TCB1 = TwoConvBlock(3, size, size)
        self.TCB2 = TwoConvBlock(size, size * 2, size * 2)
        self.TCB3 = TwoConvBlock(size * 2, size * 4, size * 4)
        self.TCB4 = TwoConvBlock(size * 4, size * 8, size * 8)
        self.TCB5 = TwoConvBlock(size * 8, size * 16, size * 16)

        self.TCB6 = TwoConvBlock(size * 16, size * 8, size * 8)
        self.TCB7 = TwoConvBlock(size * 8, size * 4, size * 4)
        self.TCB8 = TwoConvBlock(size * 4, size * 2, size * 2)
        self.TCB9 = TwoConvBlock(size * 2 , size, size)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(size * 16, size * 16) 
        self.UC2 = UpConv(size * 8, size * 8) 
        self.UC3 = UpConv(size * 4, size * 4) 
        self.UC4= UpConv(size * 2, size * 2)
        #x = self.avgpool(x)
        self.conv1 = nn.Conv2d(size, 3, kernel_size = 1)
        #self.soft = nn.Softmax(dim = 1)
        self.soft = nn.Sigmoid()
    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = self.TCB9(x)

        x = self.conv1(x)
        x = self.soft(x)
        return x
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32
    import glob
    train_list = glob.glob("F:/VRchat/python/unet/train/*.jpg")
    train_dataset =Dataset(train_list)
    print(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            shuffle=True)
    val_list = glob.glob("F:/VRchat/python/unet/train/*.jpg")
    val_dataset = Dataset(val_list)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            shuffle=True)

    test_dataset = Dataset(glob.glob("F:/VRchat/python/unet/train/*.jpg"))
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            num_workers=4)

    unet = UNet_2D().to(device)
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    history = {"train_loss": []}

    model = UNet_2D()
    #モデルを選択する。（雑
    model.load_state_dict(torch.load("./train_49.pth"))
    model.eval()
    count = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data["img"], data["label"]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print("loss: ",loss.item())
            #直下に大量に生成画像ができる。
            plt.figure()
            plt.imshow(data["img"][0,:,:,:].permute(1, 2, 0))
            plt.title("original_image")
            plt.axis("off")
            plt.savefig("original_image"+str(count) + ".png") 
            plt.figure()
            plt.imshow(outputs[0,:,:,:].permute(1, 2, 0))
            plt.title("outputs_image")
            plt.axis("off")
            plt.savefig("outputs"+str(count) + ".png") 
            count = count + 1