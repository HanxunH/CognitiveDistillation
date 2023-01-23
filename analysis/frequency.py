import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        x = self.out_conv(x)
        return x


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.block1 = nn.Sequential(
            ConvBrunch(3, 32, 3),
            ConvBrunch(32, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            ConvBrunch(32, 64, 3),
            ConvBrunch(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            ConvBrunch(64, 128, 3),
            ConvBrunch(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(4*4*128, 2)
        self.fc_size = 4*4*128

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def convert_dct2(data):
    for i in range(data.shape[0]):
        for c in range(3):
            data[i][:, :, c] = dct2(data[i][:, :, c])
    return data


class FrequencyAnalysis():
    def __init__(self, input_size=32):
        if input_size == 32:
            self.clf = Detector().to(device)
            ckpt = torch.load('checkpoints/frequency_detector_gtsrb_v2.pt') # trained on gtsrb, use for CIFAR evaluation 
            self.clf.load_state_dict(ckpt)
            self.clf.eval()

    def train(self, data):
        # Already Pretrained
        return

    def predict(self, data, t=1):
        """
            data (np.array) b,h,w,c
        """
        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(data))):
                x, _ = data[i]
                x = x.clone().numpy()
                for c in range(3):
                    x[c, :, :] = dct2((x[c, :, :]*255).astype(np.uint8))
                x = torch.from_numpy(x)
                out = self.clf(x.unsqueeze(0).to(device)).detach().cpu()
                _, p = torch.max(out.data, 1)
                predictions.append(p)
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        return predictions

    def analysis(self, data, is_test=False):
        """
            data (np.array) b,h,w,c
        """
        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(data))):
                x, _ = data[i]
                x = x.clone().numpy()
                for c in range(3):
                    x[c, :, :] = dct2((x[c, :, :]*255).astype(np.uint8))
                x = torch.from_numpy(x)
                p = self.clf(x.unsqueeze(0).to(device)).detach().cpu()
                p = F.softmax(p, dim=1)
                predictions.append(p)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions[:, 1].detach().cpu().numpy()
        return predictions
