import torch
import torch.nn.functional as F
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8))

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 64))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class SoftMaxLoss(torch.nn.Module):

    def __init__(self):
        super(SoftMaxLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.pred = []

        D_in = 128
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(D_in, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2))

    def forward(self, output1, output2, label):
        concat = torch.cat((output1, output2), 1)
        pred = self.fc(concat)
        self.pred = self.sigmoid(pred)
        current_loss = self.loss(pred, label.squeeze().long())
        return current_loss

    def get_accuracy(self, pred, label):
        return torch.sum(torch.argmax(pred, dim=1) == label.squeeze()) / (label.shape[0] + 1e-6)
