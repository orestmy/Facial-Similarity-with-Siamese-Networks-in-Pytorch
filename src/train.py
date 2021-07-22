import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
from data import SiameseNetworkDataset
from helpers import Config
from models import SiameseNetwork, SoftMaxLoss

class Trainer():
    def __init__(self):

        train_folder = dset.ImageFolder(root=Config.training_dir)
        test_folder = dset.ImageFolder(root=Config.testing_dir)

        resize_to_tensor = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

        train_dataset = SiameseNetworkDataset(imageFolderDataset=train_folder, transform=resize_to_tensor, should_invert=False)
        test_dataset = SiameseNetworkDataset(imageFolderDataset=test_folder, transform=resize_to_tensor, should_invert=False)

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=32)
        self.test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=32)

        self.net = SiameseNetwork().cuda()
        self.criterion = SoftMaxLoss().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0005)

        self.counter = []
        self.loss_history = []
        self.iteration_number = 0

    def train_epoch(self, epochNumber):
        for i, data in enumerate(self.train_dataloader, 0):
            img0, img1, label, _, _ = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            self.optimizer.zero_grad()
            output1, output2 = self.net(img0, img1)
            loss_contrastive = self.criterion(output1, output2, label)
            loss_contrastive.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epochNumber, loss_contrastive.item()))
                self.iteration_number += 10
                self.counter.append(self.iteration_number)
                self.loss_history.append(loss_contrastive.item())

if __name__ == "__main__":
    trainer = Trainer()
    for epoch in range(0, Config.train_number_epochs):
        trainer.train_epoch(epoch)