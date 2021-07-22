import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps
from helpers import Config, imshow
import os


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), \
               img0_tuple[1], img1_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

if __name__ == "__main__":
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    resize_to_tensor = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    train_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=resize_to_tensor, should_invert=False)

    vis_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())
    print(example_batch[3].numpy())
    print(example_batch[4].numpy())