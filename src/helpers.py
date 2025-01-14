import matplotlib.pyplot as plt
import numpy as np


class Config():
    training_dir = "../data/faces/training/"
    testing_dir = "../data/faces/testing/"

    train_batch_size = 32
    train_number_epochs = 100

    checkpoint_path = '../checkpoints/model.pt'


def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor': 'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()