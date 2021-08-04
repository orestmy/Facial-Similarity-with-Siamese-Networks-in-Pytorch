import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
from data import SiameseNetworkDataset
from helpers import Config, show_plot
from models import SiameseNetwork, SoftMaxLoss

class Trainer():
    def __init__(self):

        train_folder = dset.ImageFolder(root=Config.training_dir)
        test_folder = dset.ImageFolder(root=Config.testing_dir)

        resize_to_tensor = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

        train_dataset = SiameseNetworkDataset(imageFolderDataset=train_folder, transform=resize_to_tensor, should_invert=False)
        test_dataset = SiameseNetworkDataset(imageFolderDataset=test_folder, transform=resize_to_tensor, should_invert=False)

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=32)
        self.test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=32)

        self.model = SiameseNetwork().cuda()
        self.criterion = SoftMaxLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        self._init_metrics()

    def _init_metrics(self):
        self.counter, self.test_counter = [], []
        self.loss_history, self.test_loss_history = [], []
        self.acc_history, self.test_acc_history = [], []
        self.iteration_number, self.test_iteration_number = 0, 0

    def train(self):
        for epoch in range(0, Config.train_number_epochs):
            self.train_epoch(epoch)
            self.test_epoch()
        self.finalise()

    def eval(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)

        self.model.load_state_dict(state_dict['model'])
        self.criterion.load_state_dict(state_dict['criterion'])

        self.test_epoch()

    def train_epoch(self, epochNumber):
        # set to training mode
        self.model.train()
        self.criterion.train()

        for i, data in enumerate(self.train_dataloader, 0):
            img0, img1, label, _, _ = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            self.optimizer.zero_grad()
            output1, output2 = self.model(img0, img1)
            loss_contrastive = self.criterion(output1, output2, label)
            loss_contrastive.backward()
            self.optimizer.step()
            if i % 10 == 0:
                acc = self.criterion.get_accuracy(self.criterion.pred, label)
                print("Epoch number={}, Current loss={}, accuracy={}".format(epochNumber, loss_contrastive.item(), acc.item()))
                self.iteration_number += 10
                self.counter.append(self.iteration_number)
                self.loss_history.append(loss_contrastive.item())
                self.acc_history.append(acc)

    def test_epoch(self):
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_dataloader, 0):
            img0, img1, label, _, _ = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = self.model(img0, img1)
            loss_contrastive = self.criterion(output1, output2, label)
            acc = self.criterion.get_accuracy(self.criterion.pred, label)

            self.test_iteration_number += 1
            self.test_counter.append(self.test_iteration_number)
            self.test_loss_history.append(loss_contrastive.item())
            self.test_acc_history.append(acc)

    def _save_model(self, path):
        torch.save({'model': self.model.state_dict(),
                    'criterion': self.criterion.state_dict()}, path)

    def finalise(self, test_only=False):
        if test_only:
            show_plot(self.test_counter, self.test_acc_history)
            return

        self._save_model(Config.checkpoint_path)
        show_plot(self.counter, self.loss_history)
        show_plot(self.counter, self.acc_history)
        show_plot(self.test_counter, self.test_acc_history)

    def _dump_embeddings(self, feat1, q_id, s_id, same, posfix=''):
            # create embeddings for t-SNE tensorboard projector visualisation.
            # https://projector.tensorflow.org/ - append Q_id and Same as header for labels.tsv
            filename = '../viz/features_{}.tsv'.format(posfix)
            filename_label = '../viz/labels_{}.tsv'.format(posfix)
            with open(filename, 'a+') as embed_file, open(filename_label, 'a+') as label_file:
                for i in range(feat1.shape[0]):
                    embedding = feat1[i].squeeze().cpu().detach().numpy()
                    embedding_str = ''.join(["{:.1f}".format(num) + '\t' for num in embedding])
                    embed_file.write(embedding_str + '\n')
                    label_file.write(
                        str(q_id[i].cpu().detach().numpy()) + '\t' + str(s_id[i].cpu().detach().numpy()) + '\t' + str(
                            same[i].cpu().detach().numpy()[0]) + '\n')

    def save_embeddings(self, postfix=''):

        for i, data in enumerate(self.train_dataloader, 0):
            img0, img1, label, q_id, s_id = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = self.model(img0, img1)
            self._dump_embeddings(output1, q_id + 10, s_id + 10, label, postfix)

        for i, data in enumerate(self.test_dataloader, 0):
            img0, img1, label, q_id, s_id = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = self.model(img0, img1)
            self._dump_embeddings(output1, q_id, s_id, label, postfix)


def train():
    trainer = Trainer()
    trainer.train()
    trainer.finalise()
    # trainer.save_embeddings(postfix='softmax')

def eval():
    trainer = Trainer()
    trainer.eval(Config.checkpoint_path)
    trainer.finalise(test_only=True)

if __name__ == "__main__":
    train()
