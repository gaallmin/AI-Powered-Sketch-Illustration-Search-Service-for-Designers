from typing import Dict, Union
import math
import copy

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support, classification_report
import torch
import torch.optim as optim
from torch import nn
import torchvision.transforms as transforms
from torchmetrics import ConfusionMatrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer():

    def __init__(
            self,
            model: nn.Module,
            optimizer,
            criterion,
            scheduler=None,
            device: torch.device = torch.device('cpu'),
            ):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.trained_epoch = 0
        self._results = []
        self._time_spent = timedelta(0)
        self._time_prev_epoch = datetime.now()
        self._best_val_loss = math.inf

    @property
    def results(self):
        return self._results

    def train(
            self,
            epochs: int,
            train_loader,
            test_loader,
            autosave_params: Dict[str, Union[bool, str]] = {
                'use_autosave': True,
                'save_dir': './saved_models/test.obj'}
            ):

        for epoch in range(epochs):  # loop over the dataset multiple times

            self.trained_epoch += 1

            self.model.train()
            train_loss = 0

            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # accmulate train loss
                train_loss += loss.item()

                if self.scheduler:
                    self.scheduler.step()

            train_loss = train_loss / len(train_loader)

            self.model.eval()

            results = self.test(test_loader)
            results['train loss'] = train_loss
            self._results.append(results)

            if autosave_params['use_autosave'] \
                    and results['test loss'] < self._best_val_loss:

                self._best_val_loss = results['test loss']
                self.save(autosave_params['save_dir'])

            self.__print_results(
                    self.trained_epoch,
                    results)

            # measure previous epoch start time
            self._time_prev_epoch = datetime.now()

    @torch.no_grad()
    def test(
            self,
            test_loader) -> Dict[str, float]:

        correct = 0
        total = 0

        test_loss = 0

        mean_precision = 0
        mean_recall = 0
        mean_fscore = 0

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # To change it to numpy.ndarray
            labels = labels.cpu()
            predicted = predicted.cpu()
            precision, recall, fscore, _ = precision_recall_fscore_support(
                    labels.data,
                    predicted,
                    average='macro')

            mean_precision += precision
            mean_recall += recall
            mean_fscore += fscore

        mean_precision = mean_precision / len(test_loader)
        mean_recall = mean_recall / len(test_loader)
        mean_fscore = mean_fscore / len(test_loader)
        test_loss = test_loss / len(test_loader)

        accuracy = 100 * correct // total

        results = {
                'f1-score': mean_fscore,
                'accuracy': str(accuracy)+"%",
                'precision': mean_precision,
                'recall': mean_recall,
                'test loss': test_loss}

        return results

    @torch.no_grad()
    def f1_score(
            self,
            test_loader) -> Dict[str, float]:

        total_labels = np.array([])
        total_preds = np.array([])

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)

            # To change it to numpy.ndarray
            labels = labels.cpu().numpy()
            total_labels = np.concatenate([total_labels, labels])
            predicted = predicted.cpu().numpy()
            total_preds = np.concatenate([total_preds, predicted])

        result = classification_report(total_labels, total_preds, output_dict=True)

        print(pd.DataFrame(result))

        raise ValueError("test")

    def __print_results(
            self,
            epoch: int,
            results: Dict[str, float]):

        frame = "+----------------------------------------+"
        results_str = [
            f"| epoch:                      {str(epoch)[0:10]:>10} "]

        results_str[0] += (len(frame)-len(results_str[0])-1)*' ' + '|'

        time_spent = datetime.now() - self._time_prev_epoch
        results['time spent (epoch)'] = \
            time_spent

        self._time_spent += time_spent
        results['time spent (total)'] = \
            self._time_spent

        for result_name, result in results.items():
            results_str.append(f"| {result_name}: ")
            results_str[-1] += (30-len(results_str[-1]))*' '
            results_str[-1] += f"{str(result)[0:10]:>10} "
            results_str[-1] += (len(frame)-len(results_str[-1])-1)*' ' + '|'

        print(frame)
        for result_str in results_str:
            print(result_str)
        print(frame)

    # save class
    def save(self, saveDir: str):  # use .obj for saveDir

        save_dict = copy.deepcopy(self.__dict__)

        # belows are impossible to dump
        save_dict.pop('device')

        # save model state dict
        save_dict['modelStateDict'] \
            = save_dict['model'].state_dict()
        save_dict.pop('model')
        save_dict['optimizerStateDict'] \
            = save_dict['optimizer'].state_dict()
        save_dict.pop('optimizer')

        torch.save(save_dict, saveDir)

    def save_pth(
            self,
            saveDir: str):

        torch.save(self.model.state_dict(), saveDir)

    # Load class
    def load(self, loadDir: str):

        # Load torch model
        loadedDict = torch.load(loadDir, map_location=self.device)

        # Load state_dict of torch model, and optimizer
        try:

            self.model.load_state_dict(
                    loadedDict.pop('modelStateDict'))
            self.optimizer.load_state_dict(
                    loadedDict.pop('optimizerStateDict'))

        except ValueError:
            print(
                "No matching torch.nn.Module,"
                "please use equally shaped torch.nn.Module as you've done!")

        for key, value in loadedDict.items():
            self.__dict__[key] = value

        # measure previous epoch start time
        self._time_prev_epoch = datetime.now()

    def loss_graph(
            self,
            save_dir=None,
            ):

        train_losses = []
        test_losses = []
        for result in self._results:
            train_losses.append(result['train loss'])
            test_losses.append(result['test loss'])

        plt.plot(np.arange(len(train_losses)), train_losses, c='blue', label="Train Loss")
        plt.plot(np.arange(len(test_losses)), test_losses, c='red', label="Test Loss")
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def f1_graph(
            self,
            save_dir=None,
            ):

        f1s = []
        for result in self._results:
            f1s.append(result['f1-score'])

        plt.plot(np.arange(len(f1s)), f1s, c='blue', label="f1-score")
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('f1-score')
        plt.show()

    def confusion_matrix(
            self,
            test_loader,
            save_dir=None,
            ):

        confmat = ConfusionMatrix(num_classes=20)
        confmat_result = torch.zeros(20, 20)

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            confmat_result += confmat(predicted, labels).cpu().numpy()

        sns.set()
        sns.heatmap(confmat_result, annot=True)
        plt.show()




if __name__ == "__main__":

    from model import Model
    from load import Load

    # Use Gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()

    transformer = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((32, 32))]

    # Train loader
    train_dataloader = Load(
            transformer,
            batch_size=128)
    _, trainloader = train_dataloader("./train_val_test_dataset/train")

    # Validation loader
    valid_dataloader = Load(
            transformer,
            batch_size=128)
    _, valloader = valid_dataloader("./train_val_test_dataset/valid")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device)

    # Train
    trainer.train(
            50,
            trainloader,
            valloader)
