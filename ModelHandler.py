import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import *
from utils import *
from models import *


# train_dataset = TASDataset('tas500v1.1')
# val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
# test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')
def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases

def get_model(model_name):
    model = None
    if model_name == "FCN":
        model = FCN.FCN(n_class=10)
    elif model_name == "Descent_ResNet":
        model = Descent_ResNet.Descent_ResNet(n_class=10)
    elif model_name == "Awful_ResNet":
        model = Awful_ResNet.Awful_ResNet(n_class=10)
    elif model_name == "HAPNet":
        model = HAPNet.HAPNet(n_class=10)
    elif model_name == "UNet":
        model = UNet.UNet(n_class=10)
    else:
        print("ERROR: The model name doesnt match")
    return model

class ModelHandler:
    def __init__(self, config_name, verbose=False):
        self.config = load_config(config_name+'.yaml')
        
        #self.model.apply(init_weights)
        self.optimizer = None
        self.criterion = None
        self.fig_count = 0
        self.verbose = verbose

        # Reading Config File
        self.lr = self.config['lr']
        self.n_class = self.config['n_class']
        self.batch_size = self.config['batch_size']

        self.model = get_model(self.config["model_name"])
        assert self.model is not None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # determine which device to use (gpu or cpu)
        self.model = self.model.to(self.device)  # transfer the model to the device

        # Data Augmentation
        crop = self.config['crop']
        hf = self.config['horizontal_flip']
        vf = self.config['vertical_flip']
        jitter = self.config['color_jitter']

        # Load Dataset
        train_dataset = TASDataset('tas500v1.1', crop=crop,
                                   horizontal_flip=hf,
                                   vertical_flip=vf,
                                   color_jitter=jitter)
        val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
        test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        if self.config['weighted']:
            total_labels_counts = np.zeros(self.n_class)
            for iter, (inputs, labels) in enumerate(self.train_loader):
                labels, counts = torch.unique(labels.view(-1), return_counts =True)
                total_labels_counts[labels.numpy()] += counts.numpy()

            min_class= np.max(total_labels_counts)
            self.class_weights = torch.from_numpy((min_class/np.log(total_labels_counts)).astype(np.float32))
        else:
            self.class_weights = torch.from_numpy(np.ones(self.n_class).astype(np.float32))

        self.class_weights = self.class_weights.to(self.device)
        self.best_model = None
        self.set_objective(config['objective'])
        self.set_optimizer(config['optimizer'])

        # output to the directory same as the name of the model!
        self.outdir = self.config['experiment_name']
        if not os.path.isdir(self.outdir):
            try:
                os.mkdir('./' + self.outdir)
            except Exception as e:
                print(e)

    def set_optimizer(self, optimizer):
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def set_objective(self, criterion):
        if criterion == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def set_model(self, model):
        if self.model is not None:
            del self.model
        self.model = model
        self.set_optimizer(self.config['optimizer'])

    def train(self, epochs):
        print("Training Started....")
        best_iou_score = 0.0
        train_losses = []
        val_loss = []
        val_miou = []
        val_acc = []
        val_class_iou = []

        for epoch in range(epochs):
            ts = time.time()
            train_loss_in_epoch = []
            for iter, (inputs, labels) in enumerate(self.train_loader):
                # reset optimizer gradients
                self.optimizer.zero_grad()

                # both inputs and labels have to reside in the same device as the model's
                inputs = inputs.to(self.device)  # transfer the input to the same device as the model's # batch_size, 3, 384, 768
                labels = labels.to(self.device)  # transfer the labels to the same device as the model's # batch_size, 384, 768

                outputs = self.model(inputs)  # we will not need to transfer the output, it will be automatically in the same device as the model's!

                loss = self.criterion(outputs, labels.long())  # calculate loss
                train_loss_in_epoch.append(loss.item())

                # backpropagate
                loss.backward()

                # update the weights
                self.optimizer.step()

                if self.verbose and iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))


            current_miou_score, mean_val_acc, mean_val_loss, class_iou = self.validate()

            mean_train_loss = np.mean(train_loss_in_epoch)
            val_loss.append(mean_val_loss)
            val_miou.append(current_miou_score)
            val_acc.append(mean_val_acc)
            val_class_iou.append(class_iou)
            train_losses.append(mean_train_loss)

            print("Epoch {}, loss: {}, val_iou: {}, val_acc: {}, time {}".format(epoch, mean_train_loss, current_miou_score, mean_val_acc,  time.time() - ts))


            if self.verbose:
                print(f'==== Epoch {epoch} ======')
                print(f"Train Loss is {mean_train_loss}")
                print(f"Val Loss is {mean_val_loss}")
                print(f"IoU is {current_miou_score}")
                print(f"Pixel acc is {mean_val_acc}")

            if current_miou_score > best_iou_score:
                best_iou_score = current_miou_score
                # save the best model
                torch.save(self.model.state_dict(), os.path.join(self.config['experiment_name'] , 'best_model.pt'))

        self.plot(train=train_losses, validation=val_loss, title="Loss")
        self.plot(validation=val_miou, title="IoU Score")
        
        # save training curves every epoch. If need to terminate early should be fine too
        perf = pd.DataFrame([train_losses, val_loss, val_miou, val_acc], 
                     index = ['train_loss', 'val_loss', 'val_avg_miou', 'val_acc'])
        class_iou_df = pd.DataFrame(val_class_iou)

        perf.to_csv(os.path.join(self.outdir,'perf.csv'))
        class_iou_df.to_csv(os.path.join(self.outdir,'iou.csv'))
        
        print(f'csv output to {self.outdir}')


        return train_losses, val_loss, val_miou, val_acc

    def validate(self):
        self.model.eval()  # Put in eval mode (disables batchnorm/dropout) !

        losses = []
        mean_iou_scores = []
        accuracy = []
        class_specific_iou = []

        with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

            for iter, (input, label) in enumerate(self.val_loader):
                # both inputs and labels have to reside in the same device as the model's
                input = input.to(self.device)  # transfer the input to the same device as the model's
                label = label.to(self.device)  # transfer the labels to the same device as the model's

                output = self.model(input)

                loss = self.criterion(output, label.long())  # calculate the loss
                losses.append(loss.item())  # call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work

                pred = torch.argmax(self.model.forward(input), dim=1)  # Make sure to include an argmax to get the prediction from the outputs of your model ##### NOT SURE IF DIMESION IS RIGHT

                iou_for_each_class = iou(pred, label, self.n_class)
                class_specific_iou.append(iou_for_each_class)
                mean_iou_scores.append(np.nanmean(iou(pred, label, self.n_class)))  # Complete this function in the util, notice the use of np.nanmean() here

                accuracy.append(pixel_acc(pred, label))  # Complete this function in the util

        mean_loss = np.mean(losses)
        mean_iou_scores_all = np.mean(mean_iou_scores)
        mean_pixel_acc = np.mean(accuracy)
        class_iou = np.nanmean(np.stack(class_specific_iou), axis=0)

        self.model.train()  # DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

        return mean_iou_scores_all, mean_pixel_acc, mean_loss, class_iou

    def test(self):
        # TODO: load the best model and complete the rest of the function for testing
        self.model.load_state_dict(torch.load(os.path.join(self.config['experiment_name'], 'best_model.pt')))
        self.model.eval()  # Put in eval mode (disables batchnorm/dropout) !

        mean_iou_scores = []
        pixel_accuracy = []
        class_specific_iou = []

        with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
            for iter, (input, label) in enumerate(self.test_loader):
                # both inputs and labels have to reside in the same device as the model's
                input = input.to(self.device)  # transfer the input to the same device as the model's
                label = label.to(self.device)  # transfer the labels to the same device as the model's
                pred = torch.argmax(self.model.forward(input), dim=1)  # Make sure to include an argmax to get the prediction from the outputs of your model ##### NOT SURE IF DIMESION IS RIGHT
                
                iou_for_each_class = iou(pred, label, self.n_class)
                class_specific_iou.append(iou_for_each_class)
                mean_iou_scores.append(np.nanmean(iou_for_each_class))  # Complete this function in the util, notice the use of np.nanmean() here
                pixel_accuracy.append(pixel_acc(pred, label))  # Complete this function in the util

        mean_iou_scores_all = np.mean(mean_iou_scores)
        mean_pixel_acc = np.mean(pixel_accuracy)
        class_iou = np.nanmean(np.stack(class_specific_iou), axis = 0)
        
        test_perf = pd.DataFrame([[mean_iou_scores_all, mean_pixel_acc]+list(class_iou)],
                                columns = ['mean IoU', 'mean Pixel accuracy']+[f'class_{i}' for i in range(len(class_iou))])
        test_perf.to_csv(os.path.join(self.outdir,'test_perf.csv'))
        
        self.model.train()  # DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
        return mean_iou_scores_all, mean_pixel_acc, class_iou

    def plot(self, train=None, validation=None, title=None, ylabel=None):
        self.fig_count += 1
        if train:
            plt.plot(train, color='y', label='Train')
        if validation:
            plt.plot(validation, color='g', label='Validation')
        plt.title(title)
        plt.legend()
        plt.xlabel('Epochs')
        if ylabel is None:
            ylabel = title
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.config['experiment_name'] , str(self.fig_count) + '.png'))
        plt.show()
