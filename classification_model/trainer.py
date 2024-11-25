import datetime 
import json 
import os 

import numpy as np 
import matplotlib 

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, SubsetRandomSampler 
from sklearn.model_selection import train_test_split 
import wandb 
import time 

from classification_model.dataloader import Wrench2ContactTypeDataset 
from classification_model.arbitrary_mlp import MLP 

class Wrench2ContactTypeTrainer:

    def __init__(self, training_folder, training_param_dict): 
        self.training_folder = training_folder 
        self.training_param_dict = training_param_dict 

        self.seed = training_param_dict["seed"] 
        self.device = training_param_dict["device"] 
        if self.device == "cuda" and not torch.cuda.is_available(): 
            self.device = "cpu" 
            print("CUDA not available, using CPU instead") 
        
        N_classes = 132 # FIXME: there's probably a cleaner way to make this a function of the data input 
        self.model = MLP([7, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, N_classes], flag_classification=False).to(self.device)  

        self.batch_size = training_param_dict["batch_size"]
        self.num_epochs = training_param_dict["epochs"]
        self.learning_rate = training_param_dict["learning_rate"]
        self.weight_decay = training_param_dict["weight_decay"]
        self.log_freq = training_param_dict["log_freq"]
        self.dataset_filename = training_param_dict["dataset_file"]
        self.save_model = training_param_dict["save_model"]
        self.exp_name = training_param_dict["name"] 

        self.criterion = nn.CrossEntropyLoss().to(self.device) 

        # self.network_loss_func = nn.MSELoss().to(self.device) # FIXME: update loss for classification 

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 

        self.train_dataloader = torch.utils.data.DataLoader(
            Wrench2ContactTypeDataset(self.dataset_filename), 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

    def train(self): 
        print('Beginning training...')
        
        n = 0 
        iters, losses = [], [] 
        iters_sub, train_acc, val_acc = [], [], []  

        # save start time 
        start_time = datetime.datetime.now() 

        for epoch in range(self.num_epochs): 
            print(f"Epoch: {epoch}")
            for xs, ts in self.train_dataloader: 
                # if len(ts) != self.batch_size: 
                #     print("continuing...")
                #     continue 

                xs = xs.to(self.device).float()  
                ts = ts.to(self.device).long() 
                zs = self.model.forward(xs) 

                loss = self.criterion(zs, ts) 
                self.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step() 

                iters.append(n) 
                losses.append(float(loss)/self.batch_size) 

                if n % self.log_freq == 0: 
                    iters_sub.append(n) 
                    train_acc.append(self.run_epoch(self.model, Wrench2ContactTypeDataset(self.dataset_filename))) 
                    val_acc.append(self.run_epoch(self.model, Wrench2ContactTypeDataset(self.dataset_filename))) 

                    print(f"Epoch: {epoch}, Iteration: {n}, Loss: {losses[-1]}, Train Acc: {train_acc[-1]}, Val Acc: {val_acc[-1]}") 

                    # print average time per iteration 
                    avg_time_per_iter = (datetime.datetime.now() - start_time).total_seconds() / (n+1)
                    print(f"Average time per iteration: {avg_time_per_iter} seconds") 

                    # print expected finish time based on average time per iteration 
                    expected_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=(self.num_epochs - epoch) * avg_time_per_iter) 
                    print(f"Expected finish time: {expected_finish_time}") 

                    print('\n\n')
                n += 1  
            

    def run_epoch(self, dataloader, training): 
        loader = torch.utils.data.DataLoader(
            training, 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

        correct, total = 0, 0   
        for xs, ts, *extra in loader:  # Unpack the extra values (if any)
            xs = xs.to(self.device).float()   
            ts = ts.to(self.device).long()   # Ensure the target labels are on the correct device
            zs = self.model.forward(xs)  # Get logits from the model
            pred = zs.max(1, keepdim=True)[1]  # Get the predicted class
            correct += pred.eq(ts.view_as(pred)).sum().item()  # Compute the number of correct predictions
            total += int(ts.shape[0])  # Total number of samples

        return correct / total  # Return the accuracy for this epoch

