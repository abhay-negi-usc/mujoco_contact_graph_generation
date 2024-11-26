import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import wandb
import time

from loguru import logger # TODO(abhay) : install this library

class BaseContactStateTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(self, training_folder, training_param_dict):
        self.training_folder = training_folder
        self.training_param_dict = training_param_dict

        # Initialize wandb
        wandb.init(
            project=training_param_dict.get("wandb_project", "contact-state-prediction"),
            name=training_param_dict["name"],
            config=training_param_dict
        )

        self.setup_common_params(training_param_dict)
        self.setup_dataloaders()
        
    def setup_common_params(self, params):
        self.seed = params["seed"]
        self.device = params["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.debug("CUDA not available, using CPU instead")

        self.batch_size = params["batch_size"]
        self.num_epochs = params["epochs"]
        self.learning_rate = params["learning_rate"]
        self.weight_decay = params["weight_decay"]
        self.log_freq = params["log_freq"]
        self.dataset_filename = params["dataset_file"]
        self.save_model = params["save_model"]
        self.exp_name = params["name"]
        self.val_split = params.get("val_split", 0.2)  # TODO(abhay): set as needed.

        self.criterion = nn.BCELoss().to(self.device)  # TODO(abhay): change as needed.
        
    def setup_dataloaders(self):
        """Setup train and validation dataloaders with proper splitting."""
        dataset = self._get_dataset_class()(self.dataset_filename)
        
        # Create train/val splits
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.val_split * dataset_size))
        
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=train_sampler
        )
        
        self.val_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=val_sampler
        )
    
    def validate(self):
        """Run validation epoch and return metrics"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xs, ts in self.val_dataloader:
                xs = xs.to(self.device).float()
                ts = ts.to(self.device).float()
                
                zs = self.model(xs)
                loss = self.criterion(zs, ts)
                
                val_loss += loss.item() * xs.size(0)
                correct += torch.round(zs).eq(ts).sum().item()
                total += int(ts.shape[0] * ts.shape[1])
        
        val_loss = val_loss / len(self.val_dataloader.dataset)
        val_acc = correct / total
        
        return val_loss, val_acc

    def train(self):
        """Training loop with validation and wandb logging"""
        logger.debug('Beginning training...')
        
        n = 0
        best_val_acc = 0
        start_time = datetime.datetime.now()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for xs, ts in self.train_dataloader:
                xs = xs.to(self.device).float()
                ts = ts.to(self.device).float()
                
                zs = self.model(xs)
                loss = self.criterion(zs, ts)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * xs.size(0)
                
                if n % self.log_freq == 0:
                    # Run validation
                    val_loss, val_acc = self.validate()
                    train_loss = epoch_loss / ((n + 1) * self.batch_size)
                    train_acc = self.compute_accuracy(self.train_dataloader)
                    
                    # Log metrics
                    wandb.log({
                        "epoch": epoch,
                        "iteration": n,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "avg_time_per_iter": (datetime.datetime.now() - start_time).total_seconds() / (n + 1)
                    })
                    
                    logger.debug(f"Epoch: {epoch}, Iteration: {n}")
                    logger.debug(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    logger.debug(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    logger.debug(f"Avg time per iter: {(datetime.datetime.now() - start_time).total_seconds() / (n + 1):.4f}s")
                    logger.debug('\n')
                    
                    # Save best model
                    if val_acc > best_val_acc and self.save_model:
                        best_val_acc = val_acc
                        self.save_checkpoint({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_acc': val_acc,
                        })
                
                n += 1
    
    def compute_accuracy(self, dataloader):
        """Compute accuracy for a given dataloader."""
        correct = 0
        total = 0
        self.model.eval()
        
        with torch.no_grad():
            for xs, ts in dataloader:
                xs = xs.to(self.device).float()
                ts = ts.to(self.device).float()
                zs = self.model(xs)
                correct += torch.round(zs).eq(ts).sum().item()
                total += int(ts.shape[0] * ts.shape[1])
        
        return correct / total
    
    def save_checkpoint(self, state):
        """Save model checkpoint."""
        filename = os.path.join(self.training_folder, f'{self.exp_name}_best.pth')
        torch.save(state, filename)
        wandb.save(filename)  # Save to wandb as well


class Pose2ContactStateTrainer(BaseContactStateTrainer):
    def __init__(self, training_folder, training_param_dict):
        super().__init__(training_folder, training_param_dict)
        
        N_classes = 96
        self.model = BinaryVectorPredictor([7, 128, 128, N_classes]).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
    
    def _get_dataset_class(self):
        return Pose2ContactStateDataset


class Wrench2ContactStateTrainer(BaseContactStateTrainer):
    def __init__(self, training_folder, training_param_dict):
        super().__init__(training_folder, training_param_dict)
        
        N_classes = 39
        self.model = BinaryVectorPredictor([11, 128, 128, N_classes]).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
    
    def _get_dataset_class(self):
        return Wrench2ContactStateDataset