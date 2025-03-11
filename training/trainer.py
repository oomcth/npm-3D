import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union, Callable
from models.base_model import BaseModel
from training.callbacks import BaseCallback, ModelCheckpoint, EarlyStopping
from utils.logger import Logger


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        config: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.logger = logger

        self.model.to(self.device)

        self.epoch = 0
        self.best_score = float('inf')
        self.metrics = {}

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # For classification
            if outputs.size()[-1] > 1:  # Multi-class
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if batch_idx % 20 == 0 and self.logger:
                self.logger.log({
                    "training_loss_step": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }

        return metrics

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # For classification
                if outputs.size()[-1] > 1:  # Multi-class
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        metrics = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(epochs):
            self.epoch = epoch

            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate(val_loader)

            # Update metrics
            self.metrics = {**train_metrics, **val_metrics}

            epoch_time = time.time() - start_time
            self.metrics["epoch_time"] = epoch_time

            # Log metrics
            if self.logger:
                self.logger.log(self.metrics, epoch=epoch)

            # Print progress
            print(f"Epoch: {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                  f"Val Acc: {val_metrics['val_acc']:.2f}% | "
                  f"Time: {epoch_time:.2f}s")

            # Step the learning rate scheduler if it exists
            if self.scheduler:
                self.scheduler.step()

            # Execute callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self)

            # Check if training should stop (e.g., early stopping)
            stop_training = False
            for callback in self.callbacks:
                if hasattr(callback, 'stop_training') and callback.stop_training:
                    stop_training = True
                    break

            if stop_training:
                print(f"Training stopped early at epoch {epoch+1}")
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.metrics
