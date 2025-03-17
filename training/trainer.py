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
import random
from tqdm import tqdm


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

        self.history = {
            "train_loss": [],
            "train_token_accuracy": [],
            "train_perplexity": [],
            "val_loss": [],
            "val_token_accuracy": [],
            "val_perplexity": [],
            "learning_rate": [],
            "epoch_time": []
        }

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, desc="trainning"):
            inputs = batch["points"].to(self.device)

            new_prompts = []
            new_answers = []

            for prompt, target in zip(batch["prompt"], batch["answer"]):
                random_index = random.randint(0, len(target) - 1)
                new_prompt = f"{prompt} : {target[:random_index]}"
                new_answer = target[random_index]
                new_prompts.append(new_prompt)
                new_answers.append(new_answer)

            self.optimizer.zero_grad()

            loss, valid_tokens = self.model(inputs, new_prompts, new_answers, self.criterion)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            correct_tokens += valid_tokens
            total_tokens += inputs.size(0) if len(inputs.size()) == 3 else 1

            if batch_idx % 20 == 0 and self.logger:
                self.logger.log({
                    "training_loss_step": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })

        train_loss = running_loss / len(train_loader)
        token_accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(train_loss)).item()

        metrics = {
            "train_loss": train_loss,
            "train_token_accuracy": token_accuracy,
            "train_perplexity": perplexity,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }

        return metrics

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        for batch_idx, batch in tqdm(enumerate(val_loader), leave=False, desc="validation"):
            inputs = batch["points"].to(self.device)

            new_prompts = []
            new_answers = []

            for prompt, target in zip(batch["prompt"], batch["answer"]):
                random_index = random.randint(0, len(target) - 1)
                new_prompt = f"{prompt} : {target[:random_index]}"
                new_answer = target[random_index]
                new_prompts.append(new_prompt)
                new_answers.append(new_answer)

            loss, valid_tokens = self.model(inputs, new_prompts, new_answers, self.criterion)

            running_loss += loss.item()

            correct_tokens += valid_tokens
            total_tokens += inputs.size(0) if len(inputs.size()) == 3 else 1

        train_loss = running_loss / len(val_loader)
        token_accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(train_loss)).item()

        metrics = {
            "val_loss": train_loss,
            "val_token_accuracy": token_accuracy,
            "val_perplexity": perplexity,
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

        for epoch in tqdm(range(epochs)):
            self.epoch = epoch

            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            start_time = time.time()

            train_metrics = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

            self.metrics = {**train_metrics, **val_metrics}

            epoch_time = time.time() - start_time
            self.metrics["epoch_time"] = epoch_time

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_token_accuracy"].append(train_metrics["train_token_accuracy"])
            self.history["train_perplexity"].append(train_metrics["train_perplexity"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_token_accuracy"].append(val_metrics["val_token_accuracy"])
            self.history["val_perplexity"].append(val_metrics["val_perplexity"])
            self.history["learning_rate"].append(train_metrics["learning_rate"])
            self.history["epoch_time"].append(epoch_time)

            if self.logger:
                self.logger.log(self.metrics, epoch=epoch)

            print(f"Epoch: {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Train Token Acc: {train_metrics['train_token_accuracy']:.2f}% | "
                  f"Val Token Acc: {val_metrics['val_token_accuracy']:.2f}% | "
                  f"Train PPL: {train_metrics['train_perplexity']:.2f} | "
                  f"Val PPL: {val_metrics['val_perplexity']:.2f} | "
                  f"Time: {epoch_time:.2f}s")

            if self.scheduler:
                self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

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
