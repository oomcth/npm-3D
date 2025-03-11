import os
import yaml
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union


@dataclass
class DataConfig:
    data_dir: str
    train_path: str
    val_path: str
    test_path: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    augmentation: bool = True


@dataclass
class ModelConfig:
    model_type: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout_rate: float = 0.1
    activation: str = "relu"


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    early_stopping: bool = True
    patience: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        data_config = DataConfig(**config_dict.pop("data"))
        model_config = ModelConfig(**config_dict.pop("model"))
        training_config = TrainingConfig(**config_dict.pop("training"))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            **config_dict
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "data": self.data.__dict__,
                "model": self.model.__dict__,
                "training": self.training.__dict__,
                "seed": self.seed,
                "device": self.device,
                "log_dir": self.log_dir,
            }, f, indent=2)
