import argparse
import os
import torch
from utils.logger import Logger
import logging
from pathlib import Path
from data.dataset import create_data_loaders, create_train_test_val_datasets
from data.preprocessing import PreprocessingPipeline
from data.augmentation import create_aug
from models.model import Lidar_LLM
from training.trainer import Trainer
from inference.predictor import Predictor
from utils.logger import Logger
from utils.visualization import Visualizer
from data.preprocessing import StandardScaler


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Plateforme d'entraînement de Lidar LLM")

    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"],
                        default="train", help="Mode d'exécution")
    parser.add_argument("--data_path", type=str, default="random",
                        help="Chemin vers les données brutes")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Répertoire pour sauvegarder les résultats")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers un point de contrôle pour reprendre l'entraînement ou l'inférence")
    parser.add_argument("--debug", action="store_true",
                        help="Activer le mode débogage")

    return parser.parse_args()


def main():
    args = parse_arguments()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = Logger("outputs", "main")
    logger.info("Démarrage de la plateforme LidarLLM")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement et préparation des données")
    data_loaders = create_data_loaders(*create_train_test_val_datasets("data/data"))
    logger.info("Initialisation du modèle")
    model = Lidar_LLM()

    if args.checkpoint:
        logger.info(f"Chargement du point de contrôle depuis {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    visualizer = Visualizer()

    if args.mode == "train":
        logger.info("Démarrage de l'entraînement du modèle")
        trainer = Trainer(model, torch.nn.CrossEntropyLoss(),
                          torch.optim.AdamW(model.parameters(), lr=1e-4),
                          "cuda" if torch.cuda.is_available() else "mps"
                          if torch.mps.is_available() else "cpu",
                          None)

        model = trainer.fit(data_loaders[0], data_loaders[2], 2)

        logger.info("Évaluation du modèle après entraînement")
        pass

        logger.info("Génération des visualisations d'entraînement")
        visualizer.plot_training_history(trainer.history)

    elif args.mode == "evaluate":
        pass

    elif args.mode == "predict":
        logger.info("Génération de prédictions")
        predictor = Predictor(model,
                              "cuda" if torch.cuda.is_available() else "mps"
                              if torch.mps.is_available() else "cpu"
                              )
        predictions = predictor.predict(data_loaders[1])

        predictor.save_predictions(predictions, os.path.join(args.output_dir, "predictions"))

        print(predictions)

    logger.info(f"Exécution terminée. Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
