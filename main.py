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
from distillation.distillation import main as distil_main
import distillation.distillation

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Plateforme d'entraînement de Lidar LLM")
    parser.add_argument("--mode", type=str, choices=["train", "distillation", "predict"],
                        default="predict", help="Mode d'exécution")
    parser.add_argument("--data_path", type=str, default="random",
                        help="Chemin vers les données brutes ('random' pour générer un dataset aléatoire)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Répertoire pour sauvegarder les résultats")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers un point de contrôle pour reprendre l'entraînement ou l'inférence")
    parser.add_argument("--debug", action="store_true",
                        help="Activer le mode débogage")
    return parser.parse_args()


def main():
    args = parse_arguments()

    logger = Logger("outputs", "main")
    logger.info("Démarrage de la plateforme LidarLLM")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode != "distillation":
        logger.info("Initialisation du modèle")
        model = Lidar_LLM()
        if args.data_path == "random":
            logger.info("Utilisation d'un dataset généré aléatoirement")
        else:
            logger.info(f"Utilisation du dataset LiDAR depuis {args.data_path}")
        train_dataset, test_dataset, val_dataset = create_train_test_val_datasets(
            model.lidar_encoder, args.data_path
        )
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
    else:
        model = None

    if args.checkpoint and model is not None:
        logger.info(f"Chargement du point de contrôle depuis {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    visualizer = Visualizer()

    if args.mode == "train":
        logger.info("Démarrage de l'entraînement du modèle")
        trainer = Trainer(
            model, 
            torch.nn.CrossEntropyLoss(),
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
            config=None
        )
        model = trainer.fit(train_loader, val_loader, epochs=2)
        
        
        caption_metrics = trainer.evaluate_captioning(val_loader)
        logger.info(f"Captioning Metrics: {caption_metrics}")
        
        #grounding_metrics = trainer.evaluate_grounding(val_loader)
        #logger.info(f"Grounding Metrics: {grounding_metrics}")
        
        #highlevel_metrics = trainer.evaluate_highlevel(val_loader)
        #logger.info(f"High-level Instruction Metrics: {highlevel_metrics}")
        
        logger.info("Génération des visualisations d'entraînement")
        visualizer.plot_training_history(trainer.history)

    elif args.mode == "distillation":
        logger.info("Démarrage de l'entraînement du modèle distillé")
        distil_main(str(output_dir))

    elif args.mode == "predict":
        logger.info("Génération de prédictions")
        predictor = Predictor(
            model,
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        predictions = predictor.predict(next(iter(val_loader)))
        print(predictions)

    logger.info(f"Exécution terminée. Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
