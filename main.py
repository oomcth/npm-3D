import argparse
import os
from utils import logger as log
import logging
from pathlib import Path
from config.config import Config
from data.dataset import create_data_loaders, create_train_test_val_datasets
from data.preprocessing import PreprocessingPipeline
from data.augmentation import create_aug
from models.model import Lidar_LLM
from training.trainer import Trainer
from inference.predictor import Predictor
from utils.logger import Logger
from utils.visualization import Visualizer


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
    logger = log("main.txt", "main")
    logger.info("Démarrage de la plateforme LidarLLM")

    config_manager = Config()
    config = config_manager.get_config()
    logger.info(f"Configuration chargée depuis {args.config}")

    config.update({
        "model_type": args.model_type,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "checkpoint": args.checkpoint,
        "debug": args.debug,
        "mode": args.mode
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement et préparation des données")
    data_loader = DatasetLoader(config)
    raw_data = data_loader.load_data()

    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess(raw_data)

    if config.get("use_augmentation", False):
        logger.info("Application de l'augmentation de données")
        augmenter = DataAugmenter(config)
        processed_data = augmenter.augment(processed_data)

    logger.info(f"Initialisation du modèle de type {args.model_type}")
    model = Lidar_LLM(config)

    if args.checkpoint:
        logger.info(f"Chargement du point de contrôle depuis {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    visualizer = Visualizer(config)

    if args.mode == "train":
        logger.info("Démarrage de l'entraînement du modèle")
        trainer = ModelTrainer(model, config)
        trained_model = trainer.train(processed_data)

        logger.info("Évaluation du modèle après entraînement")
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate(trained_model, processed_data["val"])

        logger.info("Génération des visualisations d'entraînement")
        visualizer.plot_training_history(trainer.history)
        visualizer.plot_model_performance(metrics)

    elif args.mode == "evaluate":
        logger.info("Évaluation du modèle")
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate(model, processed_data["test"])

        logger.info("Génération des visualisations d'évaluation")
        visualizer.plot_model_performance(metrics)

    elif args.mode == "predict":
        logger.info("Génération de prédictions")
        predictor = ModelPredictor(model, config)
        predictions = predictor.predict(processed_data["test"])

        predictor.save_predictions(predictions, os.path.join(args.output_dir, "predictions"))

        logger.info("Génération des visualisations de prédictions")
        visualizer.plot_predictions(predictions, processed_data["test"])

    logger.info(f"Exécution terminée. Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
