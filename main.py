import argparse

from src.deepaudiox.config.base_config import DataConfig, ModelConfig
from src.deepaudiox.config.loss_config_registry import build_loss_config
from src.deepaudiox.config.optimization_config_registry import build_optimizer_config
from src.deepaudiox.config.scheduling_config_registry import build_scheduling_config
from src.deepaudiox.config.training_config import TrainingConfig
from src.deepaudiox.engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Welcome to DeepAudioX SDK.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--train_dir", 
        type=str, 
        required=True, 
        help="Path to the train folder."
    )
    train_parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the output folder."
    )
    train_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the model."
    )
    train_parser.add_argument(
        "--epochs", 
        type=str, 
        required=False, 
        default=10, 
        help="Number of maximum training epochs."
    )
    train_parser.add_argument(
        "--scheduler", 
        type=str, 
        required=False, 
        default="CosineAnnealingLR", 
        help="Name of the scheduler."
    )
    train_parser.add_argument(
        "--optimizer", 
        type=str, 
        required=False, 
        default="ADAM", 
        help="Name of the optimizer."
    )
    train_parser.add_argument(
        "--loss_function", 
        type=str, 
        required=False, 
        default="CrossEntropyLoss", 
        help="Name of the loss function."
    )

    args = parser.parse_args()

    if args.command == "train": 
        training_config = TrainingConfig(
            output_dir = args.output_dir,
            data_config = DataConfig(
                train_dir=args.train_dir
            ),
            model_config = ModelConfig(),
            scheduling_config = build_scheduling_config(
                params = {
                    "name": args.scheduler, 
                    "epochs": args.epochs
                }
            ),
            optimization_config = build_optimizer_config(
                params = {
                    "name": args.optimizer
                }
            ),
            loss_config = build_loss_config(
                params = {
                    "name": args.loss_function
                }
            )
        )

        trainer = Trainer(training_config)
        trainer.train()

    elif args.command == "evaluate" or args.command == "inference":
        pass

if __name__ == "__main__":
    main()