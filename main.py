import argparse
from dacite import from_dict

from src.deepaudiox.utils.io_utils import load_yaml
from src.deepaudiox.engine.trainer import Trainer
from src.deepaudiox.config.training_config import TrainingConfig
from src.deepaudiox.config.data_config import DataConfig
from src.deepaudiox.config.model_config import ModelConfig
from src.deepaudiox.config.optimization_config_registry import build_optimizer_config
from src.deepaudiox.config.scheduling_config_registry import build_scheduling_config
from src.deepaudiox.config.loss_config_registry import build_loss_config


def main():
    parser = argparse.ArgumentParser(description="Welcome to DeepAudioX SDK.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--train_dir", type=str, required=True, help="Path to the train folder.")
    train_parser.add_argument("--test_dir", type=str, required=True, help="Path to the test folder.")
    train_parser.add_argument("--scheduler", type=str, required=False, default="CosineAnnealingLR", help="Name of the scheduler.")
    train_parser.add_argument("--optimizer", type=str, required=False, default="ADAM", help="Name of the optimizer.")
    train_parser.add_argument("--loss_function", type=str, required=False, default="CrossEntropyLoss", help="Name of the loss function.")
    train_parser.add_argument("--model", type=str, required=True, help="Name of the model.")

    args = parser.parse_args()

    if args.command == "train": 
        training_config = TrainingConfig(
            data_config = DataConfig(
                train_dir=args.train_dir,
                test_dir=args.test_dir
            ),
            model_config = ModelConfig(
                name=args.model
            ),
            scheduling_config = build_scheduling_config(args.scheduler),
            optimization_config = build_optimizer_config(args.optimizer),
            loss_config = build_loss_config(args.loss_function)
        )

        trainer = Trainer(training_config)
        trainer.train()

    elif args.command == "evaluate":
        config_path = args.config
        config = load_yaml(config_path)
        # evaluate_model(config)

    elif args.command == "export":
        # export_model(args.model, args.output)
        pass

if __name__ == "__main__":
    main()