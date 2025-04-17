import argparse
import json

from classification_model.trainer import Wrench2ContactTypeTrainer 

def main():

    parser = argparse.ArgumentParser(description="MLP model training arguments.")
    parser.add_argument(
        "-p",
        "--training_params_file",
        required=True,
        type=str,
        help="Training Params Filename",
    )

    args = parser.parse_args()c

    training_folder = "data/training" # FIXME: doesn't matter for now but should be updated later 
    training_params_file = args.training_params_file

    training_param_dict: dict = {}
    with open(training_params_file, "r") as file:
        training_param_dict = json.load(file)

    trainer = Wrench2ContactTypeTrainer(training_folder, training_param_dict)
    trainer.train()


if __name__ == "__main__":
    main()
