import argparse
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vector_prediction_model.trainer import Pose2ContactStateTrainer 
from vector_prediction_model.trainer import Wrench2ContactStateTrainer 

def main():

    parser = argparse.ArgumentParser(description="MLP model training arguments.")
    parser.add_argument(
        "-p",
        "--training_params_file",
        required=True,
        type=str,
        help="Training Params Filename",
    )

    args = parser.parse_args()

    training_folder = "data/training" # FIXME: doesn't matter for now but should be updated later 
    training_params_file = args.training_params_file

    training_param_dict: dict = {}
    with open(training_params_file, "r") as file:
        training_param_dict = json.load(file)

    # trainer = Pose2ContactStateTrainer(training_folder, training_param_dict)
    trainer = Wrench2ContactStateTrainer(training_folder, training_param_dict)
    trainer.train()


if __name__ == "__main__":
    main()
