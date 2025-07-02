from baseline_agent import evaluate_agent
import argparse
import habitat
from habitat.datasets import make_dataset
import evt_bench
import numpy as np
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["eval", "train"],
        required=True,
        help="run type",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-num",
        type=int,
        default=7,
        required=False,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(run_type: str, exp_config: str, split_id: int, split_num: int, save_path: str, opts: None) -> None:
    config=habitat.get_config(exp_config)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_split = dataset.get_splits(split_num)[split_id]

    if run_type == "eval":
        evaluate_agent(config, dataset_split, save_path)
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
