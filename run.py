import argparse
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
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--model-name",
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


def run_exp(run_type: str, exp_config: str, split_id: int, split_num: int, save_path: str, model_path: str, model_name: str, opts: None) -> None:
    if run_type == "eval":
        if model_name == 'uni-navid':
            from uninavid.mm_utils import get_model_name_from_path
            from uninavid.model.builder import load_pretrained_model
            from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from uninavid.conversation import conv_templates, SeparatorStyle
            from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
            from evt_bench.default import get_config
            from habitat.datasets import make_dataset
            from agent_uninavid import evaluate_agent

            config=get_config(exp_config)
            random.seed(config.habitat.simulator.seed)
            np.random.seed(config.habitat.simulator.seed)
            dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
            dataset_split = dataset.get_splits(split_num)[split_id]
            evaluate_agent(config, model_path, dataset_split, save_path)
        elif model_name == 'baseline':
            from evt_bench.default import get_config
            from habitat.datasets import make_dataset
            from baseline_agent import evaluate_agent

            config=get_config(exp_config)
            random.seed(config.habitat.simulator.seed)
            np.random.seed(config.habitat.simulator.seed)
            dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
            dataset_split = dataset.get_splits(split_num)[split_id]
            evaluate_agent(config, dataset_split, save_path)
        else:
            raise ValueError(f"The model name: {model_name} is not supported")
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
