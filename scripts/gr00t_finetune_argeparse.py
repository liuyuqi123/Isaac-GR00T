import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import argparse

import torch
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
# from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import get_lora_model


os.environ["WANDB_API_KEY"] = "7ef4d13084e52dd6c6741cb793a834deb7035af0"  # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.8'


def parse_args():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Configuration for GR00T model fine-tuning.")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="/tmp/gr00t", help="Directory to save model checkpoints.")
    parser.add_argument("--data_config", type=str, default="gr1_arms_only", help="Data configuration name.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU for training.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of training steps.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training.")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving checkpoints.")

    # Model parameters
    parser.add_argument("--base_model_path", type=str, default="nvidia/GR00T-N1-2B", help="Base model path.")
    parser.add_argument("--tune_llm", action="store_true", help="Fine-tune the language model backbone.")
    parser.add_argument("--tune_visual", action="store_true", help="Fine-tune the vision tower.")
    parser.add_argument("--tune_projector", action="store_true", help="Fine-tune the projector.")
    parser.add_argument("--tune_diffusion_model", action="store_true", help="Fine-tune the diffusion model.")
    parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint.")

    # Advanced training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for training.")
    parser.add_argument("--lora_rank", type=int, default=0, help="Rank for the LORA model.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha value for the LORA model.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for the LORA model.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Where to report training metrics.")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment", help="Embodiment tag for training.")
    parser.add_argument("--video_backend", type=str, default="decord", help="Video backend to use for training.")

    return parser.parse_args()


def main(args):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[args.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader
    train_dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=args.video_backend,
    )

    # ------------ step 2: load model ------------
    # model = GR00T_N1.from_pretrained(
    #     pretrained_model_name_or_path=args.base_model_path,
    #     tune_llm=args.tune_llm,
    #     tune_visual=args.tune_visual,
    #     tune_projector=args.tune_projector,
    #     tune_diffusion_model=args.tune_diffusion_model,
    # )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.base_model_path,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion_model,
    )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if args.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        # evaluation_strategy="no",  # deprecated API??
        eval_strategy="no",
        save_total_limit=8,
        report_to=args.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=args.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using argparse
    args = parse_args()

    # Print the parsed arguments
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        args.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({args.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert args.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {args.num_gpus} GPUs")

    if args.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(args)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(args)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]

            # Convert args to command line arguments
            for key, value in vars(args).items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)