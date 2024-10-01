# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datetime
import time

from data import (
    DataCollatorWithMaskingAndPadding,
    ParallelTranslationDataset,
    SliceableDistributedSampler,
)
from engine import evaluate, train_one_epoch
from model import MEXMA
import utils
import random

import torch
import wandb
import numpy as np
import torch.backends.cudnn as cudnn


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if utils.is_main_process() and (not os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir)

    train_dataset = ParallelTranslationDataset(
        data_file=args.train_data_file,
        hf_dataset_directory=args.hf_dataset_directory,
    )

    test_dataset = ParallelTranslationDataset(
        data_file=args.test_data_file,
        flores_200_src_languages=args.flores_200_src_languages,
        flores_200_base_path=args.flores_200_base_path,
    )

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(train_dataset) // args.batch_size // num_tasks
    total_batch_size = args.batch_size * utils.get_world_size()

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)

    if args.distributed:
        print("Loaded distributed data")
        train_sampler = SliceableDistributedSampler(
            train_dataset,
            num_replicas=num_tasks,
            rank=sampler_rank,
            shuffle=True,
            seed=args.seed,  # Use the same seed for all GPUs, and then chunk it and each GPU get its part of the data
            epoch_to_skip=-1,
        )
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
        )
    else:
        print("Not using distributed data")
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    print("Train_sampler = %s" % str(train_sampler))

    collator = DataCollatorWithMaskingAndPadding(
        encoder=args.encoder,
        max_model_context_length=args.max_model_context_length,
        src_mlm_probability=args.src_mlm_probability,
        trg_mlm_probability=args.trg_mlm_probability,
        seed=args.seed,
        epoch=0,
        rank=global_rank,
    )

    model = MEXMA(
        encoder=args.encoder,
        dont_use_block_efficient_attention=args.dont_use_block_efficient_attention,
        number_of_transformer_layers_in_head=args.number_of_transformer_layers_in_head,
        number_of_transformer_attention_heads_in_head=args.number_of_transformer_attention_heads_in_head,
        number_of_linear_layers=args.number_of_linear_layers,
        linear_layers_inputs_dims=args.linear_layers_inputs_dims,
        linear_layers_outputs_dims=args.linear_layers_outputs_dims,
        mlm_loss_weight=args.mlm_loss_weight,
        cls_loss_weight=args.cls_loss_weight,
        koleo_loss_weight=args.koleo_loss_weight,
        use_pooler=args.use_pooler,
        use_dropout_in_attention=args.use_dropout_in_attention,
        initialization_method=args.initialization_method,
    )

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = utils.configure_optimizers(model, args)
    lr_scheduler = utils.get_lr_scheduler(
        optimizer=optimizer,
        training_iterations=args.epochs * num_training_steps_per_epoch,
        lr_scheduler=args.lr_scheduler_type,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        lr_steps=args.lr_steps,
        lr_warmup_percentage=args.lr_warmup_percentage,
        lr_warmup_method=args.lr_warmup_method,
        lr_warmup_decay=args.lr_warmup_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision_training)

    start_step = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        print("Load ckpt from %s" % args.checkpoint)
        if not args.distributed:
            checkpoint_model = {
                key.replace("module.", ""): value
                for key, value in checkpoint["model"].items()
            }
        else:
            checkpoint_model = checkpoint["model"]
        model.load_state_dict(checkpoint_model)
        model.to(device)

        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"]
        start_step = checkpoint["current_batch"]

        if args.checkpoint.split("/")[-1] == "latest.pth":
            # If it's a checkpoint saved with random generators to restart training with minimum variations
            generators_checkpoint = torch.load(
                args.checkpoint[:-4] + f"_generators_{utils.get_rank()}.pth",
                map_location="cpu",
            )
            scaler.load_state_dict(generators_checkpoint["scaler"])
            torch.set_rng_state(generators_checkpoint["cpu_rng_state"])
            torch.cuda.set_rng_state(generators_checkpoint["gpu_rng_state"])
            np.random.set_state(generators_checkpoint["numpy_rng_state"])
            random.setstate(generators_checkpoint["py_rng_state"])
            collator.source_masking_generator.set_state(
                generators_checkpoint["source_masking_generator"]
            )
            collator.target_masking_generator.set_state(
                generators_checkpoint["target_masking_generator"]
            )

            if args.distributed:
                print("Loaded Sampler from checkpoint")
                train_sampler = SliceableDistributedSampler(
                    train_dataset,
                    num_replicas=num_tasks,
                    rank=sampler_rank,
                    shuffle=True,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    start_iteration=checkpoint["current_batch"],
                    epoch_to_skip=checkpoint["epoch"],
                )

        if utils.is_main_process() and (not args.no_wandb):
            run = wandb.init(
                # set the wandb project where this run will be logged
                project=args.wandb_project,
                # track hyperparameters and run metadata
                config=args,
                save_code=True,
                settings=wandb.Settings(code_dir="."),
                group=args.wandb_group,
                resume="allow",
                id=checkpoint["wandb_id"],
            )
            wandb.watch(
                model,
                criterion=model.module.mlm_loss if args.distributed else model.mlm_loss,
                log="all",
                log_freq=100,
            )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=collator,
    )

    if utils.is_main_process() and (not args.no_wandb) and (args.checkpoint is None):
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # track hyperparameters and run metadata
            config=args,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            group=args.wandb_group,
        )
        wandb.watch(
            model,
            criterion=model.module.mlm_loss if args.distributed else model.mlm_loss,
            log="all",
            log_freq=100,
        )

    if args.evaluate:
        print("\n\nONLY EVALUATING\n\n")
        evaluate(
            model=model,
            data_loader=test_dataloader,
            args=args,
            device=device,
            tokenizer=collator.tokenizer,
        )
        if utils.is_main_process() and (not args.no_wandb):
            wandb.finish()
        exit()

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
            if (args.checkpoint is None) or (epoch != args.start_epoch):
                collator.set_epoch(epoch)
                train_sampler.set_epoch(epoch)
        # Train
        epoch_start_time = time.time()
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            args=args,
            scaler=scaler,
            device=device,
            lr_scheduler=lr_scheduler,
            wandb_run=None if args.no_wandb or (not utils.is_main_process()) else run,
            saving_frequency=args.saving_frequency,
            epoch=epoch,
            test_data_loader=test_dataloader,
            tokenizer=collator.tokenizer,
            testing_frequency=args.testing_frequency,
            source_masking_generator=collator.source_masking_generator,
            target_masking_generator=collator.target_masking_generator,
            start_step=start_step,
            save_model_checkpoint=args.save_model_checkpoint,
        )
        epoch_total_time = time.time() - epoch_start_time
        total_epoch_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print(f"Epoch: [{epoch}] time: {total_epoch_time_str}")
        start_step = 0

        # Evaluate after a whole epoch
        evaluate(
            model=model,
            data_loader=test_dataloader,
            args=args,
            device=device,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if utils.is_main_process() and (not args.no_wandb):
        wandb.finish()


def get_args(add_help=True):
    def float_value_or_none(value):
        if value == "None":
            return None
        else:
            return float(value)

    def int_value_or_none(value):
        if value == "None":
            return None
        else:
            return int(value)

    def str_value_or_none(value):
        if value == "None":
            return None
        else:
            return value

    import argparse

    parser = argparse.ArgumentParser(description="MEXMA", add_help=add_help)

    # Model settings
    parser.add_argument(
        "--dont_use_block_efficient_attention",
        action="store_true",
        help="Don't use block diagonal attention with xFormers's memory efficient attention.",
    )
    parser.add_argument(
        "--max_model_context_length",
        default=200,
        type=int,
        help="Maximum number of tokens the model can take",
    )
    parser.add_argument(
        "--encoder",
        default="xlm-roberta-large",
        type=str,
        help="The encoder name to be used, currently tied to XLM-RoBERTa only",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str_value_or_none,
        help="Checkpoint file containing state dict of model, optimizer and lr_scheduler.",
    )
    parser.add_argument(
        "--number_of_transformer_layers_in_head",
        default=6,
        type=int,
        help="Number of transformer layers to use for the MLM head, only used if use_transformer_head=True",
    )
    parser.add_argument(
        "--number_of_transformer_attention_heads_in_head",
        default=8,
        type=int,
        help="Number of attention heads to use for the transformer in the MLM head, only used if use_transformer_head=True",
    )
    parser.add_argument(
        "--number_of_linear_layers",
        default=0,
        type=int,
        help="Number of layers in the MLP head, only for cls_predict",
    )
    parser.add_argument(
        "--linear_layers_inputs_dims",
        default=[],
        nargs="*",
        type=int_value_or_none,
        help="Input dims for the MLP head, only for cls_predict",
    )
    parser.add_argument(
        "--linear_layers_outputs_dims",
        default=[],
        nargs="*",
        type=int_value_or_none,
        help="Output dims for the MLP head, only for cls_predict",
    )
    parser.add_argument(
        "--use_pooler",
        default=False,
        action="store_true",
        required=False,
        help="Whether to use an additional Linear+Activation on top of the CLS embedding, or to just use the encoder output directly.",
    )
    parser.add_argument(
        "--use_dropout_in_attention",
        default=False,
        action="store_true",
        required=False,
        help="Turn on the dropout in the attention mechanism.",
    )
    parser.add_argument(
        "--initialization_method",
        default="torch_default",
        type=str,
        choices=["torch_default", "normal_dist", "xavier_uniform", "xavier_normal"],
        help="The initialization method for the head.",
    )
    parser.add_argument(
        "--mlm_loss_weight", default=1.0, type=float, help="Weight for the mlm_loss"
    )
    parser.add_argument(
        "--cls_loss_weight", default=1.0, type=float, help="Weight for the cls_loss"
    )
    parser.add_argument(
        "--koleo_loss_weight",
        default=0.01,
        type=float,
        help="Weight for the koleo loss",
    )

    # Data settings
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str_value_or_none,
        help="Train data path",
    )
    parser.add_argument(
        "--test_data_file", default=None, type=str_value_or_none, help="Test data path"
    )
    parser.add_argument(
        "--flores_200_base_path",
        default="data/flores200",
        type=str,
        help="The path to flores200 dataset",
    )
    parser.add_argument(
        "--flores_200_src_languages",
        default=["por_Latn", "spa_Latn", "fra_Latn", "deu_Latn"],
        nargs="+",
        type=str_value_or_none,
        help="Languages from the FLORES200 dataset to evaluate the model on during training.",
    )
    parser.add_argument(
        "--hf_dataset_directory",
        default=None,
        type=str_value_or_none,
        help="Path to a huggingface dataset containing all of the training data.",
    )

    # Training settings
    parser.add_argument("--batch_size", default=150, type=int, help="Batch size")
    parser.add_argument(
        "--workers", default=12, type=int, help="Number of workers to use in dataloader"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to use, GPU (cuda) or CPU"
    )
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="Starting epoch to train"
    )
    parser.add_argument(
        "--start_sub_epoch", default=0, type=int, help="Starting sub-epoch to train"
    )
    parser.add_argument(
        "--src_mlm_probability",
        default=0.4,
        type=float,
        help="MLM probability to mask a token in the source",
    )
    parser.add_argument(
        "--trg_mlm_probability",
        default=0.4,
        type=float,
        help="MLM probability to mask a token in the target",
    )
    parser.add_argument(
        "--number_of_iterations_to_accumulated_gradients",
        default=2,
        type=int,
        help="Number of iterations to accumulate gradients before performing an optimization step",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Flag to turn on torch2 compile() method",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Indicates that no training is to be done, only model evaluation.",
    )
    parser.add_argument(
        "--testing_frequency",
        default=20000,
        type=int,
        help="The frequency to evaluate, in number of steps, the model during training. Useful when running for long epochs.",
    )
    parser.add_argument(
        "--save_model_checkpoint",
        default=20000,
        type=int,
        help="The frequency to save the model checkpoint.",
    )
    parser.add_argument(
        "--saving_frequency",
        default=1500,
        type=int,
        help="The frequency to save the latest model, which overwrites the previous. Allows us to save mid-epoch.",
    )

    # Mixed precision training settings
    parser.add_argument(
        "--mixed_precision_training",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )
    parser.add_argument(
        "--clip_grad_norm",
        default=1.2,
        type=float_value_or_none,
        help="Maximum norm of the gradients during training, e.g.: 1.0",
    )

    # Weight decay
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=None,
        type=float_value_or_none,
        metavar="W",
        help="weight decay (default: 0.1)",
        dest="weight_decay",
    )

    # LR scheduler settings
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosineannealinglr",
        type=str,
        help="name of lr scheduler (default: cosineannealinglr)",
    )
    parser.add_argument(
        "--lr_step_size",
        default=8,
        type=int_value_or_none,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float_value_or_none,
        help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr_steps",
        default=[16, 22],
        nargs="+",
        type=int_value_or_none,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr_warmup_percentage",
        default=0.3,
        type=float_value_or_none,
        help="the percentage of training to warmup",
    )
    parser.add_argument(
        "--lr_warmup_method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument(
        "--lr_warmup_decay",
        default=0.1,
        type=float_value_or_none,
        help="the decay for lr",
    )

    # Logging settings
    parser.add_argument(
        "--print_freq",
        default=10,
        type=int,
        help="Print each print_freq, e.g. print_freq=10 -> print every 10 epochs",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Deactivate wandb logging"
    )
    parser.add_argument(
        "--wandb_group", default="initial", type=str, help="The wandb group to log to"
    )
    parser.add_argument(
        "--wandb_project", default="mexma", type=str, help="The wandb project to log to"
    )

    # Checkpoint settings
    parser.add_argument(
        "--output_dir", default="checkpoints/", type=str, help="path to save outputs"
    )

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)
