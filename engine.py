# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import datetime

import torch
import torch.nn as nn

from evaluate import xsim_accuracy
from utils import MetricLogger, SmoothedValue, is_main_process
import utils


def train_one_epoch(
    model,
    args,
    optimizer,
    data_loader,
    scaler,
    device,
    lr_scheduler,
    wandb_run,
    test_data_loader,
    tokenizer,
    source_masking_generator,
    target_masking_generator,
    start_step: int = 0,
    epoch: int = 0,
    saving_frequency: int = 1500,
    testing_frequency: int = 20000,
    save_model_checkpoint: int = 50000,
):
    """
    Training logic.
    This method also saves checkpoints and logs training metrics.

    Args:
        - saving_frequency: The frequency which which we save the model, random states, and everything required to restart training (useful for preemtions).
        - testing_frequency: The frequency which which we test the model.
        - save_model_checkpoint: The frequency with which we save the model's weights, for downstream tasks and later use.
    """
    model.train()
    optimizer.zero_grad()

    total_batch_size = args.batch_size * utils.get_world_size()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    sub_epoch_start_time = time.time()
    cummulative_idx = start_step
    header = f"Epoch: [{epoch}]"
    for idx, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=args.mixed_precision_training):
            outputs = model(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                src_labels=batch["src_labels"],
                trg_input_ids=batch["trg_input_ids"],
                trg_attention_mask=batch["trg_attention_mask"],
                trg_labels=batch["trg_labels"],
                return_dict=True,
                output_hidden_states=True,
                stage="train",
            )
            loss = outputs["loss"]
            loss = loss / args.number_of_iterations_to_accumulated_gradients

        if torch.isnan(loss):
            print("\n\nNAN LOSS\n\n")
            breakpoint()

        scaler.scale(loss).backward()
        if (idx + 1) % args.number_of_iterations_to_accumulated_gradients == 0:
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # All after is for logging and saving purposes
        utils.add_training_data_to_metric_logger(
            lr=optimizer.param_groups[0]["lr"],
            outputs=outputs,
            metric_logger=metric_logger,
        )
        if not args.no_wandb and is_main_process():
            utils.add_training_data_to_wandb(
                lr=optimizer.param_groups[0]["lr"],
                outputs=outputs,
                total_batch_size=total_batch_size,
            )

        # Update the LR scheduler
        lr_scheduler.step()

        # Save current model and "training state"
        if (cummulative_idx % saving_frequency == 0) and args.output_dir:
            utils.save_model_and_random_states(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                cummulative_idx,
                args.no_wandb,
                args.output_dir,
                wandb_run,
                source_masking_generator,
                target_masking_generator,
                scaler,
            )

        if (cummulative_idx + 1) % testing_frequency == 0:
            print(
                f"Sub-Epoch time: {str(datetime.timedelta(seconds=int(time.time() - sub_epoch_start_time)))}"
            )
            sub_epoch_start_time = time.time()
            evaluate(
                model=model,
                data_loader=test_data_loader,
                args=args,
                device=device,
                tokenizer=tokenizer,
            )
            optimizer.zero_grad()
            if is_main_process():
                utils.save_model(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    idx,
                    args.no_wandb,
                    args.output_dir,
                    wandb_run,
                    cummulative_idx,
                )
            model.train()

        if (cummulative_idx + 1) % save_model_checkpoint == 0:
            if is_main_process():
                utils.save_model(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    idx,
                    args.no_wandb,
                    args.output_dir,
                    wandb_run,
                    cummulative_idx,
                )

        # This cummulative_idx has the total steps done in total during training
        # The idx only has the current steps done since start or preemption.
        cummulative_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(
    model, data_loader, args, device, print_freq=100, log_suffix=""
):
    """
    Evalution logic.
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    with torch.inference_mode():
        for idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Get results from masked inputs: We get the losses
            outputs = model(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                src_labels=batch["src_labels"],
                trg_input_ids=batch["trg_input_ids"],
                trg_attention_mask=batch["trg_attention_mask"],
                trg_labels=batch["trg_labels"],
                return_dict=True,
                output_hidden_states=True,
                stage="test",
            )
            utils.add_testing_data_to_metric_logger(
                outputs=outputs,
                metric_logger=metric_logger,
            )

    # After going through all the test data, we evaluate on xSIM
    xsim_results = xsim_accuracy(
        batch_size=args.batch_size,
        device=device,
        model=model.module if args.distributed else model,
        model_name=args.encoder,
        languages=args.flores_200_src_languages,
        block_eff_attention=not args.dont_use_block_efficient_attention,
        flores_200_base_path=args.flores_200_base_path,
    )
    utils.add_reporting_metrics__data_to_metric_logger(
        metric_logger=metric_logger,
        xsim_results=xsim_results,
    )

    metric_logger.synchronize_between_processes()
    utils.log_final_test_results(
        header=header,
        metric_logger=metric_logger,
        outputs=outputs,
        xsim_results=xsim_results,
    )
    if not args.no_wandb and is_main_process():
        utils.log_final_test_results_to_wandb(
            metric_logger=metric_logger,
            outputs=outputs,
            xsim_results=xsim_results,
        )

    logs = {}
    for k, meter in metric_logger.meters.items():
        logs[k] = meter.global_avg
    return logs
