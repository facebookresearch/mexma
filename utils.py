# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from collections import defaultdict, deque
import torch.distributed as dist
import time
import datetime
import os
import psutil
from typing import Dict
import wandb
import numpy
import random


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def get_model_parallel(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "distributed init (rank {}): {}, gpu {}, world_size: {}".format(
            args.rank, args.dist_url, args.gpu, args.world_size
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


import socket

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return self.total / 1
        else:
            return self.total / self.count

    @property
    def max(self):
        if self.count > 0:
            return max(self.deque)
        else:
            return 0

    @property
    def value(self):
        if self.count > 0:
            return self.deque[-1]
        else:
            return 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max cuda mem: {memory:.0f}",
                    "ram_percentage: {ram_usage_percentage}",
                    "cpu_percentage: {cpu_usage_percentage}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                            ram_usage_percentage=psutil.virtual_memory().percent,
                            cpu_usage_percentage=psutil.cpu_percent(),
                        )
                    )
                    torch.cuda.reset_max_memory_allocated()
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


"""
Taken from nanoGPT implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
def configure_optimizers(model, args):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": args.weight_decay if args.weight_decay else 0.0,
        },
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
    return optimizer


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_iterations: int = 5,
    lr_scheduler: str = "multisteplr",
    lr_step_size: int = 8,
    lr_gamma: float = 0.1,
    lr_steps: list = [16, 22],
    lr_warmup_percentage: int = 0,
    lr_warmup_method: str = "linear",
    lr_warmup_decay: float = 0.01,
):
    lr_warmup_epochs = int(lr_warmup_percentage * training_iterations)
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
    elif lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_iterations - lr_warmup_epochs
        )
    elif lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_gamma
        )
    elif lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_steps, gamma=lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR, CosineAnnealingLR, ExponentialLR and MultiStepLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler


def save_model_and_random_states(
    model,
    optimizer,
    lr_scheduler,
    epoch,
    idx,
    no_wandb,
    output_dir,
    wandb_run,
    source_masking_generator,
    target_masking_generator,
    scaler,
):
    if is_main_process():
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "current_batch": idx + 1,
        }
        if not no_wandb:
            checkpoint["wandb_id"] = wandb_run.id
        torch.save(checkpoint, os.path.join(output_dir, f"latest.pth"))

    checkpoint = {
        "cpu_rng_state": torch.get_rng_state(),
        "gpu_rng_state": torch.cuda.get_rng_state(),
        "numpy_rng_state": numpy.random.get_state(),
        "py_rng_state": random.getstate(),
        "source_masking_generator": source_masking_generator.get_state(),
        "target_masking_generator": target_masking_generator.get_state(),
        "scaler": scaler.state_dict(),
    }
    torch.save(
        checkpoint, os.path.join(output_dir, f"latest_generators_{get_rank()}.pth")
    )


def save_model(
    model,
    optimizer,
    lr_scheduler,
    epoch,
    idx,
    no_wandb,
    output_dir,
    wandb_run,
    cummulative_idx,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "current_batch": idx + 1,
    }
    if not no_wandb:
        checkpoint["wandb_id"] = wandb_run.id
    torch.save(
        checkpoint, os.path.join(output_dir, f"model_{epoch}_{cummulative_idx}.pth")
    )


def add_training_data_to_metric_logger(
    lr: float,
    outputs: Dict,
    metric_logger: MetricLogger,
):
    metric_logger.update(loss=outputs["loss"].item())
    metric_logger.update(lr=lr)
    if "mlm_loss" in outputs:
        metric_logger.update(mlm_loss=outputs["mlm_loss"].item())
    if "src_mlm_loss" in outputs:
        metric_logger.update(src_mlm_loss=outputs["src_mlm_loss"].item())
    if "trg_mlm_loss" in outputs:
        metric_logger.update(trg_mlm_loss=outputs["trg_mlm_loss"].item())
    if "cls_loss" in outputs:
        metric_logger.update(cls_loss=outputs["cls_loss"].item())
    if "koleo_loss" in outputs:
        metric_logger.update(koleo_loss=outputs["koleo_loss"].item())


def add_training_data_to_wandb(
    lr: float,
    outputs: Dict,
    total_batch_size: int = None,
    step: int = None,
):
    wandb_log_dict = {
        "loss": outputs["loss"].item(),
        "lr": lr,
    }
    if "src_mlm_loss" in outputs:
        wandb_log_dict["src_mlm_loss"] = outputs["src_mlm_loss"].item()
    if "trg_mlm_loss" in outputs:
        wandb_log_dict["trg_mlm_loss"] = outputs["trg_mlm_loss"].item()
    if "cls_loss" in outputs:
        wandb_log_dict["cls_loss"] = outputs["cls_loss"].item()
    if "koleo_loss" in outputs:
        wandb_log_dict["koleo_loss"] = outputs["koleo_loss"].item()
    if (step is not None) and (total_batch_size is not None):
        wandb.log(wandb_log_dict, step=step * total_batch_size)
    else:
        wandb.log(wandb_log_dict)


def add_testing_data_to_metric_logger(
    outputs: Dict,
    metric_logger: MetricLogger,
):
    metric_logger.update(test_loss=outputs["loss"].item())
    if "mlm_loss" in outputs:
        metric_logger.update(test_mlm_loss=outputs["mlm_loss"].item())
    if "src_mlm_loss" in outputs:
        metric_logger.update(test_src_mlm_loss=outputs["src_mlm_loss"].item())
    if "trg_mlm_loss" in outputs:
        metric_logger.update(test_trg_mlm_loss=outputs["trg_mlm_loss"].item())
    if "cls_loss" in outputs:
        metric_logger.update(test_cls_loss=outputs["cls_loss"].item())
    if "koleo_loss" in outputs:
        metric_logger.update(test_koleo_loss=outputs["koleo_loss"].item())
    # Bitext mining metrics
    metric_logger.update(test_accuracy=outputs["accuracy"].item())
    metric_logger.update(test_src_to_trg_accuracy=outputs["src_to_trg_accuracy"].item())
    metric_logger.update(test_trg_to_src_accuracy=outputs["trg_to_src_accuracy"].item())
    metric_logger.update(test_top3_accuracy=outputs["top3_accuracy"].item())
    metric_logger.update(
        test_src_to_trg_top3_accuracy=outputs["src_to_trg_top3_accuracy"].item()
    )
    metric_logger.update(
        test_trg_to_src_top3_accuracy=outputs["trg_to_src_top3_accuracy"].item()
    )


def add_reporting_metrics__data_to_metric_logger(
    metric_logger: MetricLogger,
    xsim_results: Dict,
):
    metric_logger.update(**xsim_results)
    xsim_average = torch.mean(torch.tensor([*xsim_results.values()]))
    metric_logger.update(xsim_average=xsim_average.item())


def log_final_test_results(
    header: str,
    metric_logger: MetricLogger,
    outputs: Dict,
    xsim_results: Dict,
):
    print(f"{header} Test Loss {metric_logger.test_loss.global_avg:.3f}")
    if "mlm_loss" in outputs:
        print(f"{header} test_mlm_loss {metric_logger.test_mlm_loss.global_avg:.3f}")
    if "src_mlm_loss" in outputs:
        print(
            f"{header} test_src_mlm_loss {metric_logger.test_src_mlm_loss.global_avg:.3f}"
        )
    if "trg_mlm_loss" in outputs:
        print(
            f"{header} test_trg_mlm_loss {metric_logger.test_trg_mlm_loss.global_avg:.3f}"
        )
    if "cls_loss" in outputs:
        print(f"{header} test_cls_loss {metric_logger.test_cls_loss.global_avg:.3f}")
    if "koleo_loss" in outputs:
        print(
            f"{header} test_koleo_loss {metric_logger.test_koleo_loss.global_avg:.3f}"
        )
    # Bitext mining metrics
    print(f"{header} test_accuracy {metric_logger.test_accuracy.global_avg:.5f}")
    print(
        f"{header} test_src_to_trg_accuracy {metric_logger.test_src_to_trg_accuracy.global_avg:.5f}"
    )
    print(
        f"{header} test_trg_to_src_accuracy {metric_logger.test_trg_to_src_accuracy.global_avg:.5f}"
    )
    print(
        f"{header} test_top3_accuracy {metric_logger.test_top3_accuracy.global_avg:.5f}"
    )
    print(
        f"{header} test_src_to_trg_top3_accuracy {metric_logger.test_src_to_trg_top3_accuracy.global_avg:.5f}"
    )
    print(
        f"{header} test_trg_to_src_top3_accuracy {metric_logger.test_trg_to_src_top3_accuracy.global_avg:.5f}"
    )
    for language in xsim_results.keys():
        print(
            f"{header} test_{language} {metric_logger.meters[language].global_avg:.5f}"
        )
    print(f"{header} test_xsim_average {metric_logger.xsim_average.global_avg:.5f}")


def log_final_test_results_to_wandb(
    metric_logger: MetricLogger,
    outputs: Dict,
    xsim_results,
):
    wandb_log_dict = {
        "flores200/test_loss": metric_logger.test_loss.global_avg,
        "flores200/test_accuracy": metric_logger.test_accuracy.global_avg,
        "flores200/test_src_to_trg_accuracy": metric_logger.test_src_to_trg_accuracy.global_avg,
        "flores200/test_trg_to_src_accuracy": metric_logger.test_trg_to_src_accuracy.global_avg,
        "flores200/test_top3_accuracy": metric_logger.test_top3_accuracy.global_avg,
        "flores200/test_src_to_trg_top3_accuracy": metric_logger.test_src_to_trg_top3_accuracy.global_avg,
        "flores200/test_trg_to_src_top3_accuracy": metric_logger.test_trg_to_src_top3_accuracy.global_avg,
    }
    for language in xsim_results.keys():
        wandb_log_dict["flores200/test_" + language] = metric_logger.meters[
            language
        ].global_avg
    wandb_log_dict["flores200/test_xsim_average"] = (
        metric_logger.xsim_average.global_avg
    )
    if "mlm_loss" in outputs:
        wandb_log_dict["flores200/test_mlm_loss"] = (
            metric_logger.test_mlm_loss.global_avg
        )
    if "src_mlm_loss" in outputs:
        wandb_log_dict["flores200/test_src_mlm_loss"] = (
            metric_logger.test_src_mlm_loss.global_avg
        )
    if "trg_mlm_loss" in outputs:
        wandb_log_dict["flores200/test_trg_mlm_loss"] = (
            metric_logger.test_trg_mlm_loss.global_avg
        )
    if "cls_loss" in outputs:
        wandb_log_dict["flores200/test_cls_loss"] = (
            metric_logger.test_cls_loss.global_avg
        )
    if "koleo_loss" in outputs:
        wandb_log_dict["flores200/test_koleo_loss"] = (
            metric_logger.test_koleo_loss.global_avg
        )
    wandb.log(wandb_log_dict)
