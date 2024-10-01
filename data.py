# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer
from typing import Tuple, Any, List, Dict, Optional, Iterator
import pandas as pd
import math
import torch.distributed as dist
import os
import datasets


class ParallelTranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file: Optional[str] = None,
        flores_200_src_languages: Optional[List[str]] = None,
        hf_dataset_directory: Optional[str] = None,
        flores_200_base_path: str = "data/flores200",
    ):
        self.src_list = []
        self.trg_list = []
        self.nllb_data = None
        if flores_200_src_languages is None:
            assert not (
                (data_file is None) and (hf_dataset_directory is None)
            ), "You need to set at least data_file or hf_dataset_directory to train the model"
        if data_file is not None:
            df = pd.read_csv(data_file)
            df = df.dropna()
            self.src_list.extend(list(df["src"].values))
            self.trg_list.extend(list(df["trg"].values))
            self.total_size = len(self.src_list)
        if flores_200_src_languages is not None:
            for language in flores_200_src_languages:
                print(f"Loading flores language: {language}")
                with open(
                    os.path.join(flores_200_base_path, f"dev/{language}.dev")
                ) as fp:
                    current_srcs_list = [src.rstrip() for src in fp.readlines()]
                    self.src_list.extend(current_srcs_list)
                with open(
                    os.path.join(flores_200_base_path, f"dev/eng_Latn.dev")
                ) as fp:
                    self.trg_list.extend([src.rstrip() for src in fp.readlines()])
            self.total_size = len(self.src_list)
        if hf_dataset_directory is not None:
            print(f"The training data will be loaded on the fly")
            self.nllb_data = datasets.load_from_disk(hf_dataset_directory)["train"]
            self.total_size = len(self.nllb_data)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if self.nllb_data is not None:
            return self.nllb_data[idx]
        else:
            return {
                "src": self.src_list[idx],
                "trg": self.trg_list[idx],
            }


class DataCollatorWithMaskingAndPadding:
    def __init__(
        self,
        encoder: str = "xlm-roberta-large",
        max_model_context_length: int = 64,
        src_mlm_probability: float = 0.15,
        trg_mlm_probability: float = 0.15,
        seed: int = 42,
        epoch: int = 0,
        rank: int = 0,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
        self.max_model_context_length = max_model_context_length
        self.trg_mlm_probability = trg_mlm_probability
        self.src_mlm_probability = src_mlm_probability
        self.seed = seed
        self.epoch = epoch
        self.rank = rank

        self.source_masking_generator = torch.Generator()
        self.source_masking_generator.manual_seed(self.seed + self.rank + self.epoch)
        self.target_masking_generator = torch.Generator()
        self.target_masking_generator.manual_seed(self.seed + self.rank + self.epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.source_masking_generator.manual_seed(self.seed + self.rank + self.epoch)
        self.target_masking_generator.manual_seed(self.seed + self.rank + self.epoch)

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Convert from list of dicts to dict of lists, i.e. group items by key in a list
        data_changed = {k: [dic[k] for dic in data] for k in data[0]}

        # Tokenize src and trg sentences, with padding to longest sentence in batch
        src_inputs = self.tokenizer(
            data_changed["src"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_context_length,
            padding="longest",
        )

        trg_inputs = self.tokenizer(
            data_changed["trg"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_context_length,
            padding="longest",
        )

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        trg_inputs = {"trg_" + k: v for k, v in trg_inputs.items()}
        model_inputs = {**src_inputs, **trg_inputs}

        if (self.src_mlm_probability == 0) and (self.trg_mlm_probability == 0):
            model_inputs["src_labels"] = model_inputs["src_input_ids"].clone()
            model_inputs["trg_labels"] = model_inputs["trg_input_ids"].clone()
            model_inputs["src_labels"][
                model_inputs["src_labels"] == self.tokenizer.pad_token_id
            ] = -100
            model_inputs["trg_labels"][
                model_inputs["trg_labels"] == self.tokenizer.pad_token_id
            ] = -100
            return model_inputs

        # Apply masking
        model_inputs["src_input_ids"], model_inputs["src_labels"] = (
            self.torch_mask_tokens(
                model_inputs["src_input_ids"],
                mlm_probability=self.src_mlm_probability,
                generator=self.source_masking_generator,
            )
        )
        model_inputs["trg_input_ids"], model_inputs["trg_labels"] = (
            self.torch_mask_tokens(
                model_inputs["trg_input_ids"],
                mlm_probability=self.trg_mlm_probability,
                generator=self.target_masking_generator,
            )
        )
        return model_inputs

    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/data/data_collator.py#L751
    """
    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
        mlm_probability: float = 0.15,
        generator: torch.Generator = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # We prevent masking special tokens
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # This prevents that no token gets masked, which would lead to all labels=-100,
        # which leads to the loss being nan.
        no_masked_indices = True
        while no_masked_indices:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(
                probability_matrix, generator=generator
            ).bool()
            no_masked_indices = torch.all(~masked_indices).item()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8), generator=generator).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5), generator=generator).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long, generator=generator
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SliceableDistributedSampler(torch.utils.data.Sampler):
    """
        Changes the implementation of DistributedSampler, so that it can start from a specific index, without
        the need for the DataLoader to consume unnecessary indices in the case of a restart.
    """
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        start_iteration: int = 0,
        batch_size: int = 8,
        epoch_to_skip: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.start_iteration = start_iteration
        self.batch_size = batch_size
        self.epoch_to_skip = epoch_to_skip
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if (
            self.epoch == self.epoch_to_skip
        ):  # Only skip the batches the first time we run it
            return iter(indices[self.start_iteration * self.batch_size :])
        else:
            return iter(indices)

    def __len__(self) -> int:
        if self.epoch == self.epoch_to_skip:
            return self.num_samples - (self.start_iteration * self.batch_size)
        else:
            return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
