# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torchmetrics.classification import MulticlassAccuracy
from typing import List, Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm


def bitext_mining_accuracy(
    src_cls_embeddings: torch.Tensor, trg_cls_embeddings: torch.Tensor
):
    src_cls_embeddings_normalized = torch.nn.functional.normalize(
        src_cls_embeddings, dim=-1, p=2
    )
    trg_cls_embeddings_normalized = torch.nn.functional.normalize(
        trg_cls_embeddings, dim=-1, p=2
    )

    # scaled pairwise cosine similarities
    distance = src_cls_embeddings_normalized @ trg_cls_embeddings_normalized.T

    # Compute src and trg accuracy
    labels = torch.arange(src_cls_embeddings.size(0), device=src_cls_embeddings.device)

    metric1 = MulticlassAccuracy(num_classes=src_cls_embeddings.size(0)).to("cuda")
    src_to_trg_accuracy = metric1(distance, labels)
    trg_to_src_accuracy = metric1(distance.T, labels)

    metric2 = MulticlassAccuracy(num_classes=src_cls_embeddings.size(0), top_k=3).to(
        "cuda"
    )
    src_to_trg_top3_accuracy = metric2(distance, labels)
    trg_to_src_top3_accuracy = metric2(distance.T, labels)

    return {
        "accuracy": 0.5 * (src_to_trg_accuracy + trg_to_src_accuracy),
        "src_to_trg_accuracy": src_to_trg_accuracy,
        "trg_to_src_accuracy": trg_to_src_accuracy,
        "top3_accuracy": 0.5 * (src_to_trg_top3_accuracy + trg_to_src_top3_accuracy),
        "src_to_trg_top3_accuracy": src_to_trg_top3_accuracy,
        "trg_to_src_top3_accuracy": trg_to_src_top3_accuracy,
    }


def xsim_accuracy(
    device,
    model,
    languages,
    batch_size,
    block_eff_attention,
    model_name="xlm-roberta-base",
    flores_200_base_path: str = "data/flores200",
):
    from evaluation.xsim.xsim import xSIM

    all_trgs = []
    dataset = xsimDataset(
        language="eng_Latn", flores_200_base_path=flores_200_base_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(
            model_name=model_name, max_model_context_length=2048
        ),
        num_workers=2,
    )
    for idx, data in enumerate(dataloader):
        data = {k: v.to(device) for k, v in data.items()}
        trg_outputs = model.encoder(
            input_ids=data["trg_input_ids"],
            attention_mask=data["trg_attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        if block_eff_attention:
            all_trgs.append(
                model.get_cls_embedding_from_hidden_state(
                    trg_outputs.hidden_states[-1], data["trg_attention_mask"]
                )
                .cpu()
                .detach()
            )
        else:
            all_trgs.append(trg_outputs.hidden_states[-1][:, 0, :].cpu().detach())
    trg_outputs = torch.cat(all_trgs)
    trg_outputs_np = trg_outputs.numpy()

    language_error_rate_pairs = []
    language_error_rate_dict = {}
    for language in tqdm(
        languages, total=len(languages), desc="Evaluating xsim on all languages: "
    ):
        dataset = xsimDataset(
            language=language, flores_200_base_path=flores_200_base_path
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                model_name=model_name, max_model_context_length=2048
            ),
            num_workers=2,
        )

        all_srcs = []

        for idx, data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}
            src_outputs = model.encoder(
                input_ids=data["src_input_ids"],
                attention_mask=data["src_attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            )
            if block_eff_attention:
                all_srcs.append(
                    model.get_cls_embedding_from_hidden_state(
                        src_outputs.hidden_states[-1], data["src_attention_mask"]
                    )
                    .cpu()
                    .detach()
                )
            else:
                all_srcs.append(src_outputs.hidden_states[-1][:, 0, :].cpu().detach())
        src_outputs = torch.cat(all_srcs)
        src_outputs_np = src_outputs.numpy()

        err, nbex, _ = xSIM(
            src_outputs_np,
            trg_outputs_np,
            dim=trg_outputs.size(-1),
        )

        error_rate = 100 * err / nbex
        language_error_rate_pairs.append((language, error_rate))
        language_error_rate_dict["xsim_" + language] = error_rate

    print("+==================+")
    print(",".join([pair[0] for pair in language_error_rate_pairs]))
    print(",".join([str(pair[1]) for pair in language_error_rate_pairs]))
    print("+==================+")
    return language_error_rate_dict


class xsimDataset(torch.utils.data.Dataset):
    def __init__(
        self, language: str = "por_Latn", flores_200_base_path: str = "data/flores200"
    ):
        with open(os.path.join(flores_200_base_path, f"dev/{language}.dev")) as fp:
            self.sources = [src.rstrip() for src in fp.readlines()]
        with open(os.path.join(flores_200_base_path, f"dev/eng_Latn.dev")) as fp:
            self.targets = [src.rstrip() for src in fp.readlines()]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return {
            "src": self.sources[idx],
            "trg": self.targets[idx],
        }


class DataCollatorWithPadding:
    def __init__(
        self, model_name: str = "xlm-roberta-large", max_model_context_length: int = 512
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_model_context_length = max_model_context_length

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

        return model_inputs
