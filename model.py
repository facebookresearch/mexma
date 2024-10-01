# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from losses.koleo import KoLeoLoss
from evaluate import bitext_mining_accuracy
import torch
import numpy as np
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import AutoTokenizer
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaForMaskedLM,
    XLMRobertaPooler,
    XLMRobertaEncoder,
)
import torch.nn as nn
from typing import Optional, Any, Tuple, List

try:  # If you do not want to use the memory efficient version, there is no need to install xformers
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    from backbone.block_diagonal_roberta import (
        EfficientXLMRobertaEncoder,
        EfficientXLMRobertaForMaskedLM,
    )
except ImportError:
    pass


def check_xformers_is_working():
    try:
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    except:
        raise Exception(
            "If you want to use the memory efficient XLM-RoBERTa model, you need to correctly install xformers"
        )


class MEXMA(nn.Module):
    def __init__(
        self,
        encoder: str = "xlm-roberta-large",
        dont_use_block_efficient_attention: bool = False,
        number_of_transformer_layers_in_head: int = 1,
        number_of_transformer_attention_heads_in_head: int = 1,
        number_of_linear_layers: int = 0,
        linear_layers_inputs_dims: List[int] = [],
        linear_layers_outputs_dims: List[int] = [],
        mlm_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        koleo_loss_weight: float = 1.0,
        use_pooler: bool = False,
        use_dropout_in_attention: bool = False,
        initialization_method: str = "torch_default",
    ):
        super().__init__()
        self.initialization_method = initialization_method
        self.extra_forward_parameters = {}
        if not dont_use_block_efficient_attention:
            check_xformers_is_working()
            self.encoder = EfficientXLMRobertaForMaskedLM.from_pretrained(
                encoder
            ).roberta
            self.extra_forward_parameters = {"dont_return_padded_input": True}
        else:
            self.encoder = XLMRobertaForMaskedLM.from_pretrained(encoder).roberta

        if not dont_use_block_efficient_attention:
            head_encoder = EfficientXLMRobertaEncoder
        else:
            head_encoder = XLMRobertaEncoder
        self.unmasking_head = head_encoder(
            config=XLMRobertaConfig(
                num_hidden_layers=number_of_transformer_layers_in_head,
                num_attention_heads=number_of_transformer_attention_heads_in_head,
                attention_probs_dropout_prob=self.encoder.config.attention_probs_dropout_prob,
                hidden_size=self.encoder.config.hidden_size,
                is_decoder=False,
            )
        )
        # Weight initialization
        self.unmasking_head.apply(self._init_weights)

        assert (
            number_of_linear_layers == len(linear_layers_inputs_dims)
            if (
                len(linear_layers_inputs_dims) > 0
                and linear_layers_inputs_dims[0] is not None
            )
            else True
        ), f"number_of_linear_layers must match the length of linear_layers_inputs_dims, got {number_of_linear_layers}, {len(linear_layers_inputs_dims)}"
        assert (
            number_of_linear_layers == len(linear_layers_outputs_dims)
            if (len(linear_layers_outputs_dims) > 0 and linear_layers_outputs_dims[0])
            is not None
            else True
        ), f"number_of_linear_layers must match the length of linear_layers_outputs_dims, got {number_of_linear_layers}, {len(linear_layers_outputs_dims)}"
        # Create the prediction layers
        linear_layers_list = []
        for i in range(number_of_linear_layers):
            linear_layers_list.append(
                nn.Linear(linear_layers_inputs_dims[i], linear_layers_outputs_dims[i])
            )
            linear_layers_list.append(torch.nn.GELU())
        vocab_head = nn.Linear(
            self.encoder.config.hidden_size, self.encoder.config.vocab_size
        )
        linear_layers_list.append(vocab_head)
        self.mlp_head = nn.Sequential(*linear_layers_list)
        # Weight initialization
        self.mlp_head.apply(self._init_weights)
        # Tie the weights of the vocab_head to the embedding_matrix
        vocab_head.weight = self.encoder.embeddings.word_embeddings.weight

        if use_pooler:
            self.pooler = XLMRobertaPooler(self.encoder.config)

        if not use_dropout_in_attention:
            for layer in self.encoder.encoder.layer:
                if not dont_use_block_efficient_attention:
                    layer.attention.self.attention_probs_dropout_prob = 0
                else:
                    layer.attention.self.dropout = nn.Identity()
            for layer in self.unmasking_head.layer:
                if not dont_use_block_efficient_attention:
                    layer.attention.self.attention_probs_dropout_prob = 0
                else:
                    layer.attention.self.dropout = nn.Identity()

        self.mlm_loss = nn.CrossEntropyLoss()
        self.alignment_loss = nn.MSELoss()
        self.koleo_loss = KoLeoLoss()

        self.mlm_loss_weight = mlm_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.koleo_loss_weight = koleo_loss_weight

        self.use_pooler = use_pooler
        self.dont_use_block_efficient_attention = dont_use_block_efficient_attention

    def _init_weights(self, module):
        """Initialize the weights"""
        if self.initialization_method == "torch_default":
            return
        elif self.initialization_method == "normal_dist":
            # Taken from RoBERTa's HuggingFace implementation
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(
                    mean=0.0, std=self.encoder.config.initializer_range
                )
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(
                    mean=0.0, std=self.encoder.config.initializer_range
                )
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif self.initialization_method == "xavier_uniform":
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif self.initialization_method == "xavier_normal":
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight, gain=1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def head_forward(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        other_cls_embedding: torch.Tensor,
    ):
        """
        Computes the forward of the MLM prediction head for this model.
        It replaces the CLS in last_hidden_states of language A, by the CLS of language B in other_cls_embedding.
        Then it feeds it to the head.

        Args:
            last_hidden_states (torch.LongTensor of shape (batch_size, sequence_length, hidden_size) or (batch_size * sequence_length, hidden_size) -> if using block_efficient_attention):
                Last hidden states coming from the target encoder.

            other_cls_embedding (torch.LongTensor of shape (batch_size, sequence_length)):
                The CLS token embedding coming from the other encoder.

            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        """
        if not self.dont_use_block_efficient_attention:
            # For memory-efficient version, there is no batch dimension, so we first split the last_hidden_states per sample,
            # replace its CLS, and then combine them all again.
            lengths = (attention_mask != 0).sum(dim=1).tolist()
            split_hidden_states = torch.split(last_hidden_states.squeeze(0), lengths)
            logits_list = [
                torch.cat(
                    [other_cls_embedding[i].unsqueeze(0), split_hidden_states[i][1:, :]]
                ).unsqueeze(0)
                for i in range(len(split_hidden_states))
            ]
            assert len(logits_list) == attention_mask.size(
                0
            ), f"You should have 1 CLS per sentence, so the same length of logits as the batch size. Got: logits: {len(logits_list)}, mask:{attention_mask.shape}"
            assert len(logits_list) == other_cls_embedding.size(
                0
            ), f"You should have 1 CLS per sentence, so the same length of logits as the batch size. Got: logits: {len(logits_list)}, cls_embedding:{other_cls_embedding.shape}"
            extended_attention_mask, logits = BlockDiagonalMask.from_tensor_list(
                logits_list
            )
        else:
            # We replace the CLS in last_hidden_states[:,0,:] by the other_cls_embedding.
            logits = torch.cat(
                [other_cls_embedding.unsqueeze(1), last_hidden_states[:, 1:, :]], dim=1
            )  # (B,S,E) -> (B,S,E)
            extended_attention_mask = self.encoder.get_extended_attention_mask(
                attention_mask, attention_mask.size()
            )
        self_attention_outputs = self.unmasking_head(
            logits,
            extended_attention_mask,
            output_attentions=False,
            return_dict=False,
        )
        predicted_embeddings = self_attention_outputs[0]
        return {
            "predicted_embeddings": predicted_embeddings,
            "vocab_probabilities": self.mlp_head(predicted_embeddings),
        }

    def get_cls_embedding_from_hidden_state(self, last_hidden_states, attention_mask):
        """
        If we are using block diagonal attention our output is flat, with no batch dimension.
        We need to recover our CLS embeddings, which are the first positions in each sentence, steps:
            1) Get the lengths
            2) Split the hidden_state by the lengths
            3) Pick the first position from each -> The CLS tensor

            Args:
                last_hidden_states (torch.LongTensor of shape (batch_size * sequence_length, hidden_size) -> if using block_efficient_attention):
                    Last hidden states coming from the target encoder.

                attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)):
                    Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                    - 1 for tokens that are **not masked**,
                    - 0 for tokens that are **masked**.
        """
        lengths = (attention_mask != 0).sum(dim=1).tolist()
        split_hidden_states = torch.split(last_hidden_states.squeeze(0), lengths)
        return torch.stack(
            [split_hidden_state[0, :] for split_hidden_state in split_hidden_states]
        )

    def check_block_diagonal_cls_embeddings_have_the_correct_shape(
        self,
        src_cls_embedding,
        trg_cls_embedding,
        src_attention_mask,
        trg_attention_mask,
    ):
        assert src_cls_embedding.size(0) == trg_cls_embedding.size(
            0
        ), f"src and trg CLS should have the same shape, since they have the same number of sentences, but got src:{src_cls_embedding.shape}, trg:{trg_cls_embedding.shape}"
        assert src_cls_embedding.size(1) == trg_cls_embedding.size(
            1
        ), f"src and trg CLS should have the same shape, since they have the same number of sentences, but got src:{src_cls_embedding.shape}, trg:{trg_cls_embedding.shape}"
        assert trg_attention_mask.size(0) == trg_cls_embedding.size(
            0
        ), f"There should be 1 CLS per sentence, but got mask: {trg_attention_mask.shape}, cls: {trg_cls_embedding.shape}"
        assert src_attention_mask.size(0) == src_cls_embedding.size(
            0
        ), f"There should be 1 CLS per sentence, but got mask: {src_attention_mask.shape}, cls: {src_cls_embedding.shape}"

    def remove_masking_from_inputs(
        self,
        src_input_ids: torch.Tensor,
        src_labels: torch.Tensor,
        trg_input_ids: torch.Tensor,
        trg_labels: torch.Tensor,
    ):
        """
        Remove the masking from the src and trg input_ids.
        """
        clean_src_input_ids = src_input_ids.clone().detach()
        clean_src_input_ids[src_labels != -100] = src_labels[src_labels != -100]
        clean_trg_input_ids = trg_input_ids.clone().detach()
        clean_trg_input_ids[trg_labels != -100] = trg_labels[trg_labels != -100]
        return clean_src_input_ids, clean_trg_input_ids

    def get_sentence_representation(
        self,
        src_last_hidden_state: torch.Tensor,
        trg_last_hidden_state: torch.Tensor,
        src_attention_mask: torch.Tensor,
        trg_attention_mask: torch.Tensor,
    ):
        """
            Get the sentence representation from the last_hidden_states, for both src and trg.
            If can be:
                1) Output of HuggingFace's pooler
                2) The CLS embedding
                    a) From the memory efficient attention version
                    b) From the standard attention version
        """
        if self.use_pooler:
            src_cls_embedding = self.pooler(src_last_hidden_state)
            trg_cls_embedding = self.pooler(trg_last_hidden_state)
        elif not self.dont_use_block_efficient_attention:
            src_cls_embedding = self.get_cls_embedding_from_hidden_state(
                src_last_hidden_state, src_attention_mask
            )
            trg_cls_embedding = self.get_cls_embedding_from_hidden_state(
                trg_last_hidden_state, trg_attention_mask
            )
            self.check_block_diagonal_cls_embeddings_have_the_correct_shape(
                src_cls_embedding,
                trg_cls_embedding,
                src_attention_mask,
                trg_attention_mask,
            )
        else:
            src_cls_embedding = src_last_hidden_state[:, 0, :]
            trg_cls_embedding = trg_last_hidden_state[:, 0, :]
        return src_cls_embedding, trg_cls_embedding

    def forward(
        self,
        src_input_ids: Optional[torch.Tensor] = None,
        src_attention_mask: Optional[torch.Tensor] = None,
        src_head_mask: Optional[torch.Tensor] = None,
        src_inputs_embeds: Optional[torch.Tensor] = None,
        src_labels: Optional[torch.LongTensor] = None,
        trg_input_ids: Optional[torch.Tensor] = None,
        trg_attention_mask: Optional[torch.Tensor] = None,
        trg_head_mask: Optional[torch.Tensor] = None,
        trg_inputs_embeds: Optional[torch.Tensor] = None,
        trg_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage: str = "inference",
        **kwargs,
    ):
        """
        The main MEXMA logic,
        It receives the input_ids and attention_masks of both src and trg, in language A and language B, respectively.
        It encodes both src and trg in a clean and masked instance, getting their sentence and token-level representations.
        It also performs the unmasking of src masked tokens with the sentence representation from the trg, and vice versa.

        Args (xxx_ stands for either src or trg), as defined 
        in XLM-RoBERTa (https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py):
            xxx_input_ids  (torch.LongTensor of shape (batch_size, sequence_length)):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

            xxx_attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            xxx_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            xxx_inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.

            xxx_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
                Note: Only used if stage is one of [train, test]

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            stage (`string`):
                Indicate which stage it is being used as, options: ['train', 'test', 'inference']
        """
        # Get src and trg embeddings for masked inputs
        masked_src_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            head_mask=src_head_mask,
            inputs_embeds=src_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **self.extra_forward_parameters,
        )
        masked_trg_outputs = self.encoder(
            input_ids=trg_input_ids,
            attention_mask=trg_attention_mask,
            head_mask=trg_head_mask,
            inputs_embeds=trg_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **self.extra_forward_parameters,
        )
        masked_src_cls_embedding, masked_trg_cls_embedding = (
            self.get_sentence_representation(
                src_last_hidden_state=masked_src_outputs.last_hidden_state,
                trg_last_hidden_state=masked_trg_outputs.last_hidden_state,
                src_attention_mask=src_attention_mask,
                trg_attention_mask=trg_attention_mask,
            )
        )

        # Get src and trg embeddings for clean inputs
        clean_src_input_ids, clean_trg_input_ids = self.remove_masking_from_inputs(
            src_input_ids, src_labels, trg_input_ids, trg_labels
        )
        clean_src_outputs = self.encoder(
            input_ids=clean_src_input_ids,
            attention_mask=src_attention_mask,
            head_mask=src_head_mask,
            inputs_embeds=src_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **self.extra_forward_parameters,
        )
        clean_trg_outputs = self.encoder(
            input_ids=clean_trg_input_ids,
            attention_mask=trg_attention_mask,
            head_mask=trg_head_mask,
            inputs_embeds=trg_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **self.extra_forward_parameters,
        )
        clean_src_cls_embedding, clean_trg_cls_embedding = (
            self.get_sentence_representation(
                src_last_hidden_state=clean_src_outputs.last_hidden_state,
                trg_last_hidden_state=clean_trg_outputs.last_hidden_state,
                src_attention_mask=src_attention_mask,
                trg_attention_mask=trg_attention_mask,
            )
        )

        # Predict the src tokens given the trg sentence representation
        src_head_outputs = self.head_forward(
            last_hidden_states=masked_src_outputs.last_hidden_state,
            attention_mask=src_attention_mask,
            other_cls_embedding=clean_trg_cls_embedding,
        )
        # Predict the trg tokens given the src sentence representation
        trg_head_outputs = self.head_forward(
            last_hidden_states=masked_trg_outputs.last_hidden_state,
            attention_mask=trg_attention_mask,
            other_cls_embedding=clean_src_cls_embedding,
        )

        data_to_return = {
            "masked_src_cls_embedding": masked_src_cls_embedding,
            "masked_trg_cls_embedding": masked_trg_cls_embedding,
            "clean_src_cls_embedding": clean_src_cls_embedding,
            "clean_trg_cls_embedding": clean_trg_cls_embedding,
            "src_labels": src_labels,
            "trg_labels": trg_labels,
            "src_vocab_probabilities": src_head_outputs["vocab_probabilities"],
            "trg_vocab_probabilities": trg_head_outputs["vocab_probabilities"],
        }
        if not self.dont_use_block_efficient_attention:
            data_to_return["src_attention_mask"] = src_attention_mask
            data_to_return["trg_attention_mask"] = trg_attention_mask
        if stage == "inference":
            return data_to_return
        elif stage == "train":
            return self.training_step(**data_to_return)
        elif stage == "test":
            return self.validation_step(**data_to_return)
        else:
            print(
                f"You chose the stage {stage}, but the options are [train, test, inference]"
            )
            exit()

    def training_step(
        self,
        src_vocab_probabilities: torch.Tensor = None,
        trg_vocab_probabilities: torch.Tensor = None,
        trg_labels: Optional[torch.LongTensor] = None,
        src_labels: Optional[torch.LongTensor] = None,
        clean_src_cls_embedding: Optional[torch.Tensor] = None,
        clean_trg_cls_embedding: Optional[torch.Tensor] = None,
        src_attention_mask: Optional[torch.LongTensor] = None,
        trg_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Compute the 3 losses:
        - Alignment loss between the 2 CLS
        - MLM loss (src and trg)
        - KoLeo loss (src and trg)

        Check forward for args description.
        """
        if not self.dont_use_block_efficient_attention:

            def remove_padding_labels(labels, attention_mask, vocab_probabilities):
                lengths = (attention_mask != 0).sum(dim=1).tolist()
                assert sum(lengths) == vocab_probabilities.size(
                    1
                ), f"The sum of lengths of sentences, and the sequence dim of vocab_probabilities should match. Got sum: {sum(lengths)}, vocab_probabilities: {trg_vocab_probabilities.size(1)}"
                split_hidden_states = [
                    labels[idx, :length] for idx, length in enumerate(lengths)
                ]
                return torch.cat(split_hidden_states).unsqueeze(0)

            src_labels = remove_padding_labels(
                src_labels, src_attention_mask, src_vocab_probabilities
            )
            trg_labels = remove_padding_labels(
                trg_labels, trg_attention_mask, trg_vocab_probabilities
            )

            assert src_vocab_probabilities.size(1) == src_labels.size(
                1
            ), f"vocab probs and labels should have the same shape, but got: {src_vocab_probabilities.shape} and {src_labels.shape}"
            assert trg_vocab_probabilities.size(1) == trg_labels.size(
                1
            ), f"vocab probs and labels should have the same shape, but got: {trg_vocab_probabilities.shape} and {trg_labels.shape}"
        src_mlm_loss = self.mlm_loss(
            src_vocab_probabilities.reshape(-1, self.encoder.config.vocab_size),
            src_labels.view(-1),
        )
        trg_mlm_loss = self.mlm_loss(
            trg_vocab_probabilities.reshape(-1, self.encoder.config.vocab_size),
            trg_labels.view(-1),
        )
        mlm_loss = (src_mlm_loss + trg_mlm_loss) / 2

        cls_loss = self.alignment_loss(clean_src_cls_embedding, clean_trg_cls_embedding)

        koleo_loss = (
            self.koleo_loss(clean_src_cls_embedding)
            + self.koleo_loss(clean_trg_cls_embedding)
        ) / 2

        loss = (
            self.cls_loss_weight * cls_loss
            + self.mlm_loss_weight * mlm_loss
            + self.koleo_loss_weight * koleo_loss
        )
        return {
            "loss": loss,
            "src_mlm_loss": src_mlm_loss,
            "trg_mlm_loss": trg_mlm_loss,
            "cls_loss": cls_loss,
            "koleo_loss": koleo_loss,
        }

    @torch.no_grad
    def validation_step(
        self,
        src_vocab_probabilities: torch.Tensor = None,
        trg_vocab_probabilities: torch.Tensor = None,
        trg_labels: Optional[torch.LongTensor] = None,
        src_labels: Optional[torch.LongTensor] = None,
        clean_src_cls_embedding: Optional[torch.Tensor] = None,
        clean_trg_cls_embedding: Optional[torch.Tensor] = None,
        src_attention_mask: Optional[torch.LongTensor] = None,
        trg_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Compute the 3 losses:
        - Alignment loss between the 2 CLS
        - MLM loss (src and trg)
        - KoLeo loss (src and trg)

        Compute simplistic mining accuracy for fast verification of model progress.

        Check forward for args description.
        """
        train_outputs = self.training_step(
            src_vocab_probabilities=src_vocab_probabilities,
            trg_vocab_probabilities=trg_vocab_probabilities,
            trg_labels=trg_labels,
            src_labels=src_labels,
            clean_src_cls_embedding=clean_src_cls_embedding,
            clean_trg_cls_embedding=clean_trg_cls_embedding,
            src_attention_mask=src_attention_mask,
            trg_attention_mask=trg_attention_mask,
            **kwargs,
        )

        mining_outputs = bitext_mining_accuracy(
            src_cls_embeddings=clean_src_cls_embedding,
            trg_cls_embeddings=clean_trg_cls_embedding,
        )

        return {
            **train_outputs,
            "accuracy": mining_outputs["accuracy"],
            "src_to_trg_accuracy": mining_outputs["src_to_trg_accuracy"],
            "trg_to_src_accuracy": mining_outputs["trg_to_src_accuracy"],
            "top3_accuracy": mining_outputs["top3_accuracy"],
            "src_to_trg_top3_accuracy": mining_outputs["src_to_trg_top3_accuracy"],
            "trg_to_src_top3_accuracy": mining_outputs["trg_to_src_top3_accuracy"],
        }

    def encode(
        self,
        sentences,
        batch_size=32,
        tokenizer=AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True),
        **kwargs,
    ) -> List[torch.Tensor]:
        """Returns a list of embeddings for the given sentences.
        Useful for MTEB, or other downstream usage.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            inputs = tokenizer(
                sentences_batch, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                outputs = self.encoder(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                if not self.dont_use_block_efficient_attention:
                    embeddings = self.get_cls_embedding_from_hidden_state(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                else:
                    embeddings = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.extend(embeddings.detach().cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings
