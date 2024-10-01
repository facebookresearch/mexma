# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from HuggingFace: https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py
# where we replace its attention mechanism to use xformers memory efficient attention with the block diagonal mask.

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaPreTrainedModel,
    XLMRobertaSelfAttention,
    XLMRobertaAttention,
    XLMRobertaLayer,
    XLMRobertaEncoder,
    XLMRobertaEmbeddings,
    XLMRobertaPooler,
    XLMRobertaLMHead,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from xformers.ops import memory_efficient_attention


class EfficientXLMRobertaSelfAttention(XLMRobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        assert (
            self.position_embedding_type == "absolute"
        ), f"Can only use xformers mem_eff_attn with absolute position_embedding_type, currently you have {self.position_embedding_type}"
        assert (
            head_mask == None
        ), "Not possible to use head_mask with xformers mem_eff_attn"
        if output_attentions:
            print("WARNING: Cannot use output_attentions with xformers mem_eff_attn")

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        context_layer = memory_efficient_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            attn_bias=attention_mask,
            p=self.attention_probs_dropout_prob if self.training else 0,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class EfficientXLMRobertaAttention(XLMRobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = EfficientXLMRobertaSelfAttention(
            config, position_embedding_type=position_embedding_type
        )


class EfficientXLMRobertaLayer(XLMRobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = EfficientXLMRobertaAttention(config)


class EfficientXLMRobertaEncoder(XLMRobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [EfficientXLMRobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )


class EfficientXLMRobertaModel(XLMRobertaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = EfficientXLMRobertaEncoder(config)

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dont_return_padded_input: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )  # (B, S, E)
        # Here we flatten the inputs into a 1D tensor with no padding, to use with BlockDiagonal
        # We first get the unpadded lengths of each sentence
        lengths = (attention_mask != 0).sum(dim=1).tolist()  # (B, S, E)
        # Then we crop each sentence up to its first paddng token
        unpadded_list = [
            embedding_output[idx, :length, :].unsqueeze(0)
            for idx, length in enumerate(lengths)
        ]
        # We feed it to BlockDiagonalMask which will create  an attention_bias and give us a flat array of embeddings
        attn_bias_enc, embeddings = BlockDiagonalMask.from_tensor_list(
            unpadded_list
        )  # (1, S, E)
        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            attention_mask=attn_bias_enc,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # If the output is to be the same size as the input, we then to add the padding back
        if not dont_return_padded_input:
            chunk_lengths = torch.diff(attn_bias_enc.q_seqinfo.seqstart)
            chunks = torch.split(
                encoder_outputs[0].squeeze(0), chunk_lengths.tolist(), dim=0
            )
            encoder_outputs.last_hidden_state = nn.utils.rnn.pad_sequence(
                chunks, batch_first=True
            )
            assert (
                encoder_outputs.last_hidden_state.shape == embedding_output.shape
            ), "The padded output should have the original shape of the input"

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class EfficientXLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            print(
                "If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = EfficientXLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
