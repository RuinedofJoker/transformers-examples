from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from transformers.utils.generic import TransformersKwargs
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Unpack
import torch
from torch.nn import CosineSimilarity, CosineEmbeddingLoss

class DualModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()

    def forward(
            self,
            # (B,2,S)
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        # (B,H) / (1,768)
        senA_pooled_output = senA_outputs.pooler_output
        senB_pooled_output = senB_outputs.pooler_output

        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)

        loss = None
        if labels is not None:
            loss = CosineEmbeddingLoss(0.3)(senA_pooled_output, senB_pooled_output, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos
        )
