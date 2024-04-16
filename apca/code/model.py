# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch
import logging

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        if "bert" in args.model_type:
            self.classifier = RobertaClassificationHead(config)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if "bert" in self.args.model_type:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            vec = outputs[0]
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        # BCELoss()
        logits = self.classifier(vec)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = self.loss(logits.reshape(-1), labels)
            return loss, prob
        else:
            return prob
