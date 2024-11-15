import sys

sys.path.append('src')

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderSOTAModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        x_gen = logits_gender[:, 0:2]
        x_age = logits_age[:, 0]
        return {'gen': x_gen, 'age': x_age}


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, 64000)).to(device)
    models = [
        {'cls': AgeGenderSOTAModel, 'model_name': 'audeering/wav2vec2-large-robust-24-ft-age-gender'}
    ]

    for model in models:
        model_name = model['model_name']
        model = model['cls'].from_pretrained(model_name)

        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['gen'].shape)
        print(res['age'].shape)
