import sys

sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torchaudio

from transformers import AutoConfig, HubertModel

from transformers.models.hubert.modeling_hubert import (
    HubertModel,
    HubertPreTrainedModel
)

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from common.models.common import TransformerLayer, StatPoolLayer, AGenderClassificationHead
from common.utils.common import AttrDict


class AGenderAudioW2V2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

        self.f_size = 1024

        self.transformer_block1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.transformer_block2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)

        self.stp = StatPoolLayer(dim=1)

        self.fc1 = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=.6)
    
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=config.output_size)
        
        self.init_weights()
        
        # freeze conv
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def early_features(self, x):
        return self.wav2vec2.feature_extractor(x)

    def intermediate_features(self, x):
        return self.wav2vec2(x)[0]

    def late_features(self, x):
        outputs = self.intermediate_features(x)

        x = self.transformer_block1(outputs, outputs, outputs)
        x = self.relu(x)
        
        x = self.transformer_block2(x, x, x)
        x = self.relu(x)

        x = self.stp(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        return x

    def forward(self, x):
        x = self.late_features(x)
        return self.cl_head(x)


class AGenderAudioHuBERTModel(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hubert = HubertModel(config)

        self.f_size = 1024

        self.transformer_block1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.transformer_block2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)

        self.stp = StatPoolLayer(dim=1)

        self.fc1 = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=.6)
    
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=config.output_size)
        
        self.init_weights()
        
        # freeze conv
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        for param in self.hubert.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def early_features(self, x):
        return self.hubert.feature_extractor(x)

    def intermediate_features(self, x):
        return self.hubert(x)[0]

    def late_features(self, x):
        outputs = self.intermediate_features(x)
        print(outputs.shape)

        x = self.transformer_block1(outputs, outputs, outputs)
        print(x.shape)
        
        x = self.relu(x)
        
        x = self.transformer_block2(x, x, x)
        x = self.relu(x)

        x = self.stp(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        return x

    def forward(self, x):
        x = self.late_features(x)
        return self.cl_head(x)
    

class AGenderModelWrapper(nn.Module):
    def __init__(self, model_cls, model_args):
        super(AGenderModelWrapper, self).__init__()
        self.model = model_cls.from_pretrained(**model_args)
        
    def forward(self, x):
        if len(x) == 2:
            return self.model(x[0])
        else:
            return self.model(x)
    

class AGenderSimpleModel(nn.Module):
    def __init__(self, input_size=64000):
        super(AGenderSimpleModel, self).__init__()
        self.fc = nn.Linear(64000, 256)
        self.relu = nn.ReLU()
        self.classificator = AGenderClassificationHead(256, 3)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)

        return self.classificator(x)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, 64000)).to(device)
    models = [
        # {'cls': AGenderAudioW2V2Model, 'model_name': 'facebook/wav2vec2-large-robust'},
        {'cls': AGenderAudioHuBERTModel, 'model_name': 'facebook/hubert-large-ls960-ft'}
    ]

    model_name = 'facebook/wav2vec2-large-robust'

    for model in models:
        model_name = model['model_name']
        config = AutoConfig.from_pretrained(model_name)

        config.output_size = 3
        model = model['cls'].from_pretrained(model_name, config=config)

        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['gen'].shape)
        print(res['age'].shape)