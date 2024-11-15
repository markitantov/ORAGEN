import sys

sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


from transformers import ViTForImageClassification

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


from models.common import TransformerLayer, StatPoolLayer, AGenderClassificationHead
from utils.common import AttrDict


class AudioWrapperModel(Wav2Vec2PreTrainedModel):
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

    def forward(self, x):
        a, v = x
        x = self.transformer_block1(a, a, a)
        x = self.relu(x)
        
        x = self.transformer_block2(x, x, x)
        x = self.relu(x)

        x = self.stp(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        return self.cl_head(x)

class DPAL(nn.Module):
    def __init__(self, input_dim):
        super(DPAL, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x1, x2, x3):
        queries = self.query(x1)
        keys = self.key(x2)
        values = self.value(x3)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class StatisticalPoolingLayer(nn.Module): # TODO
    def __init__(self):
        super(StatisticalPoolingLayer, self).__init__()

    def forward(self, x):
        mean_x = torch.mean(x, dim=1)
        std_x = torch.std(x, dim=1)
        stat_x = torch.cat((mean_x, std_x), dim=1)
        return stat_x


class Model_v2(nn.Module):
    def __init__(self, input_dim, hid_dim, drop, n_cl):
        super(Model_v2, self).__init__()

        self.fcl_x = nn.Linear(input_dim, input_dim)
        self.drop_fcl_x = nn.Dropout(p=drop)
        self.dpal = DPAL(input_dim)
        self.stat_pool = StatisticalPoolingLayer()
        self.fcl = nn.Linear(input_dim*2, hid_dim)
        self.drop_fcl = nn.Dropout(p=drop)
        self.classifier = nn.Linear(hid_dim, n_cl)

    def get_features(self, x):
        x1 = self.drop_fcl_x(self.fcl_x(x))
        att_x1 = self.dpal(x1, x1, x1)
        att_x1 = self.stat_pool(att_x1)
        return self.drop_fcl(self.fcl(att_x1))

    def forward(self, x):
        x1 = self.drop_fcl_x(self.fcl_x(x))
        att_x1 = self.dpal(x1, x1, x1)
        att_x1 = self.stat_pool(att_x1)
        stat_att_x1 = self.drop_fcl(self.fcl(att_x1))
        out = self.classifier(stat_att_x1)
        out[:, -1] = torch.sigmoid(out[:, -1])
        return out

class VideoWrapperModel(nn.Module):
    def __init__(self, input_dim=768, gated_dim=128, drop=0, n_cl=3):
        super(VideoWrapperModel, self).__init__()
        self.vit_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.model_final = Model_v2(input_dim=input_dim, hid_dim=gated_dim, drop=drop, n_cl=n_cl)

    def forward(self, inputs):       
        a, v = inputs
        out = self.model_final(v.reshape(-1, 13, 768)) # check only first frame
        x_gen = out[:, 0:-1]
        x_age = out[:, -1]
        return {'gen': x_gen, 'age': x_age}
    