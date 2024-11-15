import sys

sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models.common import MultiHeadAttention, TransformerLayer, AGenderClassificationHead, StatPoolLayer, Permute, AGenderClassificationHeadV2, MaskAGenderClassificationHead

from fusion.features.feature_extractors import FeaturesType



class MTCMAModelV1(nn.Module):
    def __init__(self, dim_q, dim_v, num_heads=4):
        super(MTCMAModelV1, self).__init__()
        self.dim_q = dim_q
        self.dim_v = dim_v
        
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_v)
        self.norm_v = nn.LayerNorm(dim_v)
        
        self.fc_q = nn.Linear(dim_q, 256)
        self.fc_k = nn.Linear(dim_v, 256)
        self.fc_v = nn.Linear(dim_v, 256)

        self.self_attention = MultiHeadAttention(input_dim=256, num_heads=4, dropout=.2)
        
        self.mlp = nn.Sequential(
          nn.Linear(256, 128),
          nn.GELU(),
          nn.Dropout(.2),
          nn.LayerNorm(128),
          nn.Linear(128, 256),
          nn.GELU(),
          nn.Dropout(.2),
        )

    def forward(self, queries, keys, values):
        queries = self.fc_q(self.norm_q(queries))
        keys = self.fc_k(self.norm_k(keys))
        values = self.fc_v(self.norm_v(values))
        
        x = self.self_attention(queries=queries, keys=keys, values=values, mask=None)
        x = x + self.mlp(x)
        return x
    

class MTCMAModelV2(nn.Module):
    def __init__(self, dim_q, dim_v, num_heads=4):
        super(MTCMAModelV2, self).__init__()
        self.dim_q = dim_q
        self.dim_v = dim_v
        
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_v)
        self.norm_v = nn.LayerNorm(dim_v)
        
        self.self_attention = MultiHeadAttention(input_dim=256, num_heads=4, dropout=.2)
        
        self.mlp = nn.Sequential(
          nn.Linear(256, 128),
          nn.GELU(),
          nn.Dropout(.2),
          nn.LayerNorm(128),
          nn.Linear(128, 256),
          nn.GELU(),
          nn.Dropout(.2),
        )

    def forward(self, queries, keys, values):
        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)
        
        x = self.self_attention(queries=queries, keys=keys, values=values, mask=None)
        x = x + self.mlp(x)
        return x
    
class MTCMAModelV3(nn.Module):
    def __init__(self, dim_q, dim_v, num_heads=4):
        super(MTCMAModelV3, self).__init__()
        self.dim_q = dim_q
        self.dim_v = dim_v
        
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_v)
        self.norm_v = nn.LayerNorm(dim_v)
        
        self.attention = TransformerLayer(input_dim=256, num_heads=4, dropout=.2)
        
        self.mlp = nn.Sequential(
          nn.Linear(256, 128),
          nn.GELU(),
          nn.Dropout(.2),
          nn.LayerNorm(128),
          nn.Linear(128, 256),
          nn.GELU(),
          nn.Dropout(.2),
        )

    def forward(self, queries, keys, values):
        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)
        
        x = self.attention(query=queries, key=keys, value=values, mask=None)
        x = x + self.mlp(x)
        return x
'''
EARLY
A: torch.Size([512, 199])
V: torch.Size([4, 197, 768])

INTERMEDIATE
A: torch.Size([199, 1024])
V: torch.Size([4, 13, 768])

LATE
A: torch.Size([256])
V: torch.Size([4, 128])
'''
class AVModelV1(nn.Module):
    def __init__(self, features_type):
        super(AVModelV1, self).__init__()
        
        self.features_type = features_type
        # Video feature branch
        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Linear(199, 4)
            self.downsampling_v = nn.Identity()
            
            self.mtcma_av = MTCMAModelV1(dim_q=512, dim_v=151296)
            self.mtcma_va = MTCMAModelV1(dim_q=151296, dim_v=512)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Linear(199, 4)
            self.downsampling_v = nn.Identity()
            
            self.mtcma_av = MTCMAModelV1(dim_q=1024, dim_v=9984)
            self.mtcma_va = MTCMAModelV1(dim_q=9984, dim_v=1024)
        else:
            self.downsampling_a = nn.Identity()
            self.downsampling_v = nn.Linear(4, 1)
            
            self.mtcma_av = MTCMAModelV1(dim_q=256, dim_v=128, num_heads=1)
            self.mtcma_va = MTCMAModelV1(dim_q=128, dim_v=256, num_heads=1)
        
        self.stp = StatPoolLayer(dim=1)
        
        self.fc = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=.3)
        
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=3)

    def forward(self, x):
        a, v = x
        bs = a.shape[0]
        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1).permute(0, 2, 1)
        
        a = self.downsampling_a(a)
        v = self.downsampling_v(v)
        
        a = a.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        
        av = self.mtcma_av(queries=a, keys=v, values=v)
        va = self.mtcma_va(queries=v, keys=a, values=a)
        
        x = torch.cat((av, va), dim=1)
        x = self.stp(x)
        
        x = self.dp(self.relu(self.fc(x)))        
        return self.cl_head(x)


class AVModelV2(nn.Module):
    def __init__(self, features_type):
        super(AVModelV2, self).__init__()
        
        self.features_type = features_type
        # Video feature branch

        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(512, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(197 * 768, 256),
            )
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(1024, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(13 * 768, 256),
            )
        else:
            self.downsampling_a = nn.Sequential(
                nn.Identity(),
            )

            self.downsampling_v = nn.Sequential(
                nn.Linear(4, 1),
                Permute((0, 2, 1)),
                nn.Linear(128, 256),
            )
            
        self.mtcma_av = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        self.mtcma_va = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        
        self.stp = StatPoolLayer(dim=1)
        
        out_features = 256 if self.features_type == FeaturesType.LATE else 512
        
        self.gender_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )

        self.age_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )
        
        self.cl_head = AGenderClassificationHeadV2(256)

    def forward(self, x):
        a, v = x
        bs = a.shape[0]        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1)

        a = self.downsampling_a(a)
        v = self.downsampling_v(v)

        av = self.mtcma_av(queries=a, keys=v, values=v)
        va = self.mtcma_va(queries=v, keys=a, values=a)

        av = a + av
        va = v + va
        if self.features_type == FeaturesType.LATE:
            av = av.squeeze()
            va = va.squeeze()
        else:
            av = self.stp(av)
            va = self.stp(va)

        x_gender = self.gender_branch(av)
        x_age = self.age_branch(va)
        
        return self.cl_head(x_gender, x_age)


class AVModelV3(nn.Module):
    def __init__(self, features_type):
        super(AVModelV3, self).__init__()
        
        self.features_type = features_type
        # Video feature branch

        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(512, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(197 * 768, 256),
            )
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(1024, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(13 * 768, 256),
            )
        else:
            self.downsampling_a = nn.Sequential(
                nn.Identity(),
            )

            self.downsampling_v = nn.Sequential(
                nn.Linear(4, 1),
                Permute((0, 2, 1)),
                nn.Linear(128, 256),
            )
            
        self.mtcma_av = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        self.mtcma_va = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        
        self.stp = StatPoolLayer(dim=1)
        
        out_features = 256 if self.features_type == FeaturesType.LATE else 512
        
        self.gender_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )

        self.age_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )
        
        self.cl_head = AGenderClassificationHead(256, output_size=3)
        
    def get_agender_features(self, x):
        a, v = x
        bs = a.shape[0]        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1)

        a = self.downsampling_a(a)
        v = self.downsampling_v(v)

        av = self.mtcma_av(queries=a, keys=v, values=v) # -> va
        va = self.mtcma_va(queries=v, keys=a, values=a) # -> av

        av = a + av # -> a + va
        va = v + va # -> v + av
        if self.features_type == FeaturesType.LATE:
            av = av.squeeze()
            va = va.squeeze()
        else:
            av = self.stp(av)
            va = self.stp(va)

        return self.gender_branch(av), self.age_branch(va)

    def forward(self, x):
        x_gender, x_age = self.get_agender_features(x)
        return self.cl_head(x_gender + x_age)


class MaskAgenderAVModelV1(nn.Module):
    def __init__(self, features_type, checkpoint_path=None):
        super(MaskAgenderAVModelV1, self).__init__()
        self.av_model = AVModelV3(features_type=features_type)
            
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
            
        for param in self.av_model.parameters():
            param.requires_grad = False
        
        self.cl_maskhead = nn.Linear(256, 6)

    def forward(self, x):        
        x_gender, x_age = self.av_model.get_agender_features(x)
        agender_res = self.av_model.cl_head(x_gender + x_age)
        mask_res = {'mask': self.cl_maskhead(x_gender + x_age)}
        return {**agender_res, **mask_res}
    
    
class MaskAgenderAVModelV2(nn.Module):
    def __init__(self, features_type, checkpoint_path=None):
        super(MaskAgenderAVModelV2, self).__init__()
        
        self.av_model = AVModelV3(features_type=features_type)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
            
        for param in self.av_model.parameters():
            param.requires_grad = False
        
        self.av_model.cl_head = MaskAGenderClassificationHead(256, output_size=3)

    def forward(self, x):
        x_gender, x_age = self.av_model.get_agender_features(x)
        return self.av_model.cl_head(x_gender + x_age)
              

class MaskAgenderAVModelV3(nn.Module):
    def __init__(self, features_type, checkpoint_path=None):
        super(MaskAgenderAVModelV3, self).__init__()
        
        self.av_model = AVModelV3(features_type=features_type)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.av_model.cl_head = MaskAGenderClassificationHead(256, output_size=3)

    def forward(self, x):
        x_gender, x_age = self.av_model.get_agender_features(x)
        return self.av_model.cl_head(x_gender + x_age)


if __name__ == "__main__":
    device = torch.device('cpu')
    
    features = [
        {
            'a': torch.zeros((10, 512, 199)).to(device),
            'v': torch.zeros((10, 4, 197, 768)).to(device),
            'features_type': FeaturesType.EARLY,
        },
        {
            'a': torch.zeros((10, 199, 1024)).to(device),
            'v': torch.zeros((10, 4, 13, 768)).to(device),
            'features_type': FeaturesType.INTERMEDIATE,
        },
        {
            'a': torch.rand((10, 256)).to(device),
            'v': torch.rand((10, 4, 128)).to(device),
            'features_type': FeaturesType.LATE,
        },
    ]
    
    for f in features:
        model = AVModelV3(features_type=f['features_type'])
        print(model([f['a'], f['v']]))