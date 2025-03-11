import torch
import torch.nn as nn
import math
from typing import List, Optional
import torch.nn.functional as F
from models.base_model import BaseModel
from torch_geometric.nn import PointNet2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig
from config.config import Config


class Lidar_LLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lidar_encoder = Lidar_Encoder()
        self.vat = VAT()
        self.mlp = MLP()
        self.query = torch.tensor(100)
        self.llm = LLM()

    def forward(self, lidar, prompts):
        bev_feature = self.lidar_encoder(lidar)
        bev_feature = self.vat(bev_feature, self.query)
        output = self.llm(bev_feature, prompts)
        return output


class Lidar_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet2 = PointNet2(segmentation=False)

    def forward(self, x):
        x = self.pointnet2(x)
        return x


class LLM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            lora_r_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def forward(self, x, prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt',
                                padding=True, truncation=True)

        input_embeds = self.model.transformer.wte(inputs['input_ids'])
        input_embeds_with_prompt = torch.cat((x, input_embeds), dim=1)
        outputs = self.model(inputs_embeds=input_embeds_with_prompt)
        return outputs


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Linear(100, 100)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(100, 100)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.decoder(x)


class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, bev_emb, queries, mask=None):
        N = queries.shape[0]
        query = self.query(queries)
        keys = self.key(bev_emb)
        values = self.value(bev_emb)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = values.permute(2, 0, 1, 3)
        keys = keys.permute(2, 0, 1, 3)
        query = query.permute(2, 0, 1, 3)

        energy = torch.einsum("qhnd,qhmd->hnqm", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("hnqm,hmvd->hnqd", [attention, values])

        out = out.reshape(N, query_len, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.embed_size = embed_size
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        out = self.gamma * x_norm + self.beta
        return out


class BatchNormWithPatching(nn.Module):
    def __init__(self, embed_size, eps=1e-6, momentum=0.1):
        super(BatchNormWithPatching, self).__init__()
        self.embed_size = embed_size
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))

        self.running_mean = torch.zeros(embed_size)
        self.running_var = torch.ones(embed_size)

    def forward(self, x):
        if x.dim() == 2:
            batch_size = 1
            mean = x.mean(dim=0)
            variance = x.var(dim=0, unbiased=False)
        else:
            batch_size = x.size(0)
            mean = x.mean(dim=0, keepdim=True)
            variance = x.var(dim=0, keepdim=True, unbiased=False)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            mean = self.running_mean
            variance = self.running_var

        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        out = self.gamma * x_norm + self.beta
        return out


class VAT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossAttention = CrossAttention(100, 5)
        self.selfAttention = CrossAttention(100, 5)
        self.batchnorm1 = BatchNormWithPatching(100)
        self.batchnorm2 = BatchNormWithPatching(100)
        self.mlp = MLP(100, 100)

    def forward(self, bev_emb, queries):
        bev_emb = self.selfAttention(bev_emb, bev_emb)
        bev_emb = self.batchnorm1(bev_emb)
        bev_emb = self.crossAttention(bev_emb, queries)
        bev_emb = self.batchnorm2(bev_emb)
        return self.mlp(bev_emb)
