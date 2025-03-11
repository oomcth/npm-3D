import torch
import torch.nn as nn
import math
from typing import List, Optional, Union
import torch.nn.functional as F
from models.base_model import BaseModel
from torch_geometric.nn.models import LightGCN
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig


class Lidar_LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_encoder = Lidar_Encoder()
        self.vat = VAT()
        self.mlp = MLP(100, 100)
        self.mlp.decoder = nn.Linear(100, 768)
        self.query = torch.rand((1, 100))
        self.llm = LLM()

    def forward(self, lidar, prompts, answer=None, criterion=None):
        self = self.to(lidar.device)
        self.query = self.query.to(lidar.device)
        bev_feature = self.lidar_encoder(lidar)
        bev_feature = self.vat(bev_feature, self.query)
        bev_feature = self.mlp(bev_feature)
        output = self.llm(bev_feature, prompts)

        if answer is not None and criterion is not None:
            answer_tokens = self.llm.tokenizer(
                answer, padding=True, truncation=True, return_tensors="pt"
            ).input_ids

            loss = criterion(output[:, -1, :].view(-1, output.size(-1)),
                             answer_tokens.view(-1).to(lidar.device))

            valid_tokens = (answer_tokens != self.llm.tokenizer.pad_token_id).sum()

            return loss, valid_tokens
        return output

    def generate(self, lidar, prompts):
        bev_feature = self.lidar_encoder(lidar)
        bev_feature = self.vat(bev_feature, self.query)
        gen = self.llm.generate(bev_feature, prompts)
        return gen


class MiniPointNet(nn.Module):
    def __init__(self, emb_size=100):
        super(MiniPointNet, self).__init__()
        self.emb_size = emb_size

        self.mlp1 = nn.Linear(4, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, emb_size)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)

        x = torch.max(x, dim=1)[0]

        return x


class Lidar_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = MiniPointNet()

    def forward(self, x):
        x = self.pointnet(x)
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
            target_modules=["c_attn"],
            bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

        self.max_length = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x, prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt',
                                padding=True, truncation=True)

        input_embeds = self.model.transformer.wte(inputs['input_ids'].to(x.device))
        input_embeds_with_prompt = torch.cat((x.unsqueeze(1), input_embeds), dim=1)
        outputs = self.model(inputs_embeds=input_embeds_with_prompt)
        return outputs.logits

    def generate(self,
                 x: torch.Tensor,
                 prompts: Union[str, List[str]],
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 num_return_sequences: int = 1,
                 do_sample: bool = True,
                 batch_size: Optional[int] = None) -> List[str]:
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k

        device = next(self.parameters()).device
        x = x.to(device)

        if isinstance(prompts, str):
            prompts = [prompts] * x.shape[0]

        if len(prompts) != x.shape[0]:
            raise ValueError(f"Le nombre de prompts ({len(prompts)}) doit correspondre "
                             f"à la taille du batch des nuages de points ({x.shape[0]})")

        if batch_size is not None and batch_size < x.shape[0]:
            all_generated_texts = []

            for i in range(0, x.shape[0], batch_size):
                batch_x = x[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]

                batch_texts = self._generate_batch(
                    batch_x, batch_prompts, max_length, temperature,
                    top_p, top_k, num_return_sequences, do_sample
                )
                all_generated_texts.extend(batch_texts)

            return all_generated_texts
        else:
            return self._generate_batch(
                x, prompts, max_length, temperature,
                top_p, top_k, num_return_sequences, do_sample
            )

    def _generate_batch(self,
                        x: torch.Tensor,
                        prompts: List[str],
                        max_length: int,
                        temperature: float,
                        top_p: float,
                        top_k: int,
                        num_return_sequences: int,
                        do_sample: bool) -> List[str]:

        device = next(self.parameters()).device

        tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        text_embeds = self.model.transformer.wte(input_ids)

        combined_embeds = torch.cat((x, text_embeds), dim=1)

        point_cloud_mask = torch.ones((x.shape[0], x.shape[1]), device=device)
        combined_attention_mask = torch.cat((point_cloud_mask, attention_mask), dim=1)

        self.eval()
        with torch.no_grad():
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": combined_attention_mask
            }

            generated_sequences = self.model.generate(
                inputs_embeds=combined_embeds,
                **gen_kwargs
            )

            generated_texts = []
            for seq in generated_sequences:
                decoded_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated_texts.append(decoded_text)

            return generated_texts


class MLP(nn.Module):
    def __init__(self, inputdim, latentdim):
        super().__init__()
        self.encoder = nn.Linear(inputdim, latentdim)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(latentdim, inputdim)
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

        values = values.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        keys = keys.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        query = query.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        values = values.permute(2, 0, 1, 3)
        keys = keys.permute(2, 0, 1, 3)
        query = query.permute(2, 0, 1, 3)
        energy = torch.einsum("qhnd,qhmd->hnqm", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("abcd,abce->abde", [attention, values])
        out = out.reshape(N, 1, self.num_heads * self.head_dim).squeeze(1)
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
        # self.batchnorm1 = nn.LayerNorm(100)
        # self.batchnorm2 = nn.LayerNorm(100)
        self.mlp = MLP(100, 100)

    def forward(self, bev_emb, queries):
        bev_emb = self.selfAttention(bev_emb, bev_emb)
        # bev_emb = self.batchnorm1(bev_emb)
        bev_emb = self.crossAttention(bev_emb, queries)
        # bev_emb = self.batchnorm2(bev_emb)
        return self.mlp(bev_emb)
