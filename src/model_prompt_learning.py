import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from transformers import Conv1D
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2PreTrainedModel, logger
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from casual_encoder import GCNModelSIGVAE


class KGPrompt(nn.Module):
    def __init__(
            self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
            n_entity, num_relations, num_bases, edge_index, edge_type,
            n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        self.dropout = nn.Dropout(p=0.1)

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gc = GCNModelSIGVAE(
            384, 768, 384, 384, 0.0,
            encsto='semi',
            ndist='Bernoulli',
            copyK=0,
            copyJ=0,
            device=self.device
        )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    # 构建无向邻接矩阵 (Tensor)
    def build_adjacency_matrix(self, node_list, edges):
        matrix_size = len(node_list)
        adjacency_matrix = torch.zeros((matrix_size, matrix_size), dtype=torch.float32)

        for edge in edges:
            if edge[0] in node_list and edge[1] in node_list:
                row = node_list.index(edge[0])
                col = node_list.index(edge[1])
                adjacency_matrix[row, col] = 1
                adjacency_matrix[col, row] = 1  # 确保无向边

        degrees = adjacency_matrix.sum(axis=1)
        degree_matrix_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        degree_matrix_inv_sqrt[degree_matrix_inv_sqrt == float('inf')] = 0
        normalized_adj_matrix = degree_matrix_inv_sqrt @ adjacency_matrix @ degree_matrix_inv_sqrt
        return normalized_adj_matrix

    # 将 edges 转换为 2 维数据
    def convert_edges_to_2d(self, edges):
        batch_size, num_edges = edges.shape

        assert num_edges % 2 == 0, "Number of edges must be even."

        # Reshape edges tensor to separate pairs
        edges_reshaped = edges.view(batch_size, -1, 2)

        return edges_reshaped

    # 构建无向邻接矩阵并进行规范化 (Tensor)
    def build_and_normalize_adjacency_matrix(self, node_list, edges):
        batch_size, num_nodes = node_list.shape
        adjacency_matrices = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.float32,
                                         device=node_list.device)

        # for b in range(batch_size):
        # row_indices = (node_list[b].unsqueeze(1) == edges[b][:, 0]).nonzero(as_tuple=True)[0]
        # col_indices = (node_list[b].unsqueeze(1) == edges[b][:, 1]).nonzero(as_tuple=True)[0]

        # a = edges[b][:, 0][:, None]
        # cc = edges[b][:, 1][:, None]
        # row_indices = torch.where(torch.eq(a, node_list[b]))[1]
        # col_indices = torch.where(torch.eq(cc, node_list[b]))[1]

        row_indices = self.get_index(node_list, edges[:, :, 0])
        col_indices = self.get_index(node_list, edges[:, :, 1])

        batch_indices = torch.arange(edges.shape[0]).unsqueeze(1).expand(-1, edges.shape[1])

        adjacency_matrices[batch_indices, row_indices, col_indices] = 1
        adjacency_matrices[batch_indices, col_indices, row_indices] = 1  # 确保无向边

        degrees = adjacency_matrices.sum(dim=-1)
        degrees = torch.where(degrees == 1, torch.full_like(degrees, 2), degrees)
        degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)  # 避免除以0

        with torch.no_grad():
            degree_matrix_inv_sqrt = torch.pow(degrees, -0.5).unsqueeze(2)
            degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0

        normalized_adj_matrices = degree_matrix_inv_sqrt * adjacency_matrices * degree_matrix_inv_sqrt.transpose(1, 2)

        # return adjacency_matrices, normalized_adj_matrices
        return normalized_adj_matrices

    def get_index(self, A, B):
        # 在batch维度上进行矢量化操作
        for i in range(A.shape[0]):
            assert set(B[i].tolist()).issubset(set(A[i].tolist())), f"Batch {i} failed!"
        sorted_A, sorted_indices = torch.sort(A, dim=1)

        sorted_A = sorted_A.contiguous()
        B = B.contiguous()

        # 对每个batch应用torch.searchsorted
        B_indices_in_sorted_A = torch.searchsorted(sorted_A, B, right=False)

        # 使用sorted_indices来获取B在A中的原始位置
        E = torch.gather(sorted_indices, 1, B_indices_in_sorted_A)

        return E

    def get_entity_embeds(self, node_list=None, edges=None, shape=None):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds

        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds.float()

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, prefix_embeds1=None, seq_len=None, node_list=None, edges=None, shape=None,
                rgcn_emb=None, cau_emb=None):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None

        mu, logvar, z, eps = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]

            entity_embeds = self.get_entity_embeds(node_list, edges, shape)

            # 这里将 shape设置为None是为了进行消融实验
            # shape = None
            rgcn_emb.append(entity_embeds[entity_ids].cpu().numpy())
            if shape is not None:
                select_entity_embeds = entity_embeds[node_list]
                edges_2d = self.convert_edges_to_2d(edges)
                adj = self.build_and_normalize_adjacency_matrix(node_list, edges_2d)
                mu, logvar, z, eps, entity_embeds = self.gc(select_entity_embeds, adj)
                E = self.get_index(node_list, entity_ids)

                expanded_B = E.unsqueeze(-1).expand(-1, -1, entity_embeds.size(2))
                entity_embeds = torch.gather(entity_embeds, 1, expanded_B)
                # entity_embeds = entity_embeds[E]
                cau_emb.append(entity_embeds.cpu().numpy())

            else:
                entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size) # 获取数据中 entity 的嵌入

        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(
                token_embeds) + token_embeds  # (batch_size, token_len, hidden_size) # 对话的隐藏状态 cls 进行线性变换
            token_embeds = self.token_proj2(token_embeds)

        # 这里将entity_embeds设置为None是为了进行消融实验
        # entity_embeds = None

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 # 这个 cross_attn 就是一个线性层 在论文中的公式对应于 W ，attn_wights对应于公式中的A
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds  # E' = E + TAt
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds  # T' = T + EA
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)

        '''
            在这里将 prompt_embeds 与 past_key_values 进行融合，形状都是(batch_size, seq_len, head_dim)
            
            特有提示为主，公共提示为辅
            
        '''
        # 方法1 进行乘积然后

        if prefix_embeds1 is not None:
            # fusion_weights = F.softmax(prompt_embeds @ prefix_embeds.permute(0, 2, 1), dim=2)
            # past_key_values = torch.cat([fusion_weights @ prefix_embeds, prompt_embeds], dim=1)
            #
            # past_key_values = past_key_values.reshape(
            #     batch_size, prompt_len * 2, self.n_layer, 2, self.n_head, self.head_dim
            # ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

            past_key_values = torch.cat([prefix_embeds1, prompt_embeds], dim=1)
            past_key_values = past_key_values.reshape(
                batch_size, prompt_len + seq_len, self.n_layer, 2, self.n_head, self.head_dim
            ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)
            past_key_values = self.dropout(past_key_values)
            return past_key_values
        else:
            prompt_embeds = prompt_embeds.reshape(
                batch_size, prompt_len, self.n_layer, 2, self.n_head, self.head_dim
            ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)
            # prompt_embeds = prompt_embeds.split(2)
            return mu, logvar, z, eps, prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config, seq_len):
        super(PrefixEncoder, self).__init__()
        # self.prefix_projection = config.prefix_projection
        self.prefix_projection = False
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.pre_seq_len = seq_len
        self.hidden_size = config.hidden_size
        self.entity_hidden_size = self.hidden_size // 2
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.embedding = torch.nn.Embedding(self.pre_seq_len,
                                            config.num_hidden_layers * 2 * config.hidden_size)

        # self.prompt_encoder = KGPrompt(self.hidden_size, self.hidden_size, self.n_head, self.n_layer, self.n_embd)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch_size, tune_flag=False):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        # 使用前缀编码器对前缀标记进行编码，得到过去的键值
        past_key_values = self.embedding(prefix_tokens)
        if tune_flag:
            # 重塑过去的键值张量的形状，使其符合预定的维度
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer,
                2,
                self.n_head,
                self.n_embd
            )
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 3, 0, 4, 1, 5])
        else:
            past_key_values = self.dropout(past_key_values)
        return past_key_values

    def init_entities_embeds(self, n_entity, num_relations, num_bases, edge_index, edge_type):
        self.kg_encoder = RGCNConv(self.entity_hidden_size, self.entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases).to(self.device)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, self.entity_hidden_size)).to(self.device)
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv).to(self.device)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False).to(self.device)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False).to(self.device)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(self.entity_hidden_size, self.entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.entity_hidden_size // 2, self.entity_hidden_size),
        ).to(self.device)
        self.entity_proj2 = nn.Linear(self.entity_hidden_size, self.hidden_size).to(self.device)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, prompt_len=0, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, head, query_len, key_len)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)

            if prompt_len > 0:
                key_length -= prompt_len

            # if query_length == key_length:
            #     causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            # else:
            #     causal_mask = self.bias[:, :, key_length - query_length: key_length,
            #                   key_length - query_length: key_length].bool()
            #     left_mask_shape = list(causal_mask.shape[:-1]) + [key_length - query_length]
            #     left_mask = causal_mask.new_ones(left_mask_shape)
            #     causal_mask = torch.cat([left_mask, causal_mask], dim=-1)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()

            if prompt_len > 0:
                left_mask_shape = list(causal_mask.shape[:-1]) + [prompt_len]
                left_mask = causal_mask.new_ones(left_mask_shape)
                causal_mask = torch.cat([left_mask, causal_mask], dim=-1)

            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            layer_past=None,
            prompt_embeds=None,  # (2, batch_size, head_num, left_prompt_len, head_size)
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        elif prompt_embeds is not None:
            key = torch.cat([prompt_embeds[0], key], dim=-2)
            value = torch.cat([prompt_embeds[1], value], dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        prompt_len = 0
        if prompt_embeds is not None:
            prompt_len = prompt_embeds.shape[-2]
        attn_output, attn_weights = self._attn(
            query, key, value, prompt_len, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            prompt_embeds=None,  # (2, batch_size, head_num, prefix_len, head_size)
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(  # 在rec训练阶段，只接受了 input_ids prompt_embeds attention_mask
            self,
            input_ids=None,
            past_key_values=None,
            prompt_embeds=None,  # (layer_num, 2, batch_size, head_num, prompt_len, head_dim)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # 这里将 use_caches 设置为了True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            if prompt_embeds is not None:
                prompt_attention_mask = prompt_embeds.new_ones((batch_size, prompt_embeds.shape[-2]))
                attention_mask = torch.cat([prompt_attention_mask, attention_mask],
                                           dim=-1)  # 在这里会对提示的attention_mask 进行拼接
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0  # 这里为什么会将attention_mask变为-0 这样会导致在注意力阶段忽略所有的输入

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # 模型并行执行 这里可以不用关注
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,  # hidden_states 是input_ids position_ids相加
                    layer_past=layer_past,
                    prompt_embeds=prompt_embeds[i] if prompt_embeds is not None else None,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:

                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2_crs(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # 冻结参数
        for param in self.transformer.parameters():
            param.requires_grad = False

        # self.pre_seq_len = seq_len
        self.hidden_size = config.hidden_size
        self.entity_hidden_size = self.hidden_size // 2

        '''
        下面是添加gp pp
        '''
        # self.local_pred = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size // 2, self.hidden_size)
        # )
        # self.global_pred = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size // 2, 1),
        #     nn.Sigmoid(),
        #     nn.Flatten(start_dim=0)
        # )
        # self.f = nn.Sigmoid()
        # self.gamma = 64
        # self.beta = -32
        self.gamma = 0
        self.beta = 0

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self, input_ids, past=None, prompt_embeds=None, layer_past=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "prompt_embeds": prompt_embeds,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "conv": True
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            prefix_embeds=None,
            prompt_embeds=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            rec=False,
            entity_embeds=None,
            rec_labels=None,
            conv=False,
            conv_labels=None,
            return_dict=True,

            global_pop=None,
            local_pop=None,
            is_train=True,
            seeker_id=None,
            gamma=0,
            beta=0,
            user_emb=None
    ):
        batch_size = input_ids.shape[0]
        # prefix_attention_mask = torch.ones(batch_size, past_key_values[0].shape[-2]).to(self.transformer.device)
        # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # entity_embeds.to(device)
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        rec_loss, rec_logits, re_rec_logits = None, None, None
        if rec:
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
            else:
                batch_size, sequence_length = inputs_embeds.shape[:2]
            assert (
                    self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."
            if input_ids is not None:
                if self.config.pad_token_id is None:
                    sequence_lengths = -1
                else:
                    sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

            rec_logits = hidden_states[range(batch_size), sequence_lengths]  # (bs, hidden_size)
            ''''''
            # 用户表示
            user_emb.append(rec_logits.cpu().numpy())
            # user_emb.append(rec_logits)
            ''''''
            # rec_logits @= entity_embeds.T  # (bs, n_item)
            rec_logits1 = torch.matmul(rec_logits, entity_embeds.T)

            # user_ci_emb = self.local_pred(rec_logits)
            # item_ci_emb = self.local_pred(entity_embeds)
            # pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
            # pred_global = self.global_pred(entity_embeds).expand(rec_logits1.shape)
            #
            real_local = local_pop[seeker_id]
            real_global = global_pop.expand(rec_logits1.shape)
            #
            # if is_train:
            #     re_rec_logits = rec_logits1 * (pred_local * pred_global)
            # else:
            #     re_rec_logits = rec_logits1 * (
            #             pred_local * pred_global) + gamma * real_local + beta * real_global

            re_rec_logits = rec_logits1 + gamma * real_local + beta * real_global
            # re_rec_logits = rec_logits1
            if rec_labels is not None:
                # loss_fct = CrossEntropyLoss()
                rec_loss = F.cross_entropy(re_rec_logits, rec_labels)

        loss, lm_logits = None, None
        if conv:
            lm_logits = self.lm_head(hidden_states)
            if conv_labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = conv_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return MultiOutput(
            conv_loss=loss,
            logits=lm_logits,
            rec_loss=rec_loss,
            rec_logits=re_rec_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


@dataclass
class MultiOutput(ModelOutput):
    conv_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    rec_loss: Optional[torch.FloatTensor] = None
    rec_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

