
from gridfm_graphkit.io.registries import MODELS_REGISTRY
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from torch.nn import functional as F

from torch_geometric.utils import to_dense_batch

# from gridfm_graphkit.datasets.transforms import AddGraphormerEncodings


@MODELS_REGISTRY.register("Graphormer")
class Graphormer(nn.Module):
    """
    TODO fill in description
    """
    def __init__(
        self,
        # n_encoder_layers,
        # n_decoder_layers,
        # num_heads,
        # hidden_dim,
        # dropout_rate,
        # intput_dropout_rate,
        # weight_decay,
        # ffn_dim,
        # dataset_name,
        # warmup_updates,
        # tot_updates,
        # peak_lr,
        # end_lr,
        # attention_dropout_rate,
        # n_node_features,
        # mask_ratio,
        # n_val_sampler,
        args
    ):
        super().__init__()

        self.n_node_features = args.model.input_dim
        self.num_heads = 8  # TODO make this configurable or to match their structure
        self.hidden_dim = args.model.hidden_size
        self.n_encoder_layers = args.model.num_layers
        intput_dropout_rate = 0.3
        dropout_rate = 0.3
        attention_dropout_rate = 0.3

        # variables flown over from GPS TODO check
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", True)
        self.output_dim = args.model.output_dim

        if self.learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(self.mask_dim) + self.mask_value,
                requires_grad=True,
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(self.mask_dim) + self.mask_value,
                requires_grad=False,
            )

        self.input_proj = nn.Linear(self.n_node_features, self.hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [
                EncoderLayer(
                        self.hidden_dim, 
                        self.hidden_dim, 
                        dropout_rate, 
                        attention_dropout_rate, 
                        self.num_heads
                        )
                    for _ in range(self.n_encoder_layers)
                    ]
        self.encoder_layers = nn.ModuleList(encoders)
        self.encoder_final_ln = nn.LayerNorm(self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        

        # for pos embeddings
        self.spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            512, self.hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            512, self.hidden_dim, padding_idx=0)

        # self.loss_fn = F.mse_loss # TODO remove eventually as they are specd elsewhere
        # self.masking_value = -4

    def compute_pos_embeddings(self, batched_data, batch):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree

        # gr_transform = AddGraphormerEncodings(
        #         attr_name="gr",
        #     )
        # batched_data = gr_transform(batched_data)
        # attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        
        
        # print('--->', attn_bias.size(), attn_bias.device, batch.size())

        # yy0, mask = to_dense_batch(attn_bias, batch=batch, max_num_nodes=2000, batch_size=8)
        # yy1, mask = to_dense_batch(spatial_pos, batch=batch, max_num_nodes=2000, batch_size=8)

        # attn_bias = yy0
        # spatial_pos = yy1

        # print('yyyyyy', yy0.size(), yy1.size(), x.size())

        # attn_bias = attn_bias.reshape(8,-1)
        # spatial_pos = spatial_pos.reshape(8,-1)
        # print('-----', attn_bias.size(), spatial_pos.size(), x.size())
        # odim = int(torch.sqrt(torch.as_tensor(attn_bias.size(-1))).item())
        # print('oooo', odim)
        # attn_bias = attn_bias.reshape(-1,odim,odim)
        # spatial_pos = spatial_pos.reshape(-1,odim,odim)
        # print('-----', attn_bias.size(), spatial_pos.size(), x.size())
        
        # graph_attn_bias
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]
        # print('aaaaaaaaaa', graph_attn_bias.size(), graph_attn_bias.device)

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # print('sssssssssss', spatial_pos_bias.size(), spatial_pos_bias.device)

        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.input_proj(x)
        # print('nf>>', node_feature.size(), in_degree.size(), out_degree.size(), self.in_degree_encoder(in_degree).size())
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_node_feature = node_feature

        return graph_node_feature, graph_attn_bias


    def encoder(self, graph_node_feature, graph_attn_bias, mask=None, batch=1):

        graph_node_feature_masked = graph_node_feature  #TODO simplify this
        graph_attn_bias_masked = graph_attn_bias

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature_masked)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, graph_attn_bias_masked, mask=mask, batch=batch)
        output = self.encoder_final_ln(output)
        return output

    def forward(self, x, pe, edge_index, edge_attr, batched_data, data):
        """
        process a batch of data, applying the input mask, while
        excluding non-valid values that arrise during processing

        mask: incoming values to mask for prediction
        """
        # print('***batch***', data)
        # print(x.size(), batched_data)
        # print(data.attn_bias.size(), data.spatial_pos.size())

        mask = None
        masked_entries = torch.sum(x < -100, axis=-1)  #TODO make this mesh with normalizn
        mask = masked_entries == x.size(-1)
        print('pad mask >>>', mask.size(), mask.sum())

        # TODO note that the x, pe are redundant or not needed, so clean up at the end

        # TODO in the baseline code the PE is an input here and passes through
        # a normalization before being concatenated to the features, follow this in final version
        
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(data, batched_data)
        # print('gnodes********', graph_node_feature.size(), graph_attn_bias.size())
        output = self.encoder(graph_node_feature, graph_attn_bias, mask=mask, batch=batched_data)
        output = self.decoder(output)

        return output


# TODO maybe set this as the decoder
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):

        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # print('**********',
        #      x.size(), q.size(), 
        #      k.size(), v.size(), 
        #      attn_bias.size(), mask.size()
        #      )
        if attn_bias is not None:
            if mask is not None:
                usm0 = mask.unsqueeze(1).unsqueeze(3)
                usm1 = mask.unsqueeze(1).unsqueeze(2)

                attn_bias = attn_bias.masked_fill(usm0 == 1, 0.0)
                attn_bias = attn_bias.masked_fill(usm1 == 1, 0.0)
            x = x + attn_bias

        # mask the data before the softmax
        if mask is not None:
            usm0 = mask.unsqueeze(1).unsqueeze(2)
            x = x.masked_fill(usm0 == 1, -1e9)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None, batch=1):
        """
        It is assumed that the mask is 1 where values are to be ignored
        and then 0 where there are valid data
        """

        # print('xxxxxxxxxxxxx', x.size(), batch.size())
        x, bmask = to_dense_batch(x, batch) # TODO remove bmask if padding remains in final
        mask, _ = to_dense_batch(mask, batch)

        y = self.self_attention_norm(x)
        # print(y.size(), attn_bias.size(), batch)
        
        attn_bias = attn_bias.squeeze()
        # attn_bias = attn_bias.permute(1, 2, 0)
        # attn_bias, maska = to_dense_batch(attn_bias, batch)
        # print('dense>>>', y.size(), bmask.size(), attn_bias.size())
        # print('msum>>>', bmask.sum(dim=-1), )
        # print('msum2>>', mask.size(),mask.sum(dim=-1))
        y = self.self_attention(y, y, y, attn_bias, mask)
        y = self.self_attention_dropout(y)
        # print('<<<<<>>>>', x.size(), y.size())
        x = x + torch.reshape(y, x.size())

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x=x.flatten(0,1)
        # print('222222222222222', x.size())
        return x
