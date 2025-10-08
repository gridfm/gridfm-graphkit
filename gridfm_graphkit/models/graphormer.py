
from gridfm_graphkit.io.registries import MODELS_REGISTRY
import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_batch



@MODELS_REGISTRY.register("Graphormer")
class Graphormer(nn.Module):
    """
    A Graph Transformer model based on the Graphormer architecture

    This model directly modifies the attention between nodes based on
    its graph encodings. This requires padding the input nodes and propogating
    the associated mask as needed.

    Args:
        args (NestedNamespace): Parameters

    Attributes:
        n_node_features (int): Dimension of input node features. From ``args.model.input_dim``.
        hidden_dim (int): Hidden dimension size for all layers. From ``args.model.hidden_size``.
        output_dim (int): Dimension of the output node features. From ``args.model.output_dim``.
        n_encoder_layers (int): Number of transformer blocks. From ``args.model.num_layers``.
        num_heads (int): Number of attention heads. From ``args.model.attention_head``. Defaults to 1.
        dropout (float, optional): Dropout rate in attention blocks. From ``args.model.dropout``. Defaults to 0.0.
        mask_dim (int, optional): Dimension of the mask vector. From ``args.data.mask_dim``. Defaults to 6.
        mask_value (float, optional): Initial value for learnable mask parameters. From ``args.data.mask_value``. Defaults to -1.0.
        learn_mask (bool, optional): Whether to learn mask values as parameters. From ``args.data.learn_mask``. Defaults to False.
        edge_type (string, optional): Type of edge to consider multi_hop or not. From ``args.data.edge_type``. Defaults to multi_hop.
        multi_hop_max_dist (int, optional): Maximum number of hops to consider at edges. From ``args.data.multi_hop_max_dist``. Defaults to 20.

    """
    def __init__(self, args):
        super().__init__()

        self.n_node_features = args.model.input_dim
        self.hidden_dim = args.model.hidden_size
        self.output_dim = args.model.output_dim
        self.n_encoder_layers = args.model.num_layers
        self.num_heads = args.model.attention_head
        self.dropout = getattr(args.model, "dropout", 0.0) 
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", False)
        self.edge_type = getattr(args.model, "edge_type", "multi_hop") 
        self.multi_hop_max_dist = getattr(args.model, "multi_hop_max_dist", 20) 
        
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

        # model layers
        self.input_proj = nn.Linear(self.n_node_features, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        encoders = [
                EncoderLayer(
                        self.hidden_dim, 
                        self.hidden_dim, 
                        self.dropout, 
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
        
        # for positional embeddings
        self.spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            512, self.hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            512, self.hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(
                512 * self.n_edge_features + 1, num_heads, padding_idx=0)


    def compute_pos_embeddings(self, data):
        """
        Calculate Graphormer positional encodings, and attention biases

        Args:
            data (Data): Pytorch geometric Data/Batch object

        Returns:
            graph_node_feature (Tensor): data.x with positional encoding appended.
            graph_attn_bias (Tensor): attention bais terms.
        """
        attn_bias, spatial_pos, x = data.attn_bias, data.spatial_pos, data.x
        in_degree, out_degree = data.in_degree, data.in_degree
        
        # graph_attn_bias
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)

        graph_attn_bias = graph_attn_bias + spatial_pos_bias

        if data.edge_input is not None:
            edge_input, attn_edge_type = data.edge_input, data.attn_edge_type
            # edge feature
            # TODO flow over the upstream logic for edge_types...
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)
            graph_attn_bias = graph_attn_bias + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.input_proj(x)
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_node_feature = node_feature

        return graph_node_feature, graph_attn_bias


    def encoder(self, graph_node_feature, graph_attn_bias, mask=None, batch=1):

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, graph_attn_bias, mask=mask, batch=batch)
        output = self.encoder_final_ln(output)
        return output


    def forward(self, x, pe=None, edge_index=None, edge_attr=None, batch=None, data=None):
        """
        Forward pass for Graphormer.

        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            pe (Tensor): Positional encoding of shape [num_nodes, pe_dim].
            edge_index (Tensor): Edge indices for graph convolution.
            edge_attr (Tensor): Edge feature tensor.
            batch (Tensor): Batch vector assigning nodes to graphs.
            data (Data): Pytorch Geometric Data/Batch object.

        Returns:
            output (Tensor): Output node features of shape [num_nodes, output_dim].
        """

        # identify buffer nodes, and create a mask for them
        masked_entries = torch.sum(x < -1e8, axis=-1)
        mask = masked_entries >= 3  # due to masking up to feature 6 of 9
        
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(data)
        output = self.encoder(graph_node_feature, graph_attn_bias, mask=mask, batch=batch)
        output = self.decoder(output)

        # return the negative of the buffer mask to select data for loss calculation
        return output, ~mask


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
    """
    This is a slight modification of vanilla attention, to allow masking
    of buffer nodes, and the addition of biasses to the attention mechanism.
    """
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
    def __init__(self, hidden_size, ffn_size, dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None, batch=1):
        """
        It is assumed that the mask is 1 where values are to be ignored
        and then 0 where there are valid data
        """
        x, _ = to_dense_batch(x, batch) 
        mask, _ = to_dense_batch(mask, batch)

        y = self.self_attention_norm(x)
        attn_bias = attn_bias.squeeze()
        y = self.self_attention(y, y, y, attn_bias, mask)
        y = self.self_attention_dropout(y)
        x = x + torch.reshape(y, x.size())

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x=x.flatten(0,1)

        return x
