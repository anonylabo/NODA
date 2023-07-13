import torch.nn as nn
import torch

from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.transformer_encoder import Temporal_SelfAttention, Spatial_SelfAttention, Encoder, EncoderLayer

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.pred_len = args.pred_len
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Temporal Layers
        self.temporal_embedding = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)

        temporal_selfattention = Temporal_SelfAttention(args.d_model, args.n_head, args.seq_len+args.pred_len)

        temporal_encoder_layers = [EncoderLayer(attention=temporal_selfattention,
                                                d_model=args.d_model,
                                                d_ff=args.d_model*4,
                                                dropout=args.dropout) for _ in range(args.temporal_num_layers)]

        temporal_norm = nn.LayerNorm(args.d_model)

        self.temporal_transformer_encoder = Encoder(temporal_encoder_layers, temporal_norm)

        self.temporal_linear = nn.Linear(args.d_model, args.num_tiles**2)



        #Spatial Layers
        self.spatial_embedding = TokenEmbedding_spatial(args.seq_len+args.pred_len, args.d_model)

        spatial_selfattention = Spatial_SelfAttention(args.d_model, args.n_head)

        spatial_encoder_layers = [EncoderLayer(attention=spatial_selfattention,
                                               d_model=args.d_model,
                                               d_ff=args.d_model*4,
                                               dropout=args.dropout) for _ in range(args.spatial_num_layers)]

        spatial_norm = nn.LayerNorm(args.d_model)

        self.spatial_transformer_encoder = Encoder(spatial_encoder_layers, spatial_norm)

        self.spatial_linear = nn.Linear(args.d_model, args.seq_len+args.pred_len)



    def forward(self, X, key_indices):
        B, L, O, D = X.shape

        X = torch.cat([X, torch.zeros([B, self.pred_len, O, D]).to(self.device)], dim=1).reshape(B, L+self.pred_len, O*D)

        temp_in = self.temporal_embedding(X)
        temp_out, A_temporal = self.temporal_transformer_encoder(temp_in, key_indices)
        temp_out = self.temporal_linear(temp_out)

        spat_in = self.spatial_embedding(X.permute(0,2,1))
        spat_out, A_spatial = self.spatial_transformer_encoder(spat_in, key_indices)
        spat_out = self.spatial_linear(spat_out)

        out = temp_out.reshape(B, L+self.pred_len, O, D) + spat_out.permute(0,2,1).reshape(B, L+self.pred_len, O, D)

        return out[:, -self.pred_len:, :, :], A_temporal, A_spatial