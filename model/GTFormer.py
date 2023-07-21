import torch.nn as nn
import torch

from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.transformer_encoder import Relative_Temporal_SelfAttention, Temporal_SelfAttention, Geopatial_SelfAttention, Spatial_SelfAttention, Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Temporal Layers
        self.temporal_embedding = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)

        if args.use_relativepos:
            temporal_selfattention = Relative_Temporal_SelfAttention(args.d_model, args.n_head, args.seq_len+1, args.save_attention)
        else:
            temporal_selfattention = Temporal_SelfAttention(args.d_model, args.n_head, args.save_attention)

        temporal_encoder_layers = [EncoderLayer(attention=temporal_selfattention,
                                                d_model=args.d_model,
                                                d_ff=args.d_model*4,
                                                dropout=args.dropout) for _ in range(args.temporal_num_layers)]

        temporal_norm = nn.LayerNorm(args.d_model)

        self.temporal_transformer_encoder = Encoder(temporal_encoder_layers, temporal_norm)

        self.temporal_linear = nn.Linear(args.d_model, args.num_tiles**2)



        #Spatial Layers
        self.spatial_embedding = TokenEmbedding_spatial(args.seq_len+1, args.d_model)

        if args.use_keyvaluereduction:
            spatial_selfattention = Geopatial_SelfAttention(args.d_model, args.n_head, args.save_attention)
        else:
            spatial_selfattention = Spatial_SelfAttention(args.d_model, args.n_head, args.save_attention)

        spatial_encoder_layers = [EncoderLayer(attention=spatial_selfattention,
                                               d_model=args.d_model,
                                               d_ff=args.d_model*4,
                                               dropout=args.dropout) for _ in range(args.spatial_num_layers)]

        spatial_norm = nn.LayerNorm(args.d_model)

        self.spatial_transformer_encoder = Encoder(spatial_encoder_layers, spatial_norm)

        self.spatial_linear = nn.Linear(args.d_model, args.seq_len+1)

        self.args = args



    def forward(self, X, key_indices):
        #B: batch size
        #L: sequence length
        #O: num origin
        #D: num destination
        B, L, O, D = X.shape

        X = torch.cat([X, torch.zeros([B, 1, O, D]).to(self.device)], dim=1).reshape(B, L+1, O*D)

        temp_in = self.temporal_embedding(X)
        temp_out, A_temporal = self.temporal_transformer_encoder(temp_in, key_indices)
        temp_out = self.temporal_linear(temp_out)

        spat_in = self.spatial_embedding(X.permute(0,2,1))
        spat_out, A_spatial = self.spatial_transformer_encoder(spat_in, key_indices)
        spat_out = self.spatial_linear(spat_out)

        out = temp_out.reshape(B, L+1, O, D) + spat_out.permute(0,2,1).reshape(B, L+1, O, D)

        if self.args.save_outputs:
            return out[:, -1:, :, :], A_temporal, A_spatial
        else:
            return out[:, -1:, :, :]