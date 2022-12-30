import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class ToMModel(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=256,
                 nhead=8,
                 dim_feedforward=128,
                 dropout=0.1,
                 activation='relu',
                 dataconfig=None
                 ):
        super(ToMModel, self).__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                6*d_model, nhead, dim_feedforward*6, dropout, activation),
            num_layers
        )
        self.embeddings = nn.ModuleList([
            nn.Embedding(dataconfig.num_players, d_model),
            nn.Embedding(dataconfig.num_actions, d_model),
            nn.Embedding(dataconfig.num_cards+1, d_model),
            nn.Embedding(dataconfig.num_colors+1, d_model),
            nn.Embedding(dataconfig.num_ranks+1, d_model),
            # nn.Embedding(dataconfig.num_offset, d_model),
        ])

        self.fc_obs = nn.Linear(dataconfig.obs_dim, d_model)
        self.fc_color_1 = nn.ModuleList([
            nn.Linear(d_model*6, d_model) for _ in range(dataconfig.num_cards)
        ])
        self.fc_color_2 = nn.ModuleList([
            nn.Linear(d_model, dataconfig.num_colors+1) for _ in range(dataconfig.num_cards)
        ])
            
        self.fc_rank_1 = nn.ModuleList([
            nn.Linear(d_model*6, d_model) for _ in range(dataconfig.num_cards)
        ])
        self.fc_rank_2 = nn.ModuleList([
            nn.Linear(d_model, dataconfig.num_ranks+1) for _ in range(dataconfig.num_cards)
        ])
        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        # data: (act_seq, cur_obs)
        act_seq, cur_obs = data
        # act_seq: (batch_size, look_back, 5)
        # cur_obs: (batch_size, 658)
        act_seq = act_seq.permute(1, 0, 2)
        # act_seq: (look_back, batch_size, 5)
        act_seq = torch.cat([
            self.embeddings[i](act_seq[:, :, i])
            for i in range(act_seq.shape[2])
        ], dim=2)
        # act_seq: (look_back, batch_size, d_model*5)
        obs = self.fc_obs(cur_obs)
        x = torch.cat([act_seq, obs.unsqueeze(
            0).repeat(act_seq.shape[0], 1, 1)], dim=2)
        x = self.encoder(x)
        x = x[-1]  # (batch_size, d_model*6)

        color = torch.stack([
            self.fc_color_2[i](F.relu(self.fc_color_1[i](x))) for i in range(len(self.fc_color_1))
        ], dim=1)  # (batch_size, card_per_player, num_colors-1)
        rank = torch.stack([
            self.fc_rank_2[i](F.relu(self.fc_rank_1[i](x))) for i in range(len(self.fc_rank_1))
        ], dim=1)
        return color, rank
