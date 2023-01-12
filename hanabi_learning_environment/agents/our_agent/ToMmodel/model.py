import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer


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
        self.model = Transformer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,activation=activation,
            num_encoder_layers=num_layers,num_decoder_layers=num_layers)

        self.dataconfig = dataconfig

        self.fc_card = nn.Linear(25, d_model)
        self.output_card = nn.Linear(d_model, 25)

    def forward(self, data):
        act_seq, obs = data

        act_seq # batch, len, dim
        act_seq = act_seq.permute(1,0,2) # len, batch, dim
        act_seq = self.fc_card(act_seq) # len, batch, d_model

        obs # batch, len, dim
        obs = obs.permute(1,0,2) # len, batch, dim
        obs = self.fc_card(obs) # len, batch, d_model

        output = self.model(act_seq, obs) # len, batch, d_model

        output = output[5:5+self.dataconfig.num_cards] # num_cards, batch, d_model
        output = output.permute(1,0,2) # batch, num_cards, d_model
        output = self.output_card(output) # batch, num_cards, 25

        return output

    def load_ckpt(self,path):
        self.load_state_dict(torch.load(path))
        return self
