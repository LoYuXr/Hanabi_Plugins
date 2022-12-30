import os
import json
import torch
from torch.utils.data.dataset import Dataset


class Config(object):
    num_players = 2
    num_colors = 5
    num_ranks = 5
    num_cards = 5
    num_actions = 5
    obs_dim = 658


class ToMDataset(Dataset):
    def __init__(self, path, look_back=10):
        super(ToMDataset, self).__init__()
        self.look_back = look_back
        self.path = path
        self.color2idx = {
            'R': 1,
            'Y': 2,
            'G': 3,
            'W': 4,
            'B': 5,
        }

        self.config = Config()
        jsonfiles = os.listdir(path)
        self.data = []
        for jsonfile in jsonfiles[:100]:
            print(jsonfile)
            with open(os.path.join(path, jsonfile)) as f:
                self.data.extend(self.preprocess(json.load(f)))

    def preprocess(self, datas):
        act_seq = []
        ret_data = []
        for data in datas[:self.look_back]:
            cur_player = int(data['current_player'])
            cur_act = data['player_action']
            cur_act = torch.tensor([
                int(data['current_player']),
                int(cur_act['action_type']),
                int(cur_act['card_index']) +
                1 if cur_act['card_index'] is not None else 0,
                int(cur_act['color']) +
                1 if cur_act['color'] in self.color2idx else 0,
                int(cur_act['rank'])+1 if cur_act['rank'] is not None else 0,
            ])
            act_seq.append(cur_act)
        for i, data in enumerate(datas[self.look_back:]):
            cur_player = int(data['current_player'])
            cur_obs = data['player_observations'][cur_player]
            cur_obs = torch.tensor(cur_obs['vectorized'], dtype=torch.float32)
            myhand = data['player_observations'][(
                cur_player+1) % len(data['player_observations'])]
            myhand = myhand['observed_hands'][cur_player]
            myhand.extend([{'color': None, 'rank': None} for _ in range(self.config.num_cards-len(myhand))])
            target_color = torch.tensor(
                [self.color2idx[card['color']] if card['color'] in self.color2idx else 0 for card in myhand])
            target_rank = torch.tensor(
                [int(card['rank'])+1 if card['rank'] is not None else 0 for card in myhand])
            ret_data.append({
                'data': (torch.stack(act_seq), cur_obs),
                'target_color': target_color,
                'target_rank': target_rank,
            })
            act_seq.pop(0)
            cur_act = data['player_action']
            cur_act = torch.tensor([
                int(data['current_player']),
                int(cur_act['action_type']),
                int(cur_act['card_index']) +
                1 if cur_act['card_index'] is not None else 0,
                int(cur_act['color']) +
                1 if cur_act['color'] in self.color2idx else 0,
                int(cur_act['rank'])+1 if cur_act['rank'] is not None else 0,
            ])
            act_seq.append(cur_act)
        return ret_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert self.data[index]['data'][0].shape[0] == self.look_back
        # assert self.data[index]['target_color'].max() <= self.config.num_colors
        # assert self.data[index]['target_rank'].max() <= self.config.num_ranks
        # assert self.data[index]['target_color'].min() >= 1
        # assert self.data[index]['target_rank'].min() >= 1
        assert self.data[index]['target_color'].shape[0] == self.config.num_cards
        assert self.data[index]['target_rank'].shape[0] == self.config.num_cards
        return self.data[index]['data'], self.data[index]['target_color'], self.data[index]['target_rank']
