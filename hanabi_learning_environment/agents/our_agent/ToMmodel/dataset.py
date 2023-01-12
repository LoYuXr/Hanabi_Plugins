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
        self.idx2colorfunc= ['R', 'Y', 'G', 'W', 'B']
        self.max_discard = 20

        self.config = Config()
        jsonfiles = os.listdir(path)
        self.data = []
        for jsonfile in jsonfiles[:100]:
            # print(jsonfile)
            with open(os.path.join(path, jsonfile)) as f:
                self.data.extend(self.preprocess(json.load(f)))

    def idx2color(self, idx):
        if idx is None:
            return None
        color = self.idx2colorfunc[idx]
        return color

    def preprocess(self, datas):
        ret_data = []

        for i, data in enumerate(datas[self.look_back:]):
            cur_player = int(data['current_player'])
            cur_obs = data['player_observations'][cur_player]
            cur_obs = self.process_obs(cur_obs)

            myhand = data['player_observations'][(
                cur_player+1) % len(data['player_observations'])]
            myhand = myhand['observed_hands'][-1]
            myhand.extend([{'color': None, 'rank': None}
                          for _ in range(self.config.num_cards-len(myhand))])
            myhandloc = 5



            target_color = torch.tensor(
                [self.color2idx[card['color']] if card['color'] in self.color2idx else 0 for card in myhand]
                + [0 for _ in range(self.config.num_cards-len(myhand))])
            target_rank = torch.tensor(
                [int(card['rank'])+1 if card['rank'] is not None else 0 for card in myhand]
                + [0 for _ in range(self.config.num_cards-len(myhand))])
            target_card = (target_rank-1)*5 + target_color-1

            act_seq = self.process_act(datas, i+self.look_back)
            

            ret_data.append({
                'data': (act_seq, cur_obs),
                'target_color': target_color,
                'target_rank': target_rank,
                'target_card': target_card,
            })

        return ret_data

    def encode_card(self, color, rank):
        if color is None:
            if rank is None or rank == -1:
                card = [1.0/25 for _ in range(25)]
            else:
                card = [1.0/5 if i // 5 == rank else 0 for i in range(25)]
        else:
            if rank is None:
                card = [1.0/5 if i % 5 == self.color2idx[color]-1 else 0 for i in range(25)]
            else:
                card = [1.0 if i % 5 == self.color2idx[color]-1 and i // 5 == rank else 0 for i in range(25)]
        
        return card

    def process_obs(self, obs):

        ret = []
        fireworks = obs['fireworks']
        fireworks = [self.encode_card(color, rank)
                     for color, rank in fireworks.items()]
        observed = obs['observed_hands']
        obh = []
        for hand in observed:
            obh.extend([self.encode_card(card['color'], card['rank'])
                       for card in hand])
            obh.extend([self.encode_card(None,None) for _ in range(self.config.num_cards-len(hand))])
        discard = obs['discard_pile']
        discard = [self.encode_card(card['color'], card['rank'])
                   for card in discard]
        discard = discard[-self.max_discard:]
        discard.extend([[1.0/25 for _ in range(25)]
                       for _ in range(self.max_discard-len(discard))])
        cardknow = obs['card_knowledge']
        ck = []
        for hand in cardknow:
            ck.extend([self.encode_card(card['color'], card['rank'])
                      for card in hand])
            ck.extend([self.encode_card(None,None) for _ in range(self.config.num_cards-len(hand))])

        ret.extend(fireworks)
        ret.extend(obh)
        # assert len(obh) % self.config.num_cards == 0
        ret.extend(ck)
        # assert len(ck) % self.config.num_cards == 0
        ret.extend(discard)

        ret = torch.tensor(ret)
        assert len(ret.shape) == 2
        return ret

    def process_act(self, datas, idx):
        act_seq = []
        myself = int(datas[idx]['current_player'])
        for i in range(idx-self.look_back, idx):
            cur_player = int(datas[i]['current_player'])
            cur_act = datas[i]['player_action']
            if cur_act['action_type'] in [3, 4] and \
                    (cur_act['target_offset']+cur_player) % len(datas[i]['player_observations']) == myself:
                act_seq.append(self.encode_card(
                    self.idx2color(cur_act['color']), cur_act['rank']))
            else:
                act_seq.append(self.encode_card(None, None))
        return torch.tensor(act_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # obs = self.data[index]['data'][1]
        # idx = self.data[index]['myhandloc']
        # for i in range(idx, idx+5):
        #     if not (torch.linalg.norm(obs[idx]-torch.ones(25)*1.0/25, ord=1) < 0.01):
        #         print(obs[idx],idx)
        #         exit()

        return self.data[index]['data'], self.data[index]['target_color'], \
            self.data[index]['target_rank'], self.data[index]['target_card']

if __name__ == '__main__':
    dataset = ToMDataset('/home/yilue/datasets/pick_best_400')
    (dataset[1][0])