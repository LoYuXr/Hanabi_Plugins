import os
import json
import torch
from torch.utils.data.dataset import Dataset


class Config(object):
    num_players = 2 # 3 or 4
    num_colors = 5
    num_ranks = 5
    num_cards = 5
    num_actions = 5
    obs_dim = 658
    vec_dim = 25
    output_start = 5


color2idx = {
    'R': 1,
    'Y': 2,
    'G': 3,
    'W': 4,
    'B': 5,
}
idx2colorfunc = ['R', 'Y', 'G', 'W', 'B']

def idx2color(idx):
    if idx is None:
        return None
    color = idx2colorfunc[idx]
    return color


def encode_card(color, rank):
    if color is None:
        if rank is None or rank == -1:
            card = [1.0/25 for _ in range(25)]
        else:
            card = [1.0/5 if i // 5 == rank else 0 for i in range(25)]
    else:
        if rank is None:
            card = [1.0/5 if i %
                    5 == color2idx[color]-1 else 0 for i in range(25)]
        else:
            card = [1.0 if i % 5 == color2idx[color] -
                    1 and i // 5 == rank else 0 for i in range(25)]

    return card

def encode_card_2head(color, rank):
    card = [0.0 for _ in range(10)]
    if color is not None:
        card[color2idx[color]-1] = 1.0
    if rank is not None and rank in range(5):
        card[5+rank] = 1.0
    return card

class ToMDataset(Dataset):
    def __init__(self, path, look_back=10, max_data_num=None, heads = 1, config=None):
        super(ToMDataset, self).__init__()
        self.look_back = look_back
        self.path = path
        self.color2idx = color2idx
        self.encode_card = encode_card_2head if heads == 2 else encode_card
        self.idx2color = idx2color
        self.max_discard = 20

        self.config = config if config is not None else Config()
        jsonfiles = os.listdir(path)
        if max_data_num is not None:
            jsonfiles = jsonfiles[:max_data_num]
        self.data = []
        for jsonfile in jsonfiles:
            # print(jsonfile)
            with open(os.path.join(path, jsonfile)) as f:
                self.data.extend(self.preprocess(json.load(f)))

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
                [self.color2idx[card['color']] if card['color']
                    in self.color2idx else 0 for card in myhand]
                + [0 for _ in range(self.config.num_cards-len(myhand))])
            target_rank = torch.tensor(
                [int(card['rank'])+1 if card['rank']
                 is not None else 0 for card in myhand]
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

    def process_obs(self, obs):

        ret = []
        fireworks = obs['fireworks']
        fireworks = [self.encode_card(color, rank)
                     for color, rank in fireworks.items()]
        observed = obs['observed_hands']
        observed.extend([[] for _ in range(self.config.num_players-len(observed))])
        obh = []
        for hand in observed:
            obh.extend([self.encode_card(card['color'], card['rank'])
                       for card in hand])
            obh.extend([self.encode_card(None, None)
                       for _ in range(self.config.num_cards-len(hand))])
        discard = obs['discard_pile']
        discard = [self.encode_card(card['color'], card['rank'])
                   for card in discard]
        discard = discard[-self.max_discard:]
        discard.extend([self.encode_card(None, None)
                       for _ in range(self.max_discard-len(discard))])
        cardknow = obs['card_knowledge']
        cardknow.extend([[] for _ in range(self.config.num_players-len(cardknow))])
        ck = []
        for hand in cardknow:
            ck.extend([self.encode_card(card['color'], card['rank'])
                      for card in hand])
            ck.extend([self.encode_card(None, None)
                      for _ in range(self.config.num_cards-len(hand))])

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


def data_process(act_seq, obs, my_id):
    '''
    act_seq: json dict, len = lookback, need add action player id in it 
    obs: json dict, len = 1 
    my_id: int, my player id

    Note : the model accept torch.tensor(return value of this func)
    '''

    config = Config()
    max_discard = 20 # {2:20,3:20,4:10} for different num_players 2,3 or 4

    def process_obs(obs):

        ret = []
        fireworks = obs['fireworks']
        fireworks = [encode_card(color, rank)
                     for color, rank in fireworks.items()]
#         print(torch.tensor(fireworks).shape)
        observed = obs['observed_hands']
        obh = []
        for hand in observed:
            obh.extend([encode_card(card['color'], card['rank'])
                       for card in hand])
            obh.extend([encode_card(None, None)
                       for _ in range(config.num_cards-len(hand))])
#         print(torch.tensor(obh).shape)
        discard = obs['discard_pile']
        discard = [encode_card(card['color'], card['rank'])
                   for card in discard]
        discard = discard[-max_discard:]
        discard.extend([encode_card(None, None)
                       for _ in range(max_discard-len(discard))])
#         print(torch.tensor(discard).shape)
        cardknow = obs['card_knowledge']
        ck = []
        for hand in cardknow:
            ck.extend([encode_card(card['color'], card['rank'])
                      for card in hand])
            ck.extend([encode_card(None, None)
                      for _ in range(config.num_cards-len(hand))])
#         print(torch.tensor(ck).shape)

        ret.extend(fireworks)
        ret.extend(obh)
        ret.extend(ck)
        ret.extend(discard)

        return ret

    def process_act(raw_seq, my_id):
        act_seq = []
        myself = my_id
        for i in range(len(raw_seq)):
            cur_player = int(raw_seq[i]['current_player'])
            cur_act = raw_seq[i]['player_action']
            if cur_act['action_type'] in [3, 4] and \
                    (cur_act['target_offset']+cur_player) % config.num_players == myself:
                act_seq.append(encode_card(
                    idx2color(cur_act['color']), cur_act['rank']))
            else:
                act_seq.append(encode_card(None, None))
        return act_seq
    
    return (process_act(act_seq, my_id), process_obs(obs))


if __name__ == '__main__':
    dataset = ToMDataset('/home/yilue/datasets/pick_best_400')
    (dataset[1][0])
