import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Adam

from model import ToM2HModel
from dataset import ToMDataset
from tqdm import tqdm
import os
import json
import numpy as np
import argparse


def train(model, dataloader, optimizer, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    color_sum = 0.
    rank_sum = 0.
    loss_sum = 0.
    acc_sum = 0
    acc_color_sum = 0
    acc_rank_sum = 0
    data_num = 0
    for data, target_color, target_rank, target_card in tqdm(dataloader):
        optimizer.zero_grad()
        act_seq, obs = data
        act_seq = act_seq.to(device)
        obs = obs.to(device)
        pred_color, pred_rank = model((act_seq, obs))
        target_color = target_color.to(device) - 1
        target_rank = target_rank.to(device) - 1
        # target_card = target_card.to(device)

        # pred_card = pred_card.permute(0,2,1) # (B,25,len)
        # print(pred_card.shape, target_card.shape, target_color.shape)
        pred_color = pred_color.permute(0, 2, 1)
        pred_rank = pred_rank.permute(0, 2, 1)

        loss_color = loss_fn(pred_color, target_color)
        loss_rank = loss_fn(pred_rank, target_rank)

        loss = loss_color + loss_rank
        loss.backward()
        optimizer.step()

        acc_color = (torch.argmax(pred_color, dim=1) == target_color)
        acc_rank = (torch.argmax(pred_rank, dim=1) == target_rank)
        acc = (acc_color & acc_rank)

        data_num += acc.shape[0]

        acc_sum += acc.sum().item()
        acc_color_sum += acc_color.sum().item()
        acc_rank_sum += acc_rank.sum().item()

        acc = acc.float().mean()
        acc_color = acc_color.float().mean()
        acc_rank = acc_rank.float().mean()

        loss_sum += loss.item()*act_seq.shape[0]

        tqdm.write(f'loss: {loss.item():.4f}, loss_avg: {loss_sum/data_num:.4f}, acc: {acc.item():.4f}, '
                   f'acc_avg: {acc_sum/data_num/5:.4f}, acc_cl_avg: {acc_color_sum/data_num/5:.4f}, '
                   f'acc_rk_avg: {acc_rank_sum/data_num/5:.4f}'
                   )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--look_back', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='tom_ckpt_2heads')
    parser.add_argument('--data_path', type=str,
                        default='/home/yilue/datasets/files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from dataset import Config
    config = Config()
    config.num_players = 2
    # config.output_start = 5
    config.output_start = 5 + config.num_players * config.num_cards
    config.vec_dim = 10
    dataset = ToMDataset(args.data_path, args.look_back, heads=2,  config=config, max_data_num=10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    with open('sample.txt', 'w') as f:
        print(dataset[0], file=f)


    return

    model = ToM2HModel(dataconfig=dataset.config)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, dataloader, optimizer, args.device)
        torch.save(model.state_dict(), os.path.join(
            args.save_path, f'tom_{epoch}.pth'))


if __name__ == '__main__':
    main()
