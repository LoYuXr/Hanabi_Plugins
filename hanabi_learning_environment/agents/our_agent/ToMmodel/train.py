import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Adam

from model import ToMModel
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
    data_num = 0
    for data, target_color, target_rank, target_card in tqdm(dataloader):
        optimizer.zero_grad()
        act_seq, obs = data
        act_seq = act_seq.to(device)
        obs = obs.to(device)
        pred_card = model((act_seq, obs))
        target_color = target_color.to(device)
        target_rank = target_rank.to(device)
        target_card = target_card.to(device)

        pred_card = pred_card.permute(0,2,1) # (B,25,len)
        # print(pred_card.shape, target_card.shape, target_color.shape)

        loss = loss_fn(pred_card, target_card)
        loss.backward()
        optimizer.step()

        acc = (torch.argmax(pred_card, dim=1)
                     == target_card)
        data_num += acc.shape[0]

        acc_sum += acc.sum().item()
        acc = acc.float().mean()

        loss_sum += loss.item()*act_seq.shape[0]

        tqdm.write(f'loss: {loss.item():.4f}, loss_avg: {loss_sum/data_num:.4f}, acc: {acc.item():.4f}, '
                   f'acc_avg: {acc_sum/data_num:.4f}'
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--look_back', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='tom_ckpt')
    parser.add_argument('--data_path', type=str,
                        default='/home/yilue/datasets/files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from dataset import Config
    config = Config()
    config.num_players = 2
    # config.output_start = 5
    config.output_start = 5

    dataset = ToMDataset(args.data_path, args.look_back, config=config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ToMModel(dataconfig=dataset.config)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, dataloader, optimizer, args.device)
        torch.save(model.state_dict(), os.path.join(
            args.save_path, f'tom_{epoch}.pth'))


if __name__ == '__main__':
    main()
