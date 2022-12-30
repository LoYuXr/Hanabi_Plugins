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
    data_num = 0
    for data, target_color, target_rank in tqdm(dataloader):
        optimizer.zero_grad()
        act_seq, obs = data
        act_seq = act_seq.to(device)
        obs = obs.to(device)
        pred_color, pred_rank = model((act_seq, obs))
        target_color = target_color.to(device)
        target_rank = target_rank.to(device)

        pred_color = pred_color.permute(0, 2, 1)
        pred_rank = pred_rank.permute(0, 2, 1)

        loss_color = loss_fn(pred_color, target_color)
        loss_rank = loss_fn(pred_rank, target_rank)
        loss = loss_color + loss_rank
        loss.backward()
        optimizer.step()

        acc_color = (torch.argmax(pred_color, dim=1)
                     == target_color).float().mean()
        acc_rank = (torch.argmax(pred_rank, dim=1)
                    == target_rank).float().mean()

        color_sum += acc_color.item()*len(data)
        rank_sum += acc_rank.item()*len(data)
        loss_sum += loss.item()*len(data)
        data_num += len(data)

        tqdm.write(f'loss_color: {loss_color.item():.4f}, loss_rank: {loss_rank.item():.4f}, loss: {loss.item():.4f}, color_avg: {color_sum/data_num:.4f}, rank_avg: {rank_sum/data_num:.4f}, loss_avg: {loss_sum/data_num:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--look_back', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='tom_ckpt')
    parser.add_argument('--data_path', type=str,
                        default='/home/yilue/datasets/pick_best_400')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    dataset = ToMDataset(args.data_path, args.look_back)
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
