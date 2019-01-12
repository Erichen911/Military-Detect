import torch
from torch.utils import data
import argparse

from agent import Agent
from dataset import VOCDataset

parser = argparse.ArgumentParser()
parser.add_argument('--testing', action='store_true',
        default=False, help="train/test run?")
args = parser.parse_args()

dataset = VOCDataset()
print(dataset.__len__())
generator = data.DataLoader(dataset, shuffle=True, batch_size=1)

n_inp = 2048
n_out = 9
n_epochs = 100

agent = Agent(n_inp, n_out)
total_steps = 0

if not args.testing:
    for ep in range(n_epochs):
        for i, (x, g) in enumerate(generator):
            steps, rewards, loss = agent.train(x, g)
            # if i % 100 == 0:
            print("Epoch/Episdoe: %3d/%4d - Loss: %.4f - Steps: %3d - Rewards: %3d" %(ep, i, loss, steps, rewards))

            total_steps += steps
            if total_steps >= 1000:
                # sync online and target DQN
                agent.sync()
                total_steps = 0

        # save model after every epoch
        agent.save_models()
else:
    img, g = dataset.random_test()
    agent.test(img, g)
