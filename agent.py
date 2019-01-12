import torch
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

from mdp import MDP
from memory import ReplayMemory
from models import DQN

class Agent:
    def __init__(self, n_inp, n_out, lr=0.001,
            alpha=0.2, hist_size=9, tau=0.6, eta=3,
            gamma=0.99, max_capacity=100000, bs=32):
        self.n_inp = n_inp
        self.n_out = n_out
        self.batch_size = bs
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = "saved_models/checkpoint.pth"

        self.online_dqn = DQN(n_inp + hist_size * n_out, n_out).to(self.device)
        self.target_dqn = DQN(n_inp + hist_size * n_out, n_out).to(self.device)
        self.memory = ReplayMemory(max_capacity, bs)
        self.mdp = MDP(self.device, n_out, alpha, tau, eta, hist_size)

        self.optimizer = optim.Adam(self.online_dqn.parameters(), lr=lr)
        self.update_eps_steps = 200000
        self.eps = 1

        self.load_models()

    def train(self, img, g):
        '''
        Train the network on a single training image with
        a single bounding box
        '''
        S = self.mdp.reset(img, g)
        done = False
        episode_loss = 0.0
        steps = 0
        total_R = 0
        while not done and steps < 50:
            steps += 1

            A = self.act(S, g)
            S_pr, R, done = self.mdp.step(A)
            S = S_pr
            total_R += R

            # store experience in replay buffer
            self.memory.push(S, A, R, S_pr, done)
            if len(self.memory) < 0.01 * self.memory.max_capacity:
                # the replay memory is not sufficiently populated yet
                continue

            # sample from memory and update params
            batch_S, batch_A, batch_R, batch_Spr, batch_done = \
                    self.memory.sample()

            var_S = Variable(batch_S).to(self.device)
            var_Spr = Variable(batch_Spr).to(self.device)
            var_A = Variable(batch_A).to(self.device)
            var_R = Variable(batch_R).to(self.device)
            var_done = Variable(batch_done).to(self.device)

            Q_S = self.online_dqn(var_S)
            Q_S_A = Q_S.gather(1, var_A.unsqueeze(1))
            Q_Spr_A, _ = self.target_dqn(var_Spr).max(dim=1)

            td_error = Q_S_A - (R + self.gamma * Q_Spr_A * (1 - var_done))
            loss = td_error.pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            episode_loss += loss.item()

        return steps, total_R, episode_loss

    def act(self, S, g, testing=False):
        self._update_eps()
        if np.random.uniform() < self.eps and not testing:
            A = self._explore(g)
        else:
            A = self._exploit(S)
        return A

    def _update_eps(self):
        if self.eps <= 0.1:
            self.eps = 0.1
        else:
            self.eps -= 0.9 / (1.0 * self.update_eps_steps)

    def _explore(self, g):
        # TODO: improve the exploration strategy as given in paper
        return np.random.randint(0, 9)

    def _exploit(self, S):
        S = Variable(torch.FloatTensor(S).unsqueeze(0)).to(self.device)
        with torch.no_grad():
            q_vals = self.online_dqn(S)
        A = q_vals.squeeze().argmax().item()
        return A

    def sync(self):
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())

    def save_models(self):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.online_dqn.state_dict(),
            'epsilon': self.eps
        }
        torch.save(state, self.model_save_path)

    def load_models(self):
        if os.path.exists(self.model_save_path):
            print("==> Loading pretrained weights... <==")
            checkpoint = torch.load(self.model_save_path, map_location=lambda storage, location: storage)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.online_dqn.load_state_dict(checkpoint['model'])
            self.eps = checkpoint['epsilon']
            self.sync()

    def _preview_image(self, img, g):
        import cv2
        np_img = img.squeeze().permute(1, 2, 0).numpy()
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        x1, y1, x2, y2 = g
        cv2.rectangle(np_img, (x1, y1), (x2, y2), color=(0, 255, 0))

        cv2.imshow("preview", np_img)
        cv2.waitKey(0)

    def test(self, img, g):
        self._preview_image(img, g)

        img = img.unsqueeze(0)
        S = self.mdp.reset(img, g)
        done = False
        while not done:
            A = self.act(S, g)
            S_pr, R, done = self.mdp.step(A)
            S = S_pr
            self._preview_image(img, self.mdp.running_box)
