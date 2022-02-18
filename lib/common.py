import collections
import numpy as np

import torch
import torch.nn as nn   

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a,dtype=torch.float32).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss( batch, net, target_net, gamma, device="cpu"):
          states, actions, rewards, dones, next_states = batch

          states_v = torch.tensor(np.array(states, copy=False),dtype=torch.float32).to(device)
          next_states_v = torch.tensor(np.array(next_states, copy=False),dtype=torch.float32).to(device)
          actions_v = torch.tensor(actions).to(device)
          rewards_v = torch.tensor(rewards).to(device)
          done_mask = torch.ByteTensor(dones).to(device)

          state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
          with torch.no_grad():
            next_state_values = target_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
          expected_state_action_values = next_state_values.detach() * gamma + rewards_v
          return nn.HuberLoss()(state_action_values, expected_state_action_values)


#Metric
@torch.no_grad()
def calc_values_of_states(batch, net, device="cpu"):
    states,_,_,_,_ = batch
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)