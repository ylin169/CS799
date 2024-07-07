import numpy as np
import torch
import torch.nn as nn
import random
from Tree import SumTree
from chainer import serializers

class ReplayMemory:
    def __init__(self, n_s, n_a):

        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 1000
        self.Tree = SumTree(size=self.MEMORY_SIZE)
        self.BATCH_SIZE  =50
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, 3 , 9), dtype=np.float64)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, 3, 9), dtype=np.float64)
        #self.all_s = np.empty(shape=(self.MEMORY_SIZE, 2,6), dtype=np.float64) # for Test_Env_1
        #self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, 2,6), dtype=np.float64)# for Test_Env_1
        #self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        #self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        #print('self.all_s: ' , self.all_s)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.uint8)
        #print(' self.all_a: ',  self.all_a)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)

        self.count = 0
        self.t = 0
        self.eps = 0.01
        self.max_priority = 0.01
        self.alpha = 0.1
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001


        # self.a1 = np.random.randint(low=0,high=)


    def add_memo(self, transition):
        s, a, r, done, s_ = transition

        self.Tree.add(self.max_priority, self.t)
        self.all_s[self.t] = s
        #print('*******************',self.all_s)
        #print('self.all_s[self.t]: ', self.all_s[self.t])
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_done[self.t] = done
        self.all_s_[self.t] = s_

        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE


    def mem_size(self):
        m = self.MEMORY_SIZE
        return m

    def count_size(self):
        c = self.count
        return c

    def sample(self):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        assert self.count >= self.BATCH_SIZE,"buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(self.BATCH_SIZE, 1, dtype=torch.float)
        #print('priorities: ',priorities)
        segment = self.Tree.total / self.BATCH_SIZE
        print('segment: ', segment, 'self.BATCH_SIZE',self.BATCH_SIZE, 'self.Tree.total',self.Tree.total)

        for i in range(self.BATCH_SIZE):

            a, b = segment * i, segment* (i + 1)
            #print('iiiiii:', i, range(self.BATCH_SIZE))

            cumsum = random.uniform(a, b)
            #print('cumsum:', cumsum)
            #print('&&&&&&&&&cumsum', cumsum, 'a',a, 'b',b)
            tree_idx, priority, sample_idx = self.Tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
            #self.all_s[sample_idx]
            #print('tree_idx:', tree_idx, 'priority:', priority, 'sample_idx: ', sample_idx)
        probs = priorities / self.Tree.total
        #print('probs: ', probs)
        weights = (self.count * probs) ** -self.beta
        #print('weights', weights,'self.count' ,self.count )

        weights = weights / weights.max()
        #print('weights: ', weights, 'self.beta', self.beta, 'self.count: ', self.count, 'self.Tree.total:', self.Tree.total )

        #print('weights: ', weights)

        #batch_s = []
        #batch_a = []
        #batch_r = []
        #batch_done = []
        #batch_s_ = []

        #batch_s.append(self.all_s[sample_idxs])
        batch_s = self.all_s[sample_idxs]
        #print('gggggg: ', gg)
        #batch_a.append(self.all_a[sample_idxs])
        batch_a = self.all_a[sample_idxs]
        #batch_r.append(self.all_r[sample_idxs])
        batch_r = self.all_r[sample_idxs]
        #batch_done.append(self.all_done[sample_idxs])
        batch_done = self.all_done[sample_idxs]
        #batch_s_.append(self.all_s_[sample_idxs])
        batch_s_ = self.all_s_[sample_idxs]

        #print('############', self.all_s)
        #print('************', sample_idxs)

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)
        #print('************', self.all_s)
        batch = (
            batch_s_tensor,
            batch_a_tensor,
            batch_r_tensor,
            batch_done_tensor,
            batch_s__tensor
        )

        #return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):

            priorities = priorities.detach().cpu().numpy()
        #print('yyyyyyyyypriorities:', priorities, 'data_idxs: ', data_idxs)
        for data_idx, priority in zip(data_idxs, priorities[0]):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            #print('xxxxxxpriorities-before:', priority, 'data_idx: ', data_idx)


            #priority = (priority + self.eps) ** self.alpha

            if priority == 0:
                #priority = (priority + 0.01) ** self.alpha
                priority = 1
            else:
                priority = 1/(priority*10)
        
                #priority = 1 - priority

            #priority =1 / (priority+1)
            #priority = (1/(priority) + 0.0001)
            #priority = (priority + 0.01) ** self.alpha
            #priority = priority * 10
            #priority = priority ** 5
            #print('xxxxxxpriorities-after:', priority, 'data_idx: ', data_idx)
            self.Tree.update(data_idx, priority)
            #print('priority: ', priority)
            self.max_priority = max(self.max_priority, priority.max())


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()  # Reuse the param of nn.Module
        in_features = n_input  # ?


        # nn.Sequential() ?®®
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_output))


    def forward(self, x):
        #print('XXX: ',x)
        return self.net(x)

    def act(self, obs):
        #print('obs: ', obs, obs.shape)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        #print('obs_tensor_Shape: ', obs_tensor.shape, 'obs_tensor: ', obs_tensor)
        q_values = self(obs_tensor.unsqueeze(0))  # ?
        #print('Q_value: ', q_values, q_values.size())
        # max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
        max_q_index = torch.argmax(q_values, dim = 1, keepdim=True)[0]
        #print('###########max_q_index： ', max_q_index, max_q_index.size())
        action = max_q_index.detach()  # get the idx of q
        #print('action: ', action)
        return action



class Agent:
    def __init__(self, idx, n_input, n_output, mode="train"):
        self.idx = idx
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output
        self.PATH2 = "state_dict_model2.pth"
        self.GAMMA = 0.99
        #self.learning_rate = 0.001
        self.learning_rate = 0.00005
        # self.MIN_REPLAY_SIZE = 1000

        self.memo = ReplayMemory(n_s=self.n_input, n_a=self.n_output)

        # Initialize the replay buffer of agent i
        if self.mode == "train":
            self.online_net = DQN(self.n_input, self.n_output)
            self.target_net = DQN(self.n_input, self.n_output)
            self.target_net.load_state_dict(self.online_net.state_dict())  # copy the current state of online_net
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

