"""
Deep Reinforcement learning.
"""

# import
import time
# import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from Q_MLP import Q_MLP
import math
import matplotlib.pyplot as plt


class Agent:
    """
    Agent use in the DRL class.
    - name: str, name of the agent;
    - actions: a list of all actions;
    - input_size: int, size of the input, len(state);
    - hidden_layers: list, the number of neurons for the hidden layer,
    - output_zise: int, size of the output, len(all actions);
    - b_policy: 'randomized' or 'e-greedy',
        if 'e-greedy', please specify 'epsilon' and 'e_decay';
    Keyword Arguments:
    - action_filter: function that filters actions for a given state;
    - epsilon: epsilon for the epsilon-greedy policy;
    - e_min: minimum of epsilon;
    - environment: the environment required for the agent to make action,
        e.g., a Graph;
    - Q_path: path to learned parameters of pytorch network.
    """

    def __init__(
        self, name, actions, input_size, hidden_layers, output_size,
        learning_rate, b_policy, learn_epoch, **kwargs
    ):
        super().__init__()
        # ------------ check behavior policy completness -------------
        self.b_policy = b_policy
        if self.b_policy == 'e-greedy':
            try:
                self.epsilon = 0.0
                # self.e_min = kwargs['e_min']
            except KeyError:
                raise KeyError(
                    'Please provide starting epsilon and minimal epsilon!'
                )
        # ------------- whether to use action_filter ----------------
        if 'action_filter' in kwargs:
            self.filter_action = True
            self.action_filter = kwargs['action_filter']
        else:
            self.filter_action = False
        # -------------------- environment --------------------------
        if 'environment' in kwargs:
            self.environment = kwargs['environment']
        # name & actions
        self.name, self.actions = name, actions
        self.action_dict = {
            self.actions[i]: i
            for i in range(len(self.actions))
        }
        # ------------------- GPU ----------------
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"
        # ------------------ augmented or stochastic or vanilla ---------------
        if 'dqn_method' in kwargs:
            self.dqn_method = kwargs['dqn_method']
            if kwargs['dqn_method'] == 'ADQN':
                self.V_ADQN = kwargs['V_ADQN']
            elif kwargs['dqn_method'] == 'SADQN':
                self.V_SADQN = kwargs['V_SADQN']
                self.ell_min_list = kwargs['ell_min_list']
        else:
            self.dqn_method = 'vanilla'
        # ----------------- construct network --------------------
        self.hidden_layers = hidden_layers
        self.input_size, self.output_size = input_size, output_size
        # prediction network
        self.Q = Q_MLP(
            hidden_layer_shape=self.hidden_layers,
            input_size=input_size,
            output_size=output_size,
            seed=1
        )
        self.Q.to(self.dev)
        # target network
        self.Q_target = Q_MLP(
            hidden_layer_shape=self.hidden_layers,
            input_size=input_size,
            output_size=output_size,
            seed=1
        )
        self.Q_target.to(self.dev)
        # for p in self.Q.parameters():
        #     p.data.fill_(-10000)
        # for p in self.Q_target.parameters():
        #     p.data.fill_(-10000)
        # load previous parameters
        if 'Q_path' in kwargs:
            self.Q.load_state_dict(torch.load(kwargs['Q_path']))
            self.Q.eval()
        # optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        # self.optimizer = optim.Adadelta(self.Q.parameters())
        # training step
        self.train_step = 1
        self.state_keys = 'None'
        self.loss_memory = []
        self.learn_epoch = learn_epoch
        self.run_time = 0

    def take_action(self, state):
        """
        take action (make prediction), based on the input state.
        """
        if self.state_keys == 'None':
            pass
            # self.state_keys = list(state.keys())
        # make state to tensor
        input_seq = torch.tensor(
            # list(state.values()), dtype=torch.float, device=self.dev
            list(state), dtype=torch.float, device=self.dev
        )
        # make a prediction
        self.Q.eval()
        with torch.no_grad():
            output_seq = list(self.Q(input_seq))
        self.Q.train()
        # valid_actions
        if self.filter_action:
            valid_actions = self.action_filter(tuple(state[1:]))
        else:
            valid_actions = self.actions
        # epsilon greedy policy
        if self.b_policy == 'e-greedy':
            # get a random number between 0, 1
            if np.random.random() > self.epsilon:
                return valid_actions[np.argmax([
                    output_seq[self.action_dict[a]]
                    for a in valid_actions
                ])]
            else:
                return valid_actions[np.random.choice(
                    range(len(valid_actions)), size=1, replace=False,
                    p=[1 / len(valid_actions)] * len(valid_actions)
                )[0]]
        # randomized policy
        elif self.b_policy == 'random':
            return valid_actions[np.random.choice(
                range(len(valid_actions)), size=1, replace=False,
                p=[1 / len(valid_actions)] * len(valid_actions)
            )[0]]

    def simulate_action(self, state):
        """
        take action (for simulation), based on the input state.
        """
        # make state to tensor
        input_seq = torch.tensor(
            # list(state.values()), dtype=torch.float, device=self.dev
            list(state), dtype=torch.float, device=self.dev
        )
        # make a prediction
        self.Q.eval()
        with torch.no_grad():
            output_seq = list(self.Q(input_seq))
        self.Q.train()
        # valid_actions
        if self.filter_action:
            valid_actions = self.action_filter(tuple(state[1:]))
        else:
            valid_actions = self.actions
        return valid_actions[np.argmax([
            output_seq[self.action_dict[i]]
            for i in valid_actions
        ])]

    def learn(self, memory, alpha, discount_factor):
        """
        train the network
        """
        # memories
        state_memory = memory[0]
        new_state_memory = memory[1]
        delta_memory = memory[2]
        action_memory = memory[3]
        reward_memory = memory[4]
        # action index
        action_ind = torch.tensor([
            [self.action_dict[a]] for a in action_memory
        ], device=self.dev)  # .flatten()
        # while True:
        for train_iter in range(self.learn_epoch):
            # set zero grad
            self.optimizer.zero_grad()
            # make a prediction
            Q_pred = self.Q(
                torch.FloatTensor(state_memory).to(self.dev)
            ).gather(1, action_ind).flatten()
            # calculate the target
            Q_targ_future = self.Q_target(
                torch.FloatTensor(new_state_memory).to(self.dev)
            ).detach().max(1)[0]
            # target
            if self.dqn_method == "vanilla":
                Q_targ = torch.FloatTensor([
                    reward_memory[i] if delta_memory[i]
                    else reward_memory[i] + discount_factor * Q_targ_future[i]
                    for i in range(len(reward_memory))
                ])
            # ADQN
            elif self.dqn_method == "ADQN":
                Q_targ = []
                for i in range(len(reward_memory)):
                    t = new_state_memory[i][0]
                    s = tuple(new_state_memory[i][1:])
                    try:
                        Q_targ.append(self.V_ADQN[t, s])
                    except KeyError:
                        Q_targ.append(
                            reward_memory[i] if delta_memory[i] else
                            reward_memory[i] + discount_factor
                            * Q_targ_future[i]
                        )
                Q_targ = torch.FloatTensor(Q_targ)
            # SADQN
            elif self.dqn_method == "SADQN":
                Q_targ = []
                for i in range(len(reward_memory)):
                    t = new_state_memory[i][0]
                    s = tuple(new_state_memory[i][1:])
                    try:
                        # sample ell
                        ell_min = np.random.choice(
                            self.ell_min_list, 1, False,
                            [1 / len(self.ell_min_list)]*len(self.ell_min_list)
                        )[0]
                        Q_targ.append(self.V_SADQN[ell_min][t, s])
                    except KeyError:
                        Q_targ.append(
                            reward_memory[i] if delta_memory[i] else
                            reward_memory[i] + discount_factor
                            * Q_targ_future[i]
                        )
                Q_targ = torch.FloatTensor(Q_targ)
            Q_pred = Q_pred.to(self.dev)
            Q_targ = Q_targ.to(self.dev)
            # loss
            loss = F.mse_loss(Q_pred, Q_targ)
            # backpropogate
            loss.backward()
            self.optimizer.step()
        self.loss_memory.append(loss.to('cpu').detach().numpy())
        self.train_step += 1
        # soft update the target network
        if self.train_step % 1 == 0:
            self.__soft_update(self.Q, self.Q_target, 0.001)
        return

    def __soft_update(self, Q, Q_target, tau):
        """
        Soft update model parameters:
            θ_target = τ*θ_trained + (1 - τ)*θ_target;
        Q: weights will be copied from;
        Q_target: weights will be copied to;
        tau: interpolation parameter.
        """
        for q_target, q in zip(Q_target.parameters(), Q.parameters()):
            q_target.data.copy_(
                tau * q.data + (1.0 - tau) * q_target.data
            )
        return


class MDRL_Env:
    """
    Multi-agents Deep Reinforcement learning class.
    - initial state: dict, or functions that returns a dict,
        denoting the initial state;
        Terminal state = 'Delta'!!!!!!!!!!!!!!!!!;
    - reward: input state and action, output a number;
        Output should be a dict, {agent_name: agent_action};
    - transition: input state and action, action should be a dict,
        {agent_name: agent_action};
        If additional environment is needed, input should be
        (state, action, envs), otherwise, (state, action);
        Output a new state;
    Keyword arguments:
    - every kwargs is considered to be an agent.
    """

    def __init__(
        self, name, initial_state, trans_func, reward_func,
        max_epoch, memory_size, sample_episodes, **kwargs
    ):
        # input parameters
        self.name = name
        self.initial_state = initial_state
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.learn_ind = 1
        self.max_epoch = max_epoch
        self.memory_size = memory_size
        self.sample_episodes = sample_episodes
        # agents
        self.agents = {}
        for key in kwargs:
            # named after the name of the agent
            self.agents[kwargs[key].name] = kwargs[key]
        # how many memory to keep
        self.memory = Memory(
            agent_names=list(self.agents.keys()),
            memory_size=self.memory_size,
            sample_episodes=self.sample_episodes
        )

    def __plot_loss(self, key):
        """
        plot train loss for agent
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.plot(
            list(range(len(self.agents[key].loss_memory)))[100000:],
            self.agents[key].loss_memory[100000:]
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig('figs/loss/{}_{}.png'.format(self.name, key))
        return

    def __Q_update(
        self, agents, alpha, discount_factor,
        learn_step, batch_size, write_log, iter
    ):
        """
        one episode of Q-learning
        """
        # =================== initialize ===================
        # initialize
        if callable(self.initial_state):
            state = self.initial_state()
        else:
            state = self.initial_state
        # agent environment and return initialization
        envs, G, G_total = {}, {}, 0
        for key in self.agents.keys():
            G[key] = 0
            try:
                envs[key] = agents[key].environment
            except AttributeError:
                continue
        # ======================= LOOP ========================
        epoch, learn_ind = 0, 0
        while state != 'Delta':
            # --------------- take action ---------------
            action = {}
            for key in self.agents.keys():
                action[key] = self.agents[key].take_action(
                    [epoch] + list(state)
                )
            # -------------- transit and reward --------------
            if len(envs) == 0:
                # transition
                new_state = self.trans_func(state, action)
                # rewards
                R = self.reward_func(epoch, state, action)
            else:
                new_state = self.trans_func(state, action, envs)
                # rewards
                R = self.reward_func(state, action, new_state, envs)
            delta = True if new_state == 'Delta' else False
            # ---------------- update G ------------------
            for key in self.agents.keys():
                G[key] += R[key]
                G_total += R[key]
            # ------------ exit if too many epochs --------------
            if epoch + 1 > self.max_epoch:
                new_state = 'Delta'
            # --------------- output experience ----------------
            if write_log:
                # logging state, action and reward
                logging.info('    epoch: {}'.format(epoch))
                logging.info('    state: {}'.format(state))
                logging.info('    action: {}'.format(action))
                logging.info('    reward: {}'.format(R))
            # ---------------- memory control --------------------
            if new_state == "Delta":
                new_state_memory = [epoch] + list(state)
            else:
                new_state_memory = [epoch] + list(new_state)
            self.memory.update(
                [epoch] + list(state), delta, action, R,
                new_state_memory
            )
            # ---------------- Learning --------------------
            # for each agent, provide data and train.
            if all([
                # len(self.memory.memory['state']) >= self.sample_episodes,
                learn_ind % learn_step == 0
            ]):
                for key in self.agents.keys():
                    agents[key].learn(
                        memory=self.memory.sample(key),
                        alpha=alpha, discount_factor=discount_factor
                    )
            # ---------- on step, transit to new state -------------
            state = new_state
            epoch += 1
            learn_ind += 1
        return G, G_total

    def deep_Q_Network(
        self, episodes, alpha, discount_factor, learn_step, batch_size,
        eps_init, eps_end, write_log
    ):
        """
        Deep Q Network.
        - episodes: how many iterations to simulation in total.
        - alpha: learning rate;
        - discount factor: perspective of future.
        """
        G, G_total = {}, []
        for key in self.agents.keys():
            G[key] = []
        # ---------------------- Learning ----------------------
        if write_log:
            logging.info("Learning...")
        # run time
        self.run_time = time.time()
        for iter in range(episodes):
            if iter % 10000 == 0:
                print("Iteration {}".format(iter))
            if write_log:
                logging.info("Iteration {}".format(iter))
            step_G, step_G_total = self.__Q_update(
                self.agents, alpha, discount_factor,
                learn_step, batch_size, write_log, iter
            )
            # step control
            for key in self.agents.keys():
                # record return
                G[key].append(step_G[key])
                # log return
                if write_log:
                    logging.info("    return for {}: {}".format(
                        key, step_G[key]
                    ))
                # set epsilon, 7-2.5, 10-5
                self.agents[key].epsilon = 1 / (1 * math.exp(
                    5 * iter / episodes
                ))
                # (eps_init - eps_end) * np.max([
                #     (episodes * 1 - iter) / (episodes * 1), 0
                # ]) + eps_end
                # 1 / (1 * math.exp(
                #     0.5 * iter / episodes
                # ))
            if write_log:
                logging.info("    -----------------------")
            # total reward
            G_total.append(step_G_total)
        # run time
        self.run_time = time.time() - self.run_time
        for key in self.agents.keys():
            try:
                self.__plot_loss(key)
            except FileNotFoundError:
                pass
        return G, G_total

    def simulate(self, write_to_file):
        """
        simulate the delivery process.
        """
        if write_to_file:
            output_file = open('results/{}_sim.txt'.format(self.name), 'w+')
            output_file.write("=============== SIMULATION ==============\n")
            output_file.write("Run Time = {}\n".format(self.run_time))
        # =================== initialize ===================
        # initialize
        if callable(self.initial_state):
            state = self.initial_state()
        else:
            state = self.initial_state
        # agent environment and return initialization
        envs, G = {}, {}
        for key in self.agents.keys():
            G[key] = 0
            try:
                envs[key] = self.agents[key].environment
            except AttributeError:
                continue
        # ======================= LOOP ========================
        epoch = 0
        while state != 'Delta':
            # --------------- take action ---------------
            action = {}
            for key in self.agents.keys():
                action[key] = self.agents[key].simulate_action(
                    [epoch] + list(state)
                )
            # -------------- transit and reward --------------
            if len(envs) == 0:
                # transition
                new_state = self.trans_func(state, action)
                # rewards
                R = self.reward_func(epoch, state, action)
            else:
                new_state = self.trans_func(state, action, envs)
                # rewards
                R = self.reward_func(state, action, new_state, envs)
            if write_to_file:
                # logging state, action and reward
                output_file.write('    epoch: {}\n'.format(epoch))
                output_file.write('    state: {}\n'.format(state))
                output_file.write('    action: {}\n'.format(action))
                output_file.write('    reward: {}\n'.format(R))
            # ------------ exit if too many epochs --------------
            if epoch + 1 > self.max_epoch:
                new_state = 'Delta'
            # ---------------- update G --------------------
            for key in self.agents.keys():
                G[key] = G[key] + R[key]
            # ---------- on step, transit to new state -------------
            state = new_state
            epoch += 1
        for key in self.agents.keys():
            if write_to_file:
                output_file.write('Return of {}: {}\n'.format(key, G[key]))
        return G


class Memory():
    """
    Memory: remembers past information regarding state,
        action, reward and terminal;
    `agent_names`: list, agent names
    """
    def __init__(self, agent_names, memory_size, sample_episodes):
        super().__init__()
        self.memory_max = memory_size
        self.agent_names = agent_names
        self.sample_size = sample_episodes
        self.memory = {
            'state': [], 'delta': [], 'n_state': [],
            'action': {key: [] for key in self.agent_names},
            'reward': {key: [] for key in self.agent_names},
        }
        self.pointer = 0
        self.memory_size = 0

    # update, keep the most recent
    def update(self, state, delta, action, reward, new_state):
        """
        `state`: list, new state, do not remember 'Delta';
        `action`: dict, key: agent name, value: new action;
        `delta`: bool, whether the NEXT state is delta;
        `reward`: dict, key: agent name, value: reward.
        """
        # not full
        if self.memory_size < self.memory_max:
            # self.memory['state'].append(list(state.values()))
            self.memory['state'].append(state)
            self.memory['delta'].append(delta)
            self.memory['n_state'].append(new_state)
            for name in self.agent_names:
                self.memory['action'][name].append(action[name])
                self.memory['reward'][name].append(reward[name])
            self.memory_size += 1
        # full
        else:
            self.memory['state'][self.pointer] = state
            self.memory['delta'][self.pointer] = delta
            self.memory['n_state'][self.pointer] = new_state
            for name in self.agent_names:
                self.memory['action'][name][self.pointer] = action[name]
                self.memory['reward'][name][self.pointer] = reward[name]
        # update pointer
        if self.pointer == self.memory_max - 1:
            self.pointer = 0
        else:
            self.pointer += 1
        return

    # smple from the entire memory, with large memory size
    def sample(self, name):
        """
        sample state, action, reward and delta
        """
        # indices
        if self.memory_size >= self.sample_size:
            choose_size = self.sample_size
        else:
            choose_size = self.memory_size
        sample_ind = np.random.choice(
            range(self.memory_size), size=choose_size, replace=False,
            p=[1/self.memory_size] * self.memory_size
        )
        # sample
        state_sample = [
            self.memory['state'][i] for i in sample_ind
        ]
        new_state_sample = [
            self.memory['n_state'][i] for i in sample_ind
        ]
        delta_sample = [
            self.memory['delta'][i] for i in sample_ind
        ]
        action_sample = [
            self.memory['action'][name][i] for i in sample_ind
        ]
        reward_sample = [
            self.memory['reward'][name][i] for i in sample_ind
        ]
        return (
            state_sample, new_state_sample, delta_sample,
            action_sample, reward_sample
        )
