"""
Deep Reinforcement learning.
"""

# import
import time
# import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy


class Agent:
    """
    Agent use in the DRL class.
    - name: str, name of the agent;
    - states: a list of all states;
    - actions: a list of all actions;
    -
    Keyword Arguments:
    -
    """
    def __init__(
        self, name, states, actions, horizon, **kwargs
    ):
        super().__init__()
        # name, states & actions
        self.name = name
        self.states, self.actions = states, actions
        self.action_dict = {
            self.actions[i]: i
            for i in range(len(self.actions))
        }
        self.state_dict = {
            self.states[i]: i
            for i in range(len(self.states))
        }
        self.horizon = horizon
        # action_filter
        if 'action_filter' in kwargs:
            self.filter_action = True
            self.action_filter = kwargs['action_filter']
        else:
            self.filter_action = False
        # initialize policy (random)
        self.policy = {}
        self.old_policy = {}
        self.__initialize_policy()
        # parameters
        self.best_obj = -1e10
        self.run_time = 0

    def __initialize_policy(self):
        """
        initialize policy
        """
        # time dependent
        for t in range(self.horizon + 1):
            # randomly pick ell
            ell = np.random.random()
            # for all states
            for s in self.states:
                if s[1] < ell:
                    self.policy[t, s] = "Replace"
                else:
                    # valid_actions
                    if self.filter_action:
                        valid_actions = self.action_filter(s)
                    else:
                        valid_actions = self.actions
                    a = "Replace"
                    while a == "Replace":
                        a = np.random.choice(
                            valid_actions, 1, False
                        )[0]
                    self.policy[t, s] = a
                    # try:
                    #     valid_actions.remove("Replace")
                    # except ValueError:
                    #     pass
                    # self.policy[t, s] = np.max(valid_actions)
        return

    def take_action(self, t, state):
        """
        take action (make prediction), based on the input state.
        """
        return self.policy[t, state]

    def simulate_action(self, t, state):
        """
        take action (for simulation), based on the input state.
        """
        return self.policy[t, state]

    def learn_LS(self):
        """
        improving policy using LS
        """
        # record
        self.old_policy = dcopy(self.policy)
        # time dependent
        for t in range(self.horizon + 1):
            # randomly pick ell
            ell = np.random.random()
            # for all states
            for s in self.states:
                if s[1] < ell:
                    # replace
                    self.policy[t, s] = "Replace"
                else:
                    # valid_actions
                    if self.filter_action:
                        valid_actions = self.action_filter(s)
                    else:
                        valid_actions = self.actions
                    # currently replace or not
                    if self.policy[t, s] == "Replace":
                        try:
                            valid_actions.remove("Replace")
                        except ValueError:
                            pass
                        # randomly pick action
                        # self.policy[t, s] = np.random.choice(
                        #     valid_actions, 1, False
                        # )[0]
                        self.policy[t, s] = np.max(valid_actions)
                    else:
                        # small probability to change
                        random = np.random.random()
                        if random <= 0.05:
                            try:
                                valid_actions.remove("Replace")
                            except ValueError:
                                pass
                            # randomly pick action
                            self.policy[t, s] = np.random.choice(
                                valid_actions, 1, False
                            )[0]
        return


class Heuristic_Env:
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
        max_epoch, **kwargs
    ):
        # input parameters
        self.name = name
        self.initial_state = initial_state
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.max_epoch = max_epoch
        # agents
        self.agents = {}
        for key in kwargs:
            # named after the name of the agent
            self.agents[kwargs[key].name] = kwargs[key]

    def __plot_loss(self, key):
        """
        plot train loss for agent
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.plot(
            list(range(len(self.agents[key].loss_memory)))[1:],
            self.agents[key].loss_memory[1:]
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig('figs/loss/{}_{}.png'.format(self.name, key))
        return

    def __episode(self, write_log):
        """
        one episode
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
                envs[key] = self.agents[key].environment
            except AttributeError:
                continue
        # ======================= LOOP ========================
        epoch, learn_ind = 0, 0
        while state != 'Delta':
            # --------------- take action ---------------
            action = {}
            for key in self.agents.keys():
                action[key] = self.agents[key].take_action(
                    epoch, state
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
            # ---------- on step, transit to new state -------------
            state = new_state
            epoch += 1
            learn_ind += 1
        # ---------------- Learning --------------------
        for key in self.agents.keys():
            # keep current or discard
            if G[key] <= self.agents[key].best_obj:
                self.agents[key].policy = dcopy(
                    self.agents[key].old_policy
                )
            else:
                self.agents[key].best_obj = G[key]
                self.agents[key].old_policy = dcopy(
                    self.agents[key].policy
                )
            self.agents[key].learn_LS()
        return G, G_total

    def heuristic(
        self, episodes, write_log
    ):
        """
        Heuristic
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
            if iter % 100 == 0:
                print("Iteration {}".format(iter))
            if write_log:
                logging.info("Iteration {}".format(iter))
            # run_time = time.time()
            step_G, step_G_total = self.__episode(write_log)
            # print(time.time() - run_time)
            # step control
            for key in self.agents.keys():
                # record return
                G[key].append(step_G[key])
                # log return
                if write_log:
                    logging.info("    return for {}: {}".format(
                        key, step_G[key]
                    ))
            if write_log:
                logging.info("    -----------------------")
            # total reward
            G_total.append(step_G_total)
        # run time
        self.run_time = time.time() - self.run_time
        # for key in self.agents.keys():
        #     try:
        #         self.__plot_loss(key)
        #     except FileNotFoundError:
        #         pass
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
                    epoch, state
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
