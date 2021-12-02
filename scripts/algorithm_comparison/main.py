#!/usr/bin/env python
# coding: utf-8

"""
DQN, ADQN and SADQN. Run main_weather-6_separated.py for V in ADQN and SADQN.
"""


# import
import time
import math
import torch
import pickle
import logging
import numpy as np
import scipy.stats as st
from MDP import MDP_finite
from DRL import Agent, MDRL_Env
from matplotlib import pyplot as plt


def define_problem(n_weather, Tmax, ell_min, subproblem=False):
    """
    define the LI-ion battery problem
    """
    # ---------- parameters ----------
    # battery capacity
    battery_cap = 13.5
    # new battery cost
    battery_cost = 620.0
    # ell max
    ell_max = 1.00  # 1.00000
    # h max
    h_max = 1.00
    # capacity after replacement
    h_0 = 1.00
    # demand
    demand = [15]
    # demand probability
    demand_pr = {
        15: 1.00
    }
    # electricity cost
    mu = 10.35
    # epsilon
    ell_eps = 0.01  # 0.00001
    ell_decimal = str(ell_eps)[::-1].find('.')
    h_eps = 0.01
    h_decimal = str(h_eps)[::-1].find('.')
    # sunlight hour
    if n_weather == 2:
        sunlight_hours = [6, 4]
    elif n_weather == 3:
        sunlight_hours = [6, 4, 2]
    elif n_weather == 6:
        sunlight_hours = [6, 4, 3, 2, 1]
    # sunlight probability
    if n_weather == 2:
        sunlight_pr = {
            6: 0.27, 4: 0.73
        }
    elif n_weather == 3:
        sunlight_pr = {
            6: 0.27, 4: 0.29, 2: 0.44
        }
    elif n_weather == 6:
        sunlight_pr = {
            6: 0.135, 5: 0.135, 4: 0.145,
            3: 0.145, 2: 0.22, 1: 0.22
        }
    # hourly output
    PV_output = demand[0] / 4
    # penalty for unreachable states/actions
    penalty = -10000
    # terminal reward
    if Tmax == 30:
        terminal = 1500
    elif Tmax == 60:
        terminal = 3000
    elif Tmax == 120:
        terminal = 6000
    elif Tmax == 365:
        terminal = 25000

    def salvage_value(ell, h):
        """
        salvage value of the battery
        """
        return 100

    def battery_degradation(ell, h, a_d, a_c):
        """
        battery degradation, output new ell
        """
        # degradation parameter
        K_Delta_1 = 140000
        K_Delta_2 = -0.501
        K_Delta_3 = -123000
        K_sigma = 1.04
        Sigma_ref = 0.5
        K_T = 0.0693
        T_ref = 25
        K_t = 0.000000000414 * 12 * 3600
        S_t = K_t
        # temperature
        tau = 25
        S_T = math.exp(K_T * (tau - T_ref) * (T_ref / tau))
        S_sigma = math.exp(K_sigma * (a_c - Sigma_ref))
        # calculate degradation
        if a_d != 0:
            S_delta = math.pow((
                K_Delta_1 * math.pow(a_d, K_Delta_2) + K_Delta_3
            ), -1)
        else:
            S_delta = 0
        F_T_D = (S_delta + S_t) * S_sigma * S_T
        return (ell) * math.exp(-(F_T_D))

    # ---------- MDP elements ----------
    name = "{}-{}-{}".format(n_weather, Tmax, ell_min)
    horizon = 2 * Tmax + 1 - 1
    # ========== states ==========
    ell_list = []
    ell = ell_min
    while ell < ell_max + ell_eps:
        ell_list.append(np.round(
            ell, decimals=ell_decimal
        ))
        ell += ell_eps
    h_list = []
    h = 0.00
    while h < h_max + h_eps:
        h_list.append(np.round(
            h, decimals=h_decimal
        ))
        h += h_eps
    # the list of all states
    state_list = []
    for phi in [0, 1]:
        for ell in ell_list:
            for h in h_list:
                if phi == 1:
                    for x in sunlight_hours:
                        state_list.append((phi, ell, h, x))
                if phi == 0:
                    for d in demand:
                        state_list.append((phi, ell, h, d))
    states = list(range(len(state_list)))
    # ========== actions ==========
    action_list = []
    action_list.append("Replace")
    for h in h_list:
        action_list.append(h)
    # action dict
    actions = {}
    for s in states:
        actions[s] = []
        # ---------- night ----------
        if state_list[s][0] == 0:
            actions[s].append(action_list.index(
                np.round(0, decimals=h_decimal)
            ))
            actions[s].append(action_list.index(
                np.round(state_list[s][2]/2, decimals=h_decimal)
            ))
            actions[s].append(action_list.index(
                np.round(state_list[s][2], decimals=h_decimal)
            ))
        # ---------- day ----------
        else:
            # replace
            actions[s].append(action_list.index("Replace"))
            # force replace
            if state_list[s][1] <= ell_min:
                continue
            # enough power
            if (1 - state_list[s][2]) * state_list[s][1]\
               * battery_cap < state_list[s][3] * PV_output:
                actions[s].append(action_list.index(
                    np.round(0, decimals=h_decimal)
                ))
                actions[s].append(action_list.index(
                    np.round((1-state_list[s][2])/2, decimals=h_decimal)
                ))
                actions[s].append(action_list.index(
                    np.round(1 - state_list[s][2], decimals=h_decimal)
                ))
            # not enough power
            elif (
                (1 - state_list[s][2]) * state_list[s][1] * battery_cap
            ) / 2 < state_list[s][3] * PV_output:
                actions[s].append(action_list.index(
                    np.round(0, decimals=h_decimal)
                ))
                actions[s].append(action_list.index(
                    np.round((1-state_list[s][2])/2, decimals=h_decimal)
                ))
                actions[s].append(action_list.index(
                    np.round((
                        state_list[s][3] * PV_output
                    ) / (
                        state_list[s][1] * battery_cap
                    ), decimals=h_decimal)
                ))
            # not half enough power
            elif (
                (1 - state_list[s][2]) * state_list[s][1] * battery_cap
            ) / 2 >= state_list[s][3] * PV_output:
                actions[s].append(action_list.index(
                    np.round(0, decimals=h_decimal)
                ))
                actions[s].append(action_list.index(
                    np.round((
                        state_list[s][3] * PV_output
                    ) / (
                        state_list[s][1] * battery_cap
                    ), decimals=h_decimal)
                ))
    # ========== transition matrix ==========
    trans_pr = {}
    # for each state
    for s in states:
        # for each action of s
        for a in actions[s]:
            # ---------- Replace ----------
            if action_list[a] == "Replace":
                # day
                if state_list[s][0] == 1:
                    # find the state with the brand new battery
                    for s_n in states:
                        if all([
                            state_list[s_n][0] == 0,
                            state_list[s_n][1] == ell_max,
                            state_list[s_n][2] == h_0,
                        ]):
                            trans_pr[s_n, s, a] = demand_pr[
                                state_list[s_n][3]
                            ]
                        else:
                            trans_pr[s_n, s, a] = 0
                # night
                else:
                    # next morning
                    for s_n in states:
                        if all([
                            state_list[s_n][0] == 1,
                            state_list[s_n][1] == state_list[s][1],
                            state_list[s_n][2] == state_list[s][2],
                        ]):
                            trans_pr[s_n, s, a] = sunlight_pr[
                                state_list[s_n][3]
                            ]
                        else:
                            trans_pr[s_n, s, a] = 0
            # ---------- Number ----------
            else:
                # day
                if state_list[s][0] == 1:
                    # charge percent
                    charge_prcent = action_list[a]
                    h_new = np.min([
                        np.round(
                            state_list[s][2] + charge_prcent,
                            decimals=h_decimal
                        ),
                        h_max
                    ])
                    # degradation
                    ell_new = np.round(np.max([
                        ell_min,
                        battery_degradation(
                            state_list[s][1], state_list[s][2],
                            0, charge_prcent
                        )
                    ]), decimals=ell_decimal)
                    # find next state
                    for s_n in states:
                        if all([
                            state_list[s_n][0] == 0,
                            state_list[s_n][1] == ell_new,
                            state_list[s_n][2] == h_new
                        ]):
                            trans_pr[s_n, s, a] = demand_pr[
                                state_list[s_n][3]
                            ]
                        else:
                            trans_pr[s_n, s, a] = 0
                # night
                else:
                    # discharge percent
                    discharge_prcent = action_list[a]
                    h_new = np.min([
                        np.round(
                            state_list[s][2] - discharge_prcent,
                            decimals=h_decimal
                        ),
                        h_max
                    ])
                    # degradation
                    ell_new = np.round(np.max([
                        battery_degradation(
                            state_list[s][1], state_list[s][2],
                            discharge_prcent, 0
                        ),
                        ell_min
                    ]), decimals=ell_decimal)
                    # find next state
                    for s_n in states:
                        if all([
                            state_list[s_n][0] == 1,
                            state_list[s_n][1] == ell_new,
                            state_list[s_n][2] == h_new
                        ]):
                            trans_pr[s_n, s, a] = sunlight_pr[
                                state_list[s_n][3]
                            ]
                        else:
                            trans_pr[s_n, s, a] = 0

    # transition function
    def trans_func(new_state, old_state, action):
        """transition function"""
        return trans_pr[int(new_state), int(old_state), action]

    # transition function
    def trans_func_dqn(old_state, action):
        """transition function"""
        a = action['agent']
        # ---------- Replace ----------
        if action_list[a] == "Replace":
            # day
            if old_state[0] == 1:
                state_candi = []
                pr_candi = []
                for new_state in state_list:
                    # find the state with the brand new battery
                    if all([
                        new_state[0] == 0,
                        new_state[1] == ell_max,
                        new_state[2] == h_0,
                    ]):
                        state_candi.append(new_state)
                        pr_candi.append(demand_pr[
                            new_state[3]
                        ])
        # ---------- Number ----------
        else:
            # day
            if old_state[0] == 1:
                # charge percent
                charge_prcent = action_list[a]
                h_new = np.min([
                    np.round(
                        old_state[2] + charge_prcent,
                        decimals=h_decimal
                    ), h_max
                ])
                # degradation
                ell_new = np.round(np.max([
                    ell_min,
                    battery_degradation(
                        old_state[1], old_state[2],
                        0, charge_prcent
                    )
                ]), decimals=ell_decimal)
                state_candi = []
                pr_candi = []
                for new_state in state_list:
                    # find next state
                    if all([
                        new_state[0] == 0,
                        new_state[1] == ell_new,
                        new_state[2] == h_new
                    ]):
                        state_candi.append(new_state)
                        pr_candi.append(demand_pr[
                            new_state[3]
                        ])
            # night
            else:
                # discharge percent
                discharge_prcent = action_list[a]
                h_new = np.min([
                    np.round(
                        old_state[2] - discharge_prcent,
                        decimals=h_decimal
                    ),
                    h_max
                ])
                # degradation
                ell_new = np.round(np.max([
                    battery_degradation(
                        old_state[1], old_state[2],
                        discharge_prcent, 0
                    ),
                    ell_min
                ]), decimals=ell_decimal)
                state_candi = []
                pr_candi = []
                for new_state in state_list:
                    # find next state
                    if all([
                        new_state[0] == 1,
                        new_state[1] == ell_new,
                        new_state[2] == h_new
                    ]):
                        state_candi.append(new_state)
                        pr_candi.append(sunlight_pr[
                            new_state[3]
                        ])
        candi_ind = np.random.choice(
            range(len(state_candi)), 1, False, pr_candi
        )[0]
        return state_candi[candi_ind]

    # ========== reward matrix ==========
    reward_pr = {}
    for s in states:
        for a in actions[s]:
            # ---------- Replace ----------
            if action_list[a] == "Replace":
                # day
                if state_list[s][0] == 1:
                    reward_pr[s, a] = -battery_cost + salvage_value(
                        state_list[s][1],  state_list[s][2]
                    )
                # night
                else:
                    reward_pr[s, a] = penalty
            # ---------- Number ----------
            else:
                # lower than ell_min
                if state_list[s][1] <= ell_min:
                    reward_pr[s, a] = penalty
                    continue
                # day
                if state_list[s][0] == 1:
                    reward_pr[s, a] = 0.0
                # night
                if state_list[s][0] == 0:
                    # calculate demand cost
                    discharge_amount = action_list[a] * state_list[s][1]\
                        * battery_cap
                    reward_pr[s, a] = -mu * (
                        state_list[s][3] - discharge_amount
                    )

    # reward function
    def reward_func(t, state, action):
        """reward function"""
        if t != horizon:
            reward = reward_pr[int(state), int(action)]
        else:
            reward = salvage_value(
                ell=state_list[state][1], h=state_list[state][2]
            )
        return reward

    # initial distribution
    initial_distr = [0] * len(states)
    for key in sunlight_pr.keys():
        ind = state_list.index((1, ell_max, h_0, key))
        initial_distr[ind] = sunlight_pr[key]
    initial_distr = np.array(initial_distr)
    # discound factor
    discount_factor = 0.99997
    # define the problem
    problem = MDP_finite(
        name=name, horizon=horizon,
        states=states, actions=actions,
        trans_func=trans_func, reward_func=reward_func,
        initial_distr=initial_distr, discount_factor=discount_factor
    )
    # ========== modify for DQN ==========
    # state dictionary, correspond state to state index
    state_dict = {}
    for s in range(len(state_list)):
        state_dict[state_list[s]] = s

    # reward function
    def reward_func_dqn(t, state, action):
        """reward function"""
        if t != horizon:
            reward = reward_pr[
                state_dict[state], action['agent']
            ]
        else:
            # terminal reward
            reward = terminal + salvage_value(
                ell=state[1], h=state[2]
            )
        return {'agent': reward}

    # action filter
    def action_filter(state):
        """
        filter action, returnan action (index)
        """
        return actions[state_dict[state]]

    # get initial state
    def get_initial_state():
        """
        get the initial state
        """
        init_sunlight = np.random.choice(
            sunlight_hours, size=1, replace=False,
            p=list(sunlight_pr.values())
        )[0]
        return (1, ell_max, h_0, init_sunlight)

    if subproblem:
        return problem, {
            'state_list': state_list, 'action_list': action_list,
            'ell_max': ell_max, 'ell_min': ell_min, 'ell_list': ell_list,
            'ell_eps': ell_eps, 'h_list': h_list,
            'sunlight_hours': sunlight_hours,
            'name': name, 'horizon': horizon,
            'discount_factor': discount_factor,
            'terminal':terminal,
            'action_filter': action_filter,
            'trans_func_dqn': trans_func,
            'reward_func_dqn': reward_func_dqn,
            'initial_state': get_initial_state
        }
    else:
        return problem, {
            'state_list': state_list, 'action_list': action_list,
            'ell_max': ell_max, 'ell_min': ell_min, 'ell_list': ell_list,
            'ell_eps': ell_eps, 'h_list': h_list,
            'sunlight_hours': sunlight_hours,
            'name': name, 'horizon': horizon,
            'discount_factor': discount_factor,
            'terminal':terminal,
            'action_filter': action_filter,
            'trans_func_dqn': trans_func_dqn,
            'reward_func_dqn': reward_func_dqn,
            'initial_state': get_initial_state
        }


def plot_G(name, G, window=1, start_ind=0, sample=1):
    """
    plot return using time window
    """
    G_plot = {}
    ind = window
    while ind <= len(G):
        # sample
        if ind % sample == 0:
            G_plot[ind - 1] = np.mean([
                G[i] for i in range(ind - window, ind, 1)
            ])
        ind += 1
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(
        list(G_plot.keys())[start_ind:],
        list(G_plot.values())[start_ind:],
        'b-'
    )
    fig.savefig('figs/G/{}.png'.format(name), dpi=600)
    plt.close()
    return


def run_DQN(problem, elements):
    """
    run DQN algorithm
    """
    # agent
    agent = Agent(
        name="agent",
        actions=list(range(len(elements['action_list']))),
        # In this model, epoch is added as another dimension
        input_size=len(elements['state_list'][0]) + 1,
        hidden_layers=[64, 128, 64],
        output_size=len(elements['action_list']),
        learning_rate=1e-1,
        b_policy='e-greedy',
        learn_epoch=1,
        action_filter=elements['action_filter']
    )
    # env
    problem_dqn = MDRL_Env(
        name=problem.name + "_DQN",
        initial_state=elements['initial_state'],
        trans_func=elements['trans_func_dqn'],
        reward_func=elements['reward_func_dqn'],
        max_epoch=problem.horizon,
        memory_size=50000,
        sample_episodes=100,
        agent=agent
    )
    # DQN
    G, G_total = problem_dqn.deep_Q_Network(
        episodes=10000,  # 40000
        alpha=1.0,
        discount_factor=problem.discount_factor,
        learn_step=10,
        batch_size=100,
        eps_init=1.0,
        eps_end=0.05,
        write_log=False
    )
    G_total = np.array(G_total) - elements['terminal']
    run_time = problem_dqn.run_time
    # save parameter
    torch.save(
        agent.Q.state_dict(),
        'policy/{}_Q.pt'.format(problem.name)
    )
    # plot G
    plot_G(
        problem.name + "_DQN", G_total,
    )
    # simulation
    G_sim = []
    n_expr = 1000
    # run
    for i in range(n_expr):
        G = problem_dqn.simulate(write_to_file=False)
        G_sim.append(G['agent'])
    G_sim = np.array(G_sim) - elements['terminal']
    # write results
    G_mean = np.mean(G_sim)
    pickle.dump(G_sim, open(
        'results/{}_DQN.pickle'.format(problem.name), 'wb'
    ))
    file = open('results/{}_DQN.txt'.format(problem.name), 'w+')
    file.write('No. Expr = {}\n'.format(n_expr))
    file.write('Ave. cost = {}\n'.format(G_mean))
    file.write('95% CI = {}\n'.format(st.t.interval(
        0.95, len(G_sim) - 1, loc=G_mean, scale=st.sem(G_sim)
    )))
    file.write('Train time = {}\n'.format(run_time))
    file.write("Costs: {}\n".format(G_sim))
    file.close()
    return


def run_ADQN(n_weather, problem, elements):
    """
    run DQN algorithm
    """
    # subproblem, BI
    subproblem, sub_elements = define_problem(
        n_weather=n_weather, Tmax=30, ell_min=0.95, subproblem=True
    )
    run_time = []
    run_time.append(time.time())
    _, _, _, V_ADQN = subproblem.modified_BI(
        state_list=sub_elements['state_list'],
        action_list=sub_elements['action_list'],
        ell_list=sub_elements['ell_list'],
        h_list=sub_elements['h_list'],
        sunlight_hours=sub_elements['sunlight_hours'], sol_dir="None"
    )
    run_time[0] = time.time() - run_time[0]
    # agent
    agent = Agent(
        name="agent",
        actions=list(range(len(elements['action_list']))),
        # In this model, epoch is added as another dimension
        input_size=len(elements['state_list'][0]) + 1,
        hidden_layers=[64, 128, 64],
        output_size=len(elements['action_list']),
        learning_rate=1e-5,
        b_policy='e-greedy',
        learn_epoch=1,
        action_filter=elements['action_filter'],
        dqn_method='ADQN',
        V_ADQN=V_ADQN
    )
    # env
    problem_dqn = MDRL_Env(
        name=problem.name + "_ADQN",
        initial_state=elements['initial_state'],
        trans_func=elements['trans_func_dqn'],
        reward_func=elements['reward_func_dqn'],
        max_epoch=problem.horizon,
        memory_size=5000,
        sample_episodes=100,
        agent=agent
    )
    # DQN
    G, G_total = problem_dqn.deep_Q_Network(
        episodes=8000,
        alpha=1.0,
        discount_factor=problem.discount_factor,
        learn_step=10,
        batch_size=100,
        eps_init=1.0,
        eps_end=0.05,
        write_log=False
    )
    G_total = np.array(G_total) - elements['terminal']
    run_time.append(problem_dqn.run_time)
    # save parameter
    torch.save(
        agent.Q.state_dict(),
        'policy/{}_ADQN.pt'.format(problem.name)
    )
    # plot G
    plot_G(
        problem.name + "_ADQN", G_total,
        window=50
    )
    # simulation
    G_sim = []
    n_expr = 1000
    # run
    for i in range(n_expr):
        G = problem_dqn.simulate(write_to_file=False)
        G_sim.append(G['agent'])
    G_sim = np.array(G_sim) - elements['terminal']
    # write results
    G_mean = np.mean(G_sim)
    pickle.dump(G_sim, open(
        'results/{}_ADQN.pickle'.format(problem.name), 'wb'
    ))
    file = open('results/{}_ADQN.txt'.format(problem.name), 'w+')
    file.write('No. Expr = {}\n'.format(n_expr))
    file.write('Ave. cost = {}\n'.format(G_mean))
    file.write('95% CI = {}\n'.format(st.t.interval(
        0.95, len(G_sim) - 1, loc=G_mean, scale=st.sem(G_sim)
    )))
    file.write('Train time = {}\n'.format(run_time))
    file.write("Costs: {}\n".format(G_sim))
    file.close()
    return


def run_SADQN(n_weather, Tmax):
    """
    run DQN algorithm
    """
    # list of ell_min
    ell_min_list = [0.97, 0.98, 0.99]
    # subproblem, BI
    V_SADQN = {}
    run_time = []
    run_time.append(time.time())
    for ell_min in ell_min_list:
        subproblem, sub_elements = define_problem(
            n_weather=n_weather, Tmax=30, ell_min=ell_min, subproblem=True
        )
        _, _, _, V_SADQN[ell_min] = subproblem.modified_BI(
            state_list=sub_elements['state_list'],
            action_list=sub_elements['action_list'],
            ell_list=sub_elements['ell_list'],
            h_list=sub_elements['h_list'],
            sunlight_hours=sub_elements['sunlight_hours'], sol_dir="None"
        )
        del subproblem
        del sub_elements
    run_time[0] = time.time() - run_time[0]

    # ---------- define the problem ----------
    print("Constructing problem...")
    problem, elements = define_problem(
        n_weather=n_weather, Tmax=Tmax, ell_min=0.75, subproblem=False
    )
    print("Done!")

    # agent
    agent = Agent(
        name="agent",
        actions=list(range(len(elements['action_list']))),
        # In this model, epoch is added as another dimension
        input_size=len(elements['state_list'][0]) + 1,
        hidden_layers=[64, 128, 64],
        output_size=len(elements['action_list']),
        learning_rate=1e-5,
        b_policy='e-greedy',
        learn_epoch=1,
        action_filter=elements['action_filter'],
        dqn_method='SADQN',
        V_SADQN=V_SADQN,
        ell_min_list=ell_min_list
    )
    # env
    problem_dqn = MDRL_Env(
        name=problem.name + "_SADQN",
        initial_state=elements['initial_state'],
        trans_func=elements['trans_func_dqn'],
        reward_func=elements['reward_func_dqn'],
        max_epoch=elements['horizon'],
        memory_size=100000,
        sample_episodes=100,
        agent=agent
    )
    # DQN
    G, G_total = problem_dqn.deep_Q_Network(
        episodes=8000,
        alpha=1.0,
        discount_factor=elements['discount_factor'],
        learn_step=10,
        batch_size=100,
        eps_init=1.0,
        eps_end=0.05,
        write_log=False
    )
    G_total = np.array(G_total) - elements['terminal']
    run_time.append(problem_dqn.run_time)
    # plot G
    plot_G(
        problem.name + "_SADQN", G_total,
        window=50
    )
    # simulation
    G_sim = []
    n_expr = 1000
    # run
    for i in range(n_expr):
        G = problem_dqn.simulate(write_to_file=False)
        G_sim.append(G['agent'])
    G_sim = np.array(G_sim) - elements['terminal']
    # write results
    G_mean = np.mean(G_sim)
    pickle.dump(G_sim, open(
        'results/{}_SADQN.pickle'.format(problem.name), 'wb'
    ))
    file = open('results/{}_SADQN.txt'.format(problem.name), 'w+')
    file.write('No. Expr = {}\n'.format(n_expr))
    file.write('Ave. cost = {}\n'.format(G_mean))
    file.write('95% CI = {}\n'.format(st.t.interval(
        0.95, len(G_sim) - 1, loc=G_mean, scale=st.sem(G_sim)
    )))
    file.write('Train time = {}\n'.format(run_time))
    file.write("Costs: {}\n".format(G_sim))
    file.close()
    return


def compare_algorithm():
    """
    Compare BI, modified BI, ADQN and SADQN
    """
    # logging
    logging.basicConfig(
        filename='log.log', filemode='w+',
        format='%(levelname)s - %(message)s', level=logging.INFO
    )
    n_weather = 2
    Tmax = 30
    # ---------- define the problem ----------
    print("Constructing problem...")
    problem, elements = define_problem(
        n_weather=n_weather, Tmax=Tmax, ell_min=0.75
    )
    print("Done!")
    # ---------- BI ----------
    # print("Solving with BI...")
    # policy, values, total_value = problem.BI(sol_dir="results/")
    # print("Done!")
    # ---------- modified BI ----------
    print("solving with modified BI...")
    _, _, _, _ = problem.modified_BI(
        state_list=elements['state_list'],
        action_list=elements['action_list'],
        ell_list=elements['ell_list'],
        h_list=elements['h_list'],
        sunlight_hours=elements['sunlight_hours'],
        sol_dir="results"
    )
    print("Done!")
    # ---------- DQN ----------
    print("Solving with DQN...")
    run_DQN(problem, elements)
    print("Done!")
    # ---------- ADQN ----------
    print("Solving with ADQN...")
    run_ADQN(n_weather, problem, elements)
    print("Done!")
    # ---------- SADQN ----------
    print("Solving with SADQN...")
    run_SADQN(n_weather, Tmax)
    print("Done!")


np.random.seed(1)
# compare algrothim
compare_algorithm()
