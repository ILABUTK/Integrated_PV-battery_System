#!/usr/bin/env python
# coding: utf-8

# import
import math
import pickle
import logging
import numpy as np
import scipy.stats as st
from MDP import MDP_finite
from heuristic import Agent, Heuristic_Env
from matplotlib import pyplot as plt


def define_problem(Tmax, ell_min):
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
    sunlight_hours = [6, 4]
    # sunlight probability
    sunlight_pr = {
        # 2 weather
        6: 0.27, 4: 0.73
        # 3 weather
        # 6: 0.27, 4: 0.29, 2: 0.44
        # 6 weather
        # 6: 0.135, 5: 0.135, 4: 0.145,
        # 3: 0.145, 2: 0.22, 1: 0.22
    }
    # hourly output
    PV_output = demand[0] / 4
    # penalty for unreachable states/actions
    penalty = -10000
    # terminal reward
    terminal = 0

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
    name = "PV-battery-{}-{}".format(ell_min, Tmax)
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

    # transition function
    def trans_func(new_state, old_state, action):
        """transition function"""
        return trans_pr[int(new_state), int(old_state), action]

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

    # transition function
    def trans_func_heuristic(old_state, action):
        """transition function"""
        s = state_dict[old_state]
        new_state_ind = np.random.choice(
            states, size=1, replace=False,
            p=[
                trans_pr[s_n, s, action['agent']]
                for s_n in states
            ]
        )[0]
        return state_list[new_state_ind]

    # reward function
    def reward_func_heuristic(t, state, action):
        """reward function"""
        if t != horizon:
            reward = reward_pr[
                state_dict[state], action['agent']
            ]
        else:
            # terminal reward of 5000, 8000 for 120 days
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

    return problem, {
        'state_list': state_list, 'action_list': action_list,
        'ell_max': ell_max, 'ell_min': ell_min, 'ell_list': ell_list,
        'ell_eps': ell_eps, 'h_list': h_list, 'sunlight_hours': sunlight_hours,
        'action_filter': action_filter,
        'trans_func_heuristic': trans_func_heuristic,
        'reward_func_heuristic': reward_func_heuristic,
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


def run_LS(Tmax, problem, elements):
    """
    run LS algorithm
    """
    # agent
    agent = Agent(
        name="agent",
        states=elements['state_list'],
        actions=elements['action_list'],
        horizon=problem.horizon,
        action_filter=elements['action_filter']
    )
    # env
    problem_LS = Heuristic_Env(
        name=problem.name + "_LS",
        initial_state=elements['initial_state'],
        trans_func=elements['trans_func_heuristic'],
        reward_func=elements['reward_func_heuristic'],
        max_epoch=problem.horizon,
        agent=agent
    )
    # LS
    G, G_total = problem_LS.heuristic(
        episodes=500, write_log=False
    )
    G_total = np.array(G_total) - 0
    run_time = problem_LS.run_time
    # save policy
    pickle.dump(
        agent.policy,
        open('policy/{}_Q.pt'.format(problem.name), 'wb')
    )
    # plot G
    plot_G(
        problem.name + "_LS", G_total,
    )
    # simulation
    G_sim = []
    n_expr = 1000
    # run
    for i in range(n_expr):
        G = problem_LS.simulate(write_to_file=False)
        G_sim.append(G['agent'])
    G_sim = np.array(G_sim) - 0
    # write results
    G_mean = np.mean(G_sim)
    pickle.dump(G_sim, open(
        'results/{}_LS.pickle'.format(problem.name), 'wb'
    ))
    file = open('results/{}_LS.txt'.format(problem.name), 'w+')
    file.write('No. Expr = {}\n'.format(n_expr))
    file.write('Ave. cost = {}\n'.format(G_mean))
    CI = st.t.interval(
        0.95, len(G_sim) - 1, loc=G_mean, scale=st.sem(G_sim)
    )
    file.write('95% CI = {}\n'.format(CI[1] - CI[0]))
    file.write('Train time = {}\n'.format(run_time))
    file.write("Costs: {}\n".format(G_sim))
    file.close()
    return


def compare_algorithm():
    """
    Compare BI, modified BI, ADQN and SADQN
    """
    # logging
    # logging.basicConfig(
    #     filename='test.log', filemode='w+',
    #     format='%(levelname)s - %(message)s', level=logging.INFO
    # )
    # ---------- define the problem ----------
    Tmax = 30
    print("Constructing problem...")
    problem, elements = define_problem(Tmax=Tmax, ell_min=0.75)
    print("Done!")
    # ---------- LS ----------
    print("Solving with LS...")
    run_LS(Tmax, problem, elements)
    print("Done!")
    return


def main():
    np.random.seed(1)
    # compare algrothim
    compare_algorithm()


if __name__ == "__main__":
    main()
