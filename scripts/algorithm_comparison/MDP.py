"""
MDP problem definition as a class
with value iteration algorithm
"""

# import
import time
import numpy as np
import gurobipy as grb
from copy import deepcopy as dcopy


class MDP:
    """MDP problem class"""

    def __init__(
        self, name, states, actions, trans_func,
        reward_func, initial_distr, discount_factor
    ):
        """
        `name`: str, name of the MDP;
        `states`: list, states;
        `actions`: list, actions;
        `trans_func`: function, the transition function,
            input: (new_state, old_state, action), output: pr;
        `reward_func`: function, the reward function,
            input: (state, action), output: number;
        `initial_distr`: list, initial distribution of states;
        `discount_factor`: numeric, discount factor, < 1.
        """
        super().__init__()
        self.name = name
        self.states = states
        self.actions = actions
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.initial_distr = initial_distr
        self.discount_factor = discount_factor

    # value iteration
    def VI(self, epsilon, sol_dir='None'):
        """
        Value iteration, MDP problems. Using in-place value updates.
        """
        # time
        run_time = time.time()
        # initialization
        threshold = epsilon
        epoch = 0
        value = {
            state: 0
            for state in self.states
        }
        # iteration
        while True:
            old_value = dcopy(value)
            # calculate new values
            for state in self.states:
                value[state] = np.max([
                    self.reward_func(state, action) + np.sum([
                        self.discount_factor * self.trans_func(
                            n_state, state, action
                        ) * value[n_state]
                        for n_state in self.states
                    ])
                    for action in self.actions
                ])
            # value difference between two iterations
            difference = np.max([
                np.absolute(value[state] - old_value[state])
                for state in self.states
            ])
            # check optimality condition
            if difference < threshold:
                break
            else:
                # entering next period
                epoch = epoch + 1
                continue
        # total value
        total_value = np.dot(self.initial_distr, list(value.values()))
        # Finding the best policy
        policy = {}
        for state in self.states:
            policy[state] = self.actions[np.argmax([
                self.reward_func(state, action) + np.sum([
                    self.discount_factor * self.trans_func(
                        n_state, state, action
                    ) * value[n_state]
                    for n_state in self.states
                ])
                for action in self.actions
            ])]
        # time
        run_time = time.time() - run_time
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}_VI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write("Optimality reached at epoch {};\n".format(epoch))
            file.write("Total running time: {} seconds;\n".format(run_time))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(total_value))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.close()
        return policy, total_value

    # linear programming, dual
    def LP_dual(self, sol_dir='None'):
        """
        Solving using linear programming, dual formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_x = {}
        for s in state_dict.keys():
            for a in action_dict.keys():
                var_x[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="x_{}_{}".format(s, a)
                )
        model.update()
        # objective
        objective = grb.quicksum([
            self.reward_func(self.states[s], self.actions[a]) * var_x[s, a]
            for s in state_dict.keys()
            for a in action_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        var_x[s, a]
                        for a in action_dict.keys()
                    ]),
                    -1 * grb.quicksum([
                        self.discount_factor * self.trans_func(
                            self.states[s], self.states[s_old], self.actions[a]
                        ) * var_x[s_old, a]
                        for s_old in state_dict.keys()
                        for a in action_dict.keys()
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=self.initial_distr[s],
                name="constr_{}".format(s)
            )
            model.update()
        # ------------------------ Solving --------------------------
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                var_x[s, a].X for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = model.getConstrByName(
                "constr_{}".format(s)
            ).Pi
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LPD.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy

    # linear programming, primal
    def LP(self, sol_dir='None'):
        """
        Solving using linear programming, primal formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_v = {}
        for s in state_dict.keys():
            var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            self.initial_distr[s] * var_v[s]
            for s in state_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            for a in action_dict.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        var_v[s],
                        -1 * grb.quicksum([
                            self.discount_factor * self.trans_func(
                                self.states[s_new], self.states[s],
                                self.actions[a]
                            ) * var_v[s_new]
                            for s_new in state_dict.keys()
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.reward_func(self.states[s], self.actions[a]),
                    name="constr_{}_{}".format(s, a)
                )
        model.update()
        # ------------------------ Solving --------------------------
        # model.write("model/{}-LP.lp".format(self.name))
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                model.getConstrByName(
                    "constr_{}_{}".format(s, a)
                ).Pi
                for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = var_v[s].X
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LP.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy


class MDP_finite:
    """
    MDP problem class, finite horizon
    actions are dependent on states
    """

    def __init__(
        self, name, horizon, states, actions, trans_func,
        reward_func, initial_distr, discount_factor,
        **kwargs
    ):
        """
        `name`: str, name of the MDP;
        `horizon': int, start from 0;
        `states`: list, states;
        `actions`: list, actions;
        `trans_func`: function, the transition function,
            input: (new_state, old_state, action), output: pr;
        `reward_func`: function, the reward function,
            input: (state, action), output: number;
        `initial_distr`: list, initial distribution of states;
        `discount_factor`: numeric, discount factor, < 1.
        """
        super().__init__()
        self.name = name
        self.horizon = horizon
        self.states = states
        self.actions = actions
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.initial_distr = initial_distr
        self.discount_factor = discount_factor
        self.kwargs = kwargs

    # value iteration, depreciated
    def VI(self, epsilon, sol_dir='None'):
        """
        Value iteration, MDP problems. Using in-place value updates.
        """
        # time
        run_time = time.time()
        # initialization
        threshold = epsilon
        epoch = 0
        value = {
            state: 0
            for state in self.states
        }
        # iteration
        while True:
            old_value = dcopy(value)
            # calculate new values
            for state in self.states:
                value[state] = np.max([
                    self.reward_func(state, action) + np.sum([
                        self.discount_factor * self.trans_func(
                            n_state, state, action
                        ) * value[n_state]
                        for n_state in self.states
                    ])
                    for action in self.actions
                ])
            # value difference between two iterations
            difference = np.max([
                np.absolute(value[state] - old_value[state])
                for state in self.states
            ])
            # check optimality condition
            if difference < threshold:
                break
            else:
                # entering next period
                epoch = epoch + 1
                continue
        # total value
        total_value = np.dot(self.initial_distr, list(value.values()))
        # Finding the best policy
        policy = {}
        for state in self.states:
            policy[state] = self.actions[np.argmax([
                self.reward_func(state, action) + np.sum([
                    self.discount_factor * self.trans_func(
                        n_state, state, action
                    ) * value[n_state]
                    for n_state in self.states
                ])
                for action in self.actions
            ])]
        # time
        run_time = time.time() - run_time
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}_VI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write("Optimality reached at epoch {};\n".format(epoch))
            file.write("Total running time: {} seconds;\n".format(run_time))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(total_value))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.close()
        return policy, total_value

    # Backward induction
    def BI(self, sol_dir='None'):
        """
        Backward induction.
        Actions dependent on states
        """
        # initialize
        values = {
            (t, s): 0
            for s in self.states
            for t in range(self.horizon)
        }
        policy = {
            (t, s): float("nan")
            for s in self.states
            for t in range(self.horizon)
        }
        # time
        run_time = time.time()
        # ---------- last epoch ----------
        t = self.horizon
        for s in self.states:
            # find policy
            policy[(t, s)] = self.actions[s][np.argmax([
                self.reward_func(t, s, a) for a in self.actions[s]
            ])]
            # calculate value
            values[(t, s)] = self.reward_func(t, s, policy[(t, s)])
        # new epoch
        t = t - 1
        # ---------- moving forward ----------
        while t >= 0:
            if time.time() - run_time > 18 * 3600:
                break
            for s in self.states:
                # find policy
                policy[(t, s)] = self.actions[s][np.argmax([
                    np.sum([
                        self.reward_func(t, s, a),
                        np.sum([
                            self.trans_func(
                                s_new, s, a
                            ) * values[(t + 1, s_new)]
                            for s_new in self.states
                        ])
                    ])
                    for a in self.actions[s]
                ])]
                # calculate value
                values[(t, s)] = np.sum([
                    self.reward_func(t, s, policy[(t, s)]),
                    np.sum([
                        self.trans_func(
                            s_new, s, policy[(t, s)]
                        ) * values[(t + 1, s_new)]
                        for s_new in self.states
                    ])
                ])
            # new epoch
            t = t - 1
        # time
        run_time = time.time() - run_time
        # total value
        total_value = np.dot(
            self.initial_distr,
            # [values[(0, s)] for s in self.states]
            [values[(t+1, s)] for s in self.states]
        )
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}_BI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write("Total value: {};\n".format(total_value))
            file.write("Total running time: {} seconds;\n".format(run_time))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for t in range(self.horizon):
                for s in self.states:
                    file.write("{}, {}: {}\n".format(t, s, policy[(t, s)]))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(total_value))
            for t in range(self.horizon):
                for s in self.states:
                    file.write("{}, {}: {}\n".format(t, s, values[(t, s)]))
            file.write("==============================\n")
            file.close()
        return policy, values, total_value

    # modified BI, for the PV-battery problem
    def modified_BI(
        self, state_list, action_list,
        ell_list, h_list, sunlight_hours,
        sol_dir='None'
    ):
        """
        modified BI, for the PV-battery problem
        """
        # ========== Preparations ==========
        # action index that does not include Replace, and Replace
        keep_actions = {}
        repl_action = {}
        for s in self.states:
            keep_actions[s] = []
            for a in self.actions[s]:
                if action_list[a] != "Replace":
                    keep_actions[s].append(a)
                else:
                    repl_action[s] = a
        # day and night states
        day_states, night_states = [], []
        for s in self.states:
            if state_list[s][0] == 0:
                night_states.append(s)
            else:
                day_states.append(s)
        # state dictionary, correspond state to state index
        state_dict = {}
        for s in range(len(state_list)):
            state_dict[state_list[s]] = s
        # ========== Start Algorithm ==========
        # initialize
        values = {}
        for s in self.states:
            for t in range(self.horizon + 1):
                values[t, s] = 0
        policy = {
            (t, s): float("nan")
            for s in self.states
            for t in range(self.horizon)
        }
        repl_threshold = {}
        # time
        run_time = time.time()
        # ---------- last epoch ----------
        t = self.horizon
        for s in self.states:
            # find policy
            policy[(t, s)] = "Terminate"
            # calculate value
            values[t, s] = self.reward_func(t, s, policy[(t, s)])
        # new epoch
        t = t - 1
        # ---------- moving forward ----------
        while t >= 0:
            if time.time() - run_time > 18 * 3600:
                break
            # ---------- night ----------
            for s in self.states:
                if state_list[s][0] == 1:
                    values[t, s] = 0
                    policy[(t, s)] = 'None'
                    continue
                # find policy
                policy[(t, s)] = self.actions[s][np.argmax([
                    np.sum([
                        self.reward_func(t, s, a),
                        np.sum([
                            self.trans_func(
                                s_new, s, a
                            ) * values[t + 1, s_new]
                            for s_new in day_states
                        ])
                    ])
                    for a in self.actions[s]
                ])]
                # calculate value
                values[t, s] = np.sum([
                    self.reward_func(t, s, policy[(t, s)]),
                    np.sum([
                        self.trans_func(
                            s_new, s, policy[(t, s)]
                        ) * values[t + 1, s_new]
                        for s_new in day_states
                    ])
                ])
                # dummy variable, at day-time, values are 0.
                values[t - 1, s] = 0
                policy[(t - 1, s)] = 'None'
            # next horizon
            t = t - 1
            # ---------- day ----------
            for h in h_list:
                for x in sunlight_hours:
                    ell_opt = 'None'
                    # loop, ell starts from max
                    for ell in sorted(ell_list, reverse=True):
                        # find state index
                        s = state_dict[(1, ell, h, x)]
                        # if already found
                        if ell_opt != 'None':
                            # find policy
                            policy[(t, s)] = "Replace"
                            values[t, s] = self.reward_func(
                                t, s, repl_action[s]
                            ) + np.sum([
                                self.trans_func(
                                    s_new, s, repl_action[s]
                                ) * values[t + 1, s_new]
                                for s_new in night_states
                            ])
                            continue
                        # replace value
                        repl_val = self.reward_func(
                            t, s, repl_action[s]
                        ) + np.sum([
                            self.trans_func(
                                s_new, s, repl_action[s]
                            ) * values[t + 1, s_new]
                            for s_new in night_states
                        ])
                        # if not the threshold
                        if len(keep_actions[s]) != 0:
                            # best keep action
                            best_ind = np.argmax([
                                self.reward_func(t, s, a) + np.sum([
                                    self.trans_func(
                                        s_new, s, a
                                    ) * values[t + 1, s_new]
                                    for s_new in night_states
                                ])
                                for a in keep_actions[s]
                            ])
                            keep_act = keep_actions[s][best_ind]
                            # keep value
                            keep_val = self.reward_func(
                                t, s, keep_act
                            ) + np.sum([
                                self.trans_func(
                                    s_new, s, keep_act
                                ) * values[t + 1, s_new]
                                for s_new in night_states
                            ])
                        else:
                            keep_act = action_list.index("Replace")
                            keep_val = repl_val - 10000
                        # compare, keep is better
                        if repl_val < keep_val:
                            # find policy
                            policy[(t, s)] = keep_act
                            # record value
                            values[t, s] = keep_val
                            continue
                        else:
                            # find policy
                            policy[(t, s)] = action_list.index("Replace")
                            # register optimal ell
                            ell_opt = ell
                            repl_threshold[t, h, x] = ell_opt
                            # record value
                            values[t, s] = repl_val
                            continue
            # next horizon
            t = t - 1
        # time
        run_time = time.time() - run_time
        # total value
        total_value = np.dot(
            self.initial_distr,
            # [values[0, s] for s in self.states]
            [values[t+1, s] for s in self.states]
        )
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}_mBI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write("Total value: {};\n".format(total_value))
            file.write("Total running time: {} seconds;\n".format(run_time))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for t in range(self.horizon):
                for s in self.states:
                    file.write("{}, {}: {}\n".format(t, s, policy[(t, s)]))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(total_value))
            for t in range(self.horizon):
                for s in self.states:
                    file.write("{}, {}: {}\n".format(t, s, values[(t, s)]))
            file.write("==============================\n")
            file.close()
        # modify values
        V_ADQN = {}
        for key in values.keys():
            t, s = key[0], key[1]
            if values[key] != 0:
                V_ADQN[t, state_list[s]] = values[t, s]
        return policy, values, total_value, V_ADQN
