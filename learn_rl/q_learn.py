#-*- coding: utf-8 -*-

import learn_rl.find_goal as fg
import numpy as np
import random


class QFunction(object):
    def __init__(self, env):
        self.learn_rate = 0.5
        self.q_tbl = {}
        self.allow_action = env.get_allow_actions()

    def _key(self, state):
        return str(state[0]) + ',' + str(state[1])

    def get_value(self, state, action):
        key = self._key(state)
        if not self.q_tbl.has_key(key):
            return 0

    def get_max_value(self, state):
        key = self._key(state)
        if not self.q_tbl.has_key(key):
            return 0
        return max(self.q_tbl[key])

    def get_max_action(self, state):
        key = self._key(state)
        if not self.q_tbl.has_key(key):
            return random.choice(self.allow_action)

        val_list = [(val, i) for i, val in enumerate(self.q_tbl[key])]
        val_list = sorted(val_list, key=lambda v:v[0], reverse=True)
        equal_end = 1
        while equal_end < len(val_list):
            if val_list[equal_end][0] != val_list[equal_end - 1][0]:
                break
            equal_end += 1
        action = random.choice(val_list[:equal_end])[1]
        return action

    def update_value(self, state, action, value):
        # q(s,a) = q(s,a) + alpha(r + gamma * max_a'[q(s',a')] - q(s, a))
        key = self._key(state)
        if not self.q_tbl.has_key(key):
            self.q_tbl[key] = [0] * len(self.allow_action)

        old_q = self.q_tbl[key][action]
        self.q_tbl[key][action] += self.learn_rate * (value - old_q)

    def update(self, trajectory, gamma=0.8):
        q_tbl_before = self.q_tbl.copy()
        len_traj = len(trajectory)

        updated = False
        j = 0
        while j < len_traj - 1:
            state, action, reward = trajectory[j]

            next_state = trajectory[j + 1][0]
            next_key = self._key(next_state)
            if q_tbl_before.has_key(next_key):
                max_q = max(q_tbl_before[next_key])
            else:
                max_q = 0

            new_q = reward + gamma * max_q

            state_key = self._key(state)
            if q_tbl_before.has_key(state_key) and new_q != q_tbl_before[state_key][action]:
                updated = True

            self.update_value(state, action, new_q)

            j += 1

        return updated


class Agent(object):
    def __init__(self, env, q_function):
        self.state = 0
        self.trajectory = []
        self.env = env
        self.q_function = q_function
        self.ended = False

    def begin(self):
        init_state = self.env.state()
        self.state = init_state
        self.trajectory = []
        self.ended = False

    def do_epsilon_greedy(self):
        e = 0.1  # random
        pivot = np.random.uniform()
        if pivot < e:
            action_list = self.env.get_allow_actions()
            action = random.choice(action_list)
        else:
            action = self.q_function.get_max_action(self.state)

        #역방향 금지
        while len(self.trajectory) > 1:
            if self.trajectory[len(self.trajectory) - 1][1] == 0 and action == 1:                                
                self.do_epsilon_greedy()
            elif self.trajectory[len(self.trajectory) - 1][1] == 1 and action == 0:
               self.do_epsilon_greedy()
            elif self.trajectory[len(self.trajectory) - 1][1] == 2 and action == 3:
               self.do_epsilon_greedy()
            elif self.trajectory[len(self.trajectory) - 1][1] == 3 and action == 2:
               self.do_epsilon_greedy()
            break
        ###

        self.state, reward, self.ended = self.env.get_reward(action)
        self.trajectory.append((self.state, action, reward))

    def finalize_trajectory(self):
        self.trajectory.append((self.state, -1, -1))

    def is_goal(self):
        return self.ended

    def reward(self):
        return sum([trj[2] for trj in self.trajectory])

    def make_experience(self):
        self.begin()
        while not self.is_goal():
            self.do_epsilon_greedy()
        self.finalize_trajectory()


def q_learning():
    env = fg.FindGoal()
    q_function = QFunction(env)
    agent = Agent(env, q_function)

    no_update_count = 0
    max_episode = 50000
    i = 0
    while i < max_episode:
        agent.make_experience()

        updated = q_function.update(agent.trajectory)

        if updated == False:
            no_update_count += 1
        else:
            print('episode %d: %s' % (i, agent.trajectory))
            print(q_function.q_tbl)

        i += 1


if __name__=='__main__':
    q_learning()