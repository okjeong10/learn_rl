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

        if value != old_q:
            return True
        return False

    def update(self, trajectory, gamma=0.8):
        len_traj = len(trajectory)

        updated = False
        j = len_traj - 1
        while j >= 0:
            state, action, reward, new_state = trajectory[j]
            max_q = 0
            if j != len_traj - 1:
                next_state = new_state
                next_key = self._key(next_state)
                if self.q_tbl.has_key(next_key):
                    max_q = max(self.q_tbl[next_key])

            new_q = reward + gamma * max_q

            updated_value = self.update_value(state, action, new_q)
            if updated_value:
                updated = True

            j -= 1

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
        e = 0.2  # random
        pivot = np.random.uniform()
        if pivot < e:
            action_list = self.env.get_allow_actions()
            if self.trajectory:
                last_direction = self.trajectory[-1][1]
                if last_direction == 0 or last_direction == 2:
                    counter_direction = last_direction + 1
                else:
                    counter_direction = last_direction - 1
                action_list.remove(counter_direction)
            action = random.choice(action_list)
        else:
            action = self.q_function.get_max_action(self.state)

        new_state, reward, self.ended = self.env.get_reward(action)

        self.trajectory.append((self.state, action, reward, new_state))
        self.state = new_state

    def is_goal(self):
        return self.ended

    def reward(self):
        return sum([trj[2] for trj in self.trajectory])

    def make_experience(self):
        self.begin()
        while not self.is_goal():
            self.do_epsilon_greedy()


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