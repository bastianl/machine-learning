from __future__ import division
import random
import operator
from itertools import product
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


ALL_ACTIONS = [None, 'forward', 'right', 'left']


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, learning_rate=0.44, discount=0.56, epsilon=0.3):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # q-learning params
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        # all default q values should be 1
        self.Q = defaultdict(lambda: defaultdict(lambda: 1))
        # Initialize any additional variables here
        self.total_reward = 0
        self.important_states = ['light', 'oncoming', 'left']
        self.scores = []
        # runs variable used to scale epsilon
        self.runs = 1

    def reset(self, destination=None):
        print("Trip Complete! Total reward: {}".format(self.total_reward))
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0
        self.runs += 1

    def assemble_state(self, inputs, next_waypoint, deadline):
        # filter out additional items
        valid_inputs = {k: v for k, v in inputs.items()
                        if k in self.important_states}
        # we also want to keep the next_waypoint state
        valid_inputs['next_waypoint'] = next_waypoint
        return valid_inputs

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # update state in the pygame window
        self.state = self.assemble_state(inputs,
                                         self.next_waypoint,
                                         deadline)

        # Select action according to your policy
        action = self.get_q_action(self.state)

        # naive action (random)
        # actions = self.get_naive_action()

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        my_states = self.env.agent_states[self]
        if my_states['location'] != my_states['destination']:
            pass
        # TODO: Learn policy based on state, action, reward
        self.learn_Q(action, reward)

        print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}"
              .format(deadline, inputs, action, reward))

    def learn_Q(self, action, reward):
        next_waypoint = self.planner.next_waypoint()
        next_inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        next_state = self.assemble_state(next_inputs, next_waypoint,
                                         deadline)
        # get max expected value at next location
        best_future_value = max(
            [self.Q[hash_dict(next_state)][next_action]
             for next_action in ALL_ACTIONS])

        rates = self.Q[hash_dict(self.state)]
        old_value = rates[action]

        # use q learning formula to update new value
        utility = reward + self.discount * best_future_value - old_value
        new_value = old_value + self.learning_rate * utility
        rates[action] = new_value

    def get_q_action(self, state):
        actions = self.Q[hash_dict(state)]

        # save the state for debugging later
        # but remove it from the dict for max calculation.
        actions['state'] = state
        actions = deepcopy(actions)
        del actions['state']
        items = actions.items()
        # if we haven't learned any actions yet, return a random one
        # incorperate randomness of "epsilon" to encourage learning
        # and avoiding local minima. However, we decrease this probability
        # proportionally to the timestep so that as the agent learns,
        # we avoid erroneous decisions later on.
        if (len(items) == 0 or
                random.random() < self.epsilon / self.runs):
            return self.get_naive_action()

        max_action = self.get_best_action(items)
        # get the action with the highest reward
        if actions[max_action] < 0:
            # if the best action has negative reward,
            # return a random action that has not been tried
            # if every action has been tried and has bad reward
            # return a random action.

            valid_actions = [x for x in ALL_ACTIONS
                             if x not in actions.keys()]
            if len(valid_actions) == 0:
                valid_actions = ALL_ACTIONS
            return self.get_naive_action(valid_actions)
        return max_action

    def get_best_action(self, items):
        return max(items, key=operator.itemgetter(1))[0]

    def get_naive_action(self, actions=ALL_ACTIONS):
        return random.choice(actions)


def hash_dict(dictionary):
    """Hash a dictionary.
    Source: http://stackoverflow.com/a/5884123"""
    return hash(frozenset(dictionary.items()))


class MyEnvironment(Environment):

    def __init__(self, *args, **kwds):
        super(MyEnvironment, self).__init__(*args, **kwds)
        self.scores = []

    def reset(self):
        agent = self.primary_agent
        state = self.agent_states[agent]
        # first run does not have destination.
        if 'destination' in state:
            finished = state['location'] == state['destination']
            self.scores.append(dict(steps=self.t, reward=agent.total_reward,
                                    finished=finished))
        super(MyEnvironment, self).reset()


def print_q_values(agent):
    # print learned states
    for _, actions in agent.Q.items():
        state = actions.pop('state', None)
        print("state {} yielded rewards {}".format(state, actions.items()))


def run():
    """Run the agent with GUI"""
    scores = []
    states = []
    for _ in range(100):
        e = MyEnvironment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent, learning_rate=0.78,
                           discount=0.22)  # create agent
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

        # Now simulate it
        # create simulator (uses pygame when display=True, if available)
        sim = Simulator(e, update_delay=0.0, display=False)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=100)  # run for a specified number of trials

        agent = sim.env.primary_agent
        df = pd.DataFrame(e.scores)
        K = 10
        score = np.sum(df['finished'][-K:]) / K * 100
        print("Percentage completed in last {} runs: {:.2f}%"
              .format(K, score))
        # filter out erroneous states that may have been added
        # when looking at the next state. This occurs right before
        # the agent reaches its destination, and next_waypoint = None.
        visits = len([key for key, val in agent.Q.items() if 'state' in val])
        print_q_values(agent)
        print("{} states visited".format(visits))
        scores.append(score)
        states.append(visits)
    print("Scores: {}".format(scores))
    print("States visited: {}".format(states))
    print("Mean Score: {}".format(np.mean(scores)))
    print("Mean States visited: {}".format(np.mean(states)))


def calculate_params():
    """Use a grid search to determine the best params
    for learning rate and discount params."""

    # Set up environment and agent
    rates = np.linspace(0, 1, 10)
    discount = np.linspace(0, 1, 10)
    total_scores = []

    for rate, discount in product(rates, discount):
        rates = []
        discounts = []
        scores = []
        rewards = []
        # merge 5 best runs for each rate / discount pair
        for i in range(5):
            e = MyEnvironment()  # custom env tracks scores
            a = e.create_agent(LearningAgent, learning_rate=rate,
                               discount=discount)
            e.set_primary_agent(a, enforce_deadline=True)

            sim = Simulator(e, update_delay=0.0, display=False)

            sim.run(n_trials=100)

            # print the rewards associated with each action
            df = pd.DataFrame(e.scores)

            # normalized reward - higher distance usually means
            # higher reward
            df['norm_reward'] = df['reward'] / df['steps']
            # ax = df['norm_reward'].plot()
            # fig = ax.get_figure()
            # fig.savefig('runs/run{}.png'.format(N))

            agent = sim.env.primary_agent
            print_q_values(agent)
            # calculate reward from last 2 runs, as well as the rate
            # the agent reached its destination
            # store these in the `scores` array for later analysis
            K = 20
            score = np.sum(df['finished'][-K:]) / K * 100
            mean_reward = np.mean(df['norm_reward'][-K:])
            rates.append(rate)
            discounts.append(discount)
            scores.append(score)
            rewards.append(mean_reward)
        total_scores.append(dict(rate=np.mean(rates),
                                 discount=np.mean(discounts),
                                 score=np.mean(scores),
                                 mean_reward=np.mean(rewards)))

    df = pd.DataFrame(total_scores)
    print("Best Rates:\n{}".format(
        df.sort_values(by='mean_reward', ascending=False).head()))


if __name__ == '__main__':
    run()
    # calculate_params()
