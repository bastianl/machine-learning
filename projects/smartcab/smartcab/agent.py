import random
import operator
from copy import deepcopy
from collections import defaultdict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


ALL_ACTIONS = [None, 'forward', 'right', 'left']


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.total_reward = 0
        self.learning_rate = 0.25
        self.discount = 0.8
        self.important_states = ['light', 'oncoming', 'left']
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))

    def reset(self, destination=None):
        print("Trip Complete! Total reward: {}".format(self.total_reward))
        self.planner.route_to(destination)
        self.total_reward = 0
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        state = {k: v for k, v in inputs.items()
                 if k in self.important_states}
        # we also want the next_waypoint state
        state['next_waypoint'] = self.next_waypoint

        # Select action according to your policy
        action = self.get_q_action(state)

        # naive action (random)
        # actions = self.get_naive_action()

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        my_states = self.env.agent_states[self]
        if my_states['location'] != my_states['destination']:
            pass
        # TODO: Learn policy based on state, action, reward
        self.learn_Q(state, action, reward)

        print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}"
              .format(deadline, inputs, action, reward))

    def learn_Q(self, state, action, reward):
        rates = self.Q[hash_dict(state)]
        utility = reward + self.discount * rates[action]
        new_rate = ((1 - self.learning_rate) * utility +
                    self.learning_rate * utility)
        rates[action] = new_rate

    def get_q_action(self, state):
        actions = self.Q[hash_dict(state)]

        # save the state for debugging later
        # but remove it from the dict for max calculation.
        actions['state'] = state
        actions = deepcopy(actions)
        del actions['state']
        items = actions.items()
        if len(items) == 0:
            # if we haven't learned any actions yet, return a random one
            return self.get_naive_action()

        # get the action with the highest reward
        max_action = max(items, key=operator.itemgetter(1))[0]
        if actions[max_action] <= 0:
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

    def get_naive_action(self, actions=ALL_ACTIONS):
        return random.choice(actions)


def hash_dict(dictionary):
    """Hash a dictionary.
    Source: http://stackoverflow.com/a/5884123"""
    return hash(frozenset(dictionary.items()))


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.3, display=True)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # print the rewards associated with each action
    for _, actions in sim.env.primary_agent.Q.items():
        state = actions.pop('state')
        print("state {} yielded rewards {}".format(state, actions.items()))


if __name__ == '__main__':
    run()
