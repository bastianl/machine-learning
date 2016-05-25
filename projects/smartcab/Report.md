## Basic Driving Agent

The first thing we did was to implement a basic driving agent. This agent selects a random move from `[None, 'forward', 'right', 'left']` at each iteration, regardless of the input variables, and deadline. We observed that the agent would occasionally reach the destination, although typically well after the deadline had passed. Sometimes the hard deadline of -100 would prevent the agent from proceeding.

## Identifying States

There are several states which we deem important in deciding on an action.

* 'next_waypoint' is important, as it is our desired direction.
* 'light' is important, as the agent is penalized when going 'forward' or 'left' on a green light. 
* 'oncoming' is important as the agent is penalized when going left with oncoming traffic. 
* 'left' is important as we can only turn right if there is no traffic coming from the left, or oncoming traffic turning left.

States which are not important

* 'duration': we don't care about the duration, as when the duration ends a new trip will start, but the agent is not penalized for this. Also, incorperating duration would make an impractical amount of states, requiring a huge amount of trails in order for the agent to learn.
* 'right': we don't care about any cars to the right, as they do not affect our ability to go 'forward', 'right', or 'left'. (If the agent is at a red light, it can't proceed anyways.)

## Q-Learning

At first the agent meanders around randomly, but after that it slowly learns how to drive! Due to this initial meandering, many of the rewards we see are negative. In order to encourage the agent to seek out the best action, we choose a random move if the agent has only sought out moves with non-positive rewards. This prevents the agent from becoming stuck in local maximums. Eventually, as the agent tries all combination of moves this random restarting is no longer necessary.

## Enhanced Driving

The initial Q-learning implementation worked, but applying a learning rate of `0.25`, and a discount factor of `0.8` caused the learning agent to prioritize new rewards, and therefore learn slightly faster. 
Within 100 trails, the agent finishes almost every task on time, and with a positive reward.
Something interesting to note is that the agent learns to turn right at red lights, instead of waiting. This is due to a reward of `0.0` being given for waiting at a red light. Since waiting at a red light can sometimes be the correct action (for instance, if the next_waypoint is 'forward', or 'left'), the agent should be given a positive reward in these cases. 
[Considering these cases](./smartcab/environment.py#L196-L197) causes the learning agent to learn to stop at red lights as expected. This is closer to an optimal policy.
However even without these changes to the reward system, the agent learns a good policy. After a certain number of trails, it reaches the destination on time nearly every time, and always recieves a positive reward.
