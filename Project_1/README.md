# Project 1 - AlphaGo knock-off: On-Policy Monte Carlo Tree Search for Game Playing
The goal of this project is to implement a general-purpose Monte Carlo Tree search (MCTS) system for use in a complex 2-person game, Hex. This system should employ a neural network as the target policy for on-policy MCTS (i.e., the target policy and behavior policy are the same), and should be able to be trained and re-deployed in a head-to-head competitions with other networks. 

## Overview
Implement MCTS and combine it with Reinforcement Learning (RL) and Deep Learning (DL) to play Hex. This explosion in possible game states neccessitates the use of function approximators (e.g. neural nets) for the value function policy. A policy is the mapping from game states to a probability distribution over the possible actions. In this project we implement that policy as a neural-net approximation. 

Runs of MCTS will provide target probability distributions for training the policy network, which will fulfill the roles of both target policy and behavior policy, thus yielding an on-policy version of MCTS. Once trained via MCTS, the policy network will participate in competitions against other Hex-playing agents.

Strenght of MCTS over MiniMax:
- Does not need heuristics, only need to know the rules of the game and how to evaluate final states. 
- Even advanced Go players strugle to create proper heuristics. Instead of heursitics, MCTS perform hundreds or thousands of ollouts and averages the result across the search trees as a basis of each move in the actual game. Over time, the evaluations of nodes and edges in the tree approach those achieved by MiniMax, but without the need for a heuristic

Parts of the program:
- A game manager
- MCTS module
- RL module
- Neural-net package (tensorflow/PyTorch)

## Kernel MCTS System
MCTS details: Monte-Carlo tree search and rapid action value estimation in computer Go (Gelly and Silver, 2011). The implementation of MCTS must perform these fur basic processes (one simulation of MCTS): 
1. Tree Search: Traversing the tree from the root to a leaf node by using the tree policy.
2. Node Expansion: Generating some or all child states of a parent state, and then connecting the tree node housing the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
3. Leaf Evaluation: Estimating the value of a leaf node in the tree by doing a rollout simulation using the default policy from the leaf node’s state to a final state. 
  - Optional: Critic as a supplement to the rollout
4. Backpropagation: Passing the evaluation of a final state back up the tree, updating relevant data (see course lecture notes) at all nodes and edges on the path from the final state to the tree root.
**Each process must exist as a separate, modular unit in your code**: </br >
A method, function or subroutine. These units should
be easy to isolate and explain during a demonstration of your working system.

### Rollouts
Perform **the entire** rollout game of MCTS to asses the value of the leaf node, not just one simulation. Must be able to perform M (default=500) simulations of MCTS. M can be dynamic; increase it as the game progresses.

## Simple Preliminary Games
Use simpler games while developing the MCTS algorithm, for example *NIM* and *The old gold game*.

### A basic NIM Game
TODO
### The Old Gold Game
TODO

## Hex
Hex is ....

### Game rules of Hex: 
- ...

## On-Policy Monte Carlo Tree Search
On policy = Policy employed for exploring the state space (aka. behavior policy). Also the policy that is gradually refined via learning (a.k.a target policy). On policy MCTS involves:

- Behavior policy = target policy = default policy: The policy used to perform rollouts from a leaf node to a final state in hte MCTS tree. Improving as we go. Keep exploration using epsilon.
- Tree Policy: Controls searches from the trees root to a leaf.

In this project, a neural network (e.g. ANET) constitues the target policy. It takes the board state as input and produces probbability distributions over all possible moves (from the state) as output.

Main goal of MCTS: Produce an intelligent target policy which can be used independently of MCTS as an actor module. 

### The Reinforcement Learning (RL) Algorithm
1. Perform an episode: Making actual moves in a game. An episode is a completed game.
2. Seach moves: Making simulated moves during MCTS
3. Updating the target policy via supervised learning.

**See lecture notes** 


### **Optional**: Adding a Critic
Do not remove rollouts, since it is an important exercies, but you are fee to add a critic that represent the probability of doing a rolout during any given visit to a leaf node. Initially, $\sigma$=1.0 and then gradually decrease it over the episodes. 
## Tournament of Progressive Policies (TOPP)
Used to gauge the improvements of your target policy over time.
1. Save current state of ANET to file
  - if `M=5` and `episodes=200`, ANET trained for `[0, 50, 100, 150, 200]` episodes. 
2. Reload into M different agents. Round robin format. 

The ANETs used in **the final version of your TOPP must be saved to file** for later use in a (very short) tournament – but one involving all of them – during the demonstration session. 


## Implementation "issues"
### Hex Board
Visualization: Find something on the internet, either display it graphically or just on the command line. Very good to create early to debug implementation. 

### State Manager
Seperate game logic from MC tree search. Both must be leanly seperated from the neural network. 
State manager should hold all game logic for Hex (perform all functions relatied to the game of Hex):
- Understand game states
- Produces initial game states
- Generate child states from a parent state
- Recognizes winning states
- ...

The **MCTS code should only make generic calls to the state manager**:
- Requesting start states, child states, confirmation of final state...

### Policy Network


### Modlarity and Flexibility of the code
Clean separation between key componenets (own classes):
- State manager
- Neural network
- MCTS algo
- RL system: Houses the actor
- Tournament of Progressive Policies (TOPP)

Handle variations:
- [ ] Hex board with size of `k`, k x k, `3 <= k <= 10` -> `k: [3,10]`
- [ ] Standard MCTS paraments, nr_epsisodes, nr_searches per actual move...
- [ ] ANET: learning rate, number of hidden layers, neurons per layer, activations functions: linear, sigmoid, yanh, **RELU**
- [ ] Optimizer in ANET, options: Adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam.
- [ ] Number (`M`) of ANET's to be cached in preparation for a TOPP
- [ ] Number of games `G` to be played between any two ANET-based agents during the round-robin play of the TOPP. 

**A separate configuration file (e.g. .env) provides an easy and efficient solution to this problem and is highly recommended.**
# Requirements
- [ ] Your system must include `M` (simulations fo MCTS followed by a rollout) as a user-specifiable parameter (Can be dynamic, increase as you go). Default: 500
- [ ] Implemented a game manager
- [ ] Implemented MCTS module
- [ ] Implemented RL module
- [ ] Implemented a neural-net package (tensorflow/PyTorch)
- [ ] Implemented TOPP module
- [ ] Handle variations in configuration
- [ ] Play in the Online Hex Tournamenet (OHT)

# Deliverables