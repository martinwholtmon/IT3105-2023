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

## Tournament of Progressive Policies (TOPP)


## Implementation
### Hex Board
### State Manager
### Policy Network
### Modlarity and Flexibility of the code
# Requirements
- [ ] Your system must include `M` (simulations fo MCTS followed by a rollout) as a user-specifiable parameter (Can be dynamic, increase as you go). Default: 500
- [ ] Implemented a game manager
- [ ] Implemented MCTS module
- [ ] Implemented RL module
- [ ] Implemented a neural-net package (tensorflow/PyTorch)
- [ ] ...
- [ ] ...
- [ ] ...

# Deliverables