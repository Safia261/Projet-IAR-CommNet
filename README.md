# Learning Multiagent Communication with Backpropagation

## Project Overview

This project explores the use of **CommNet (Communication Network)**, a multi-agent neural network architecture with differentiable communication, in several environements. This model is defined by Sukhbaatar et al. in their article [Learning Multiagent Communicationwith Backpropagation](https://arxiv.org/pdf/1605.07736) (2016).

The goal is to study whether explicit communication between agents enables better coordination and improved performance in **competitive multi-agent tasks**, compared to independent agents.

We considered three environments:
1. A **Lever Pulling environment**, implemented from scratch.
2. A **Traffic Junction environment**, implemented from scratch (visual render with tkinter).
3. A **CombatTask environment**, implemented from scratch (visual render with tkinter). (As well as an existing **Combat-v0 environment** from the [`ma-gym`](https://github.com/koulanurag/ma-gym/tree/master) library).

Despite a correct implementation of the model and training pipeline, the experimental results highlight the **practical limitations of CommNet trained with simple policy-gradient methods** in complex combat scenarios.

---
## Installation

We recommend you to create a virtual environment to install the necessary dependencies and run the code.

The dependencies are: \
`torch gym numpy tk`

If you want to run ``train_combatv0.py`` that uses the Combat Task environment from ma-gym, you need to install ``ma-gym``. Careful, this library requires a python version strictly under 3.12. We recommend you Python 3.10.

---

## Model: CommNet

CommNet is a neural architecture designed for cooperative multi-agent reinforcement learning.  
Its key characteristics are:

- Shared parameters across agents
- Differentiable communication channel
- Iterative message passing between agents
- End-to-end learning via policy gradient

### Architecture

Each agent:
1. Encodes its local observation into a latent representation
2. Exchanges information with other agents via a communication vector (mean of hidden states)
3. Updates its internal state through multiple communication steps
4. Decodes its final hidden state into an action distribution (discrete)

The implementation follows the original CommNet paper and supports:
- Independent agents
- Fully connected baseline
- Discrete communication
- Continuous communication (main CommNet model)

---

## Environments

### 1. Lever Pulling Environment

A simple cooperative multi-agent environment designed to study explicit communication between agents:
- Fully cooperative task
- Each agent has only partial knowledge of the correct lever
- Each agent needs to pull exactly one lever
- Sparse binary reward (success / failure)
- Communication is necessary to solve the task

### 2. Traffic Junction Environment

A cooperative multi-agent environment inspired by traffic management at an intersection:
- Multiple agents representing vehicles approaching a junction
- Partial observability with local perception
- Simple discrete actions (gas / break)
- Negative rewards for collisions
- Requires coordinated decision-making among agents


### 3. Custom CombatTask Environment

A grid-based combat environment implemented specifically for this project, featuring:
- Two opposing teams of agents
- Discrete movement and attack actions
- Partial observability
- Sparse terminal rewards (win / loss / draw)

### 4. Combat-v0 (ma-gym)

An existing benchmark environment from the `ma-gym` library:
- Standardized multi-agent API
- Competitive team-based combat
- Discrete observations and actions
- Sparse rewards

---

## Training Method

- Policy-gradient based learning (REINFORCE-style)
- Shared policy across agents
- Cooperative reward aggregation

Training was performed with:
- Small batch sizes (hardware constraints)
- Multiple updates per epoch
