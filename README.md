# CoRE Final Project (2022 Autumn)
## Playing Hanabi with ToM and Intrinsic Rewards

### Introduction
Recent years have witnessed drastic advances in multi-player games such as Go and poker. However, cooperative games with imperfect information are relatively underexplored. The Hanabi Challenge is one of them, where reasoning about other agents’ mental states is brought to the foreground. It is a meaningful benchmark because it requires Theory of Mind (ToM) reasoning and challenges an agent’s ability in a partially observable and cooperative setting.

We propose two novel plug-in modules to tackle the problem and boost the performance of arbitrary RL agents in the game environment. Both of them can be attached to any RL agent easily. The Hand Card Information Completion module (HCIC) complements hidden hand card knowledge with ToM, and the Goal-Oriented Intrinsic Reward module (GOIR) encourages agents’ exploration and collaboration.

### The Hanabi Learning Environment

We use the Hanabi\_Learning\_Environment (HLE) for all experimentation. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py. Please follow the instructions to setup the environment. Our agent can be found in directory `./hanabi_learning_environment/agents/our_agent`

### Getting started
Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```
