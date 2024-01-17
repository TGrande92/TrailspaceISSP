# Reinforcement Learning Model for Flight Control Using JSBSim

## Team Members

-   Raphael Eustaquio
-   Tyler Grande
-   Caleb Seeman
-   Justin Tan
-   Ian
-   Aashay Bharadwaj
-   Andrew

## Table of Contents

1.  [JSBSim Setup Guide](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#jsbsim-setup-guide)
2.  [Reinforcement Learning Model Documentation (JSBmodel.py)](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#reinforcement-learning-model-documentation)
3.  [JSBSim Interface Documentation (JSBSim.py)](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#jsbsim-interface-documentation)
4.  [References and Resources](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#references-and-resources)
5.  [Visualization Tool](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#visualization-tool)

## JSBSim Setup Guide

Follow these steps to set up JSBSim successfully. Ensure each step is executed correctly to avoid issues.

### Prerequisites

-   Python 3.11
-   Git

### Installation Steps

1.  **Install Python 3.11**
    
    -   Required for PyTorch compatibility.
2.  **Install Git**
    
    -   Necessary for cloning repositories.
3.  **Install Modules**
    
    -   Execute commands:
        
        
        `pip install torch`
        `pip install jsbsim` 
        
4.  **Install JSBSim**
    
    -   **Windows**:
        -   Download and install from JSBSim Releases v1.2.0.
    -   **Ubuntu Linux**:
        -   Debian packages available in the JSBSim Release Section for Ubuntu 20.04 LTS and 22.04 LTS (64-bit).
        -   Install `JSBSim_1.2.0-1191.amd64.deb`, `JSBSim-devel_1.2.0-1191.amd64.deb`, `python3-JSBSim_1.2.0-1191.amd64.deb`.
5.  **Clone Rascal110 Model**
    
    -   Example for Windows:
        
        
        `git clone https://github.com/ThunderFly-aerospace/FlightGear-Rascal.git` 
        
    -   Rename the cloned folder to “Rascal110-JSBSim”.
6.  **Clone the Main Model**
    
    -   As of January 17th, 2024, Tyler Grande updated the link:
      
        
        `git clone -b JSB-Model https://github.com/TGrande92/TrailspaceISSP.git` 
        
7.  **Run the Model**
    
    -   Navigate to the algo folder and execute `py JSBmodel.py`.

## Reinforcement Learning Model Documentation (JSBmodel.py)

This section explains the RL algorithm implementation for flight control using the DQN approach with JSBSim. Please refer to the word document for a further indepth explanation of the models and functions.

### Key Concepts

-   **Epsilon Greedy Exploration**
-   **Batch Size**
-   **Gamma (Discount Factor)**
-   **Tau (Soft Update Parameter)**
-   **LR (Learning Rate)**

### Modules and Classes

-   **Transition**
-   **DQN (Deep Q-Network)**
-   **ReplayMemory**
-   **Saving and Loading Functions**

### Model Workflow

-   **Action Space**
-   **DQN Initialization**
-   **Epsilon Decay**
-   **Action Selection**
-   **Reward Functions**
-   **Model Optimization**
-   **Run Logging**
-   **State Processing and Action Execution**
-   **Validation Functions**
-   **Model Validation**
-   **Main Execution (Training Loop)**
-   **Global Variables in Main Function**

## JSBSim Interface Documentation (JSBSim.py)

Overview of the `JsbsimInterface` class for JSBSim interaction.

### Class: JsbsimInterface

-   Methods for simulation control and data retrieval.
-   **Methods**:
    -   `__init__(self)`
    -   `get_sim_time(self)`
    -   `start(self)`
    -   `get_state(self)`
    -   `get_altitude_change(self)`
    -   `set_controls(self, elevator, aileron, rudder)`
    -   `run(self)`
    -   `stop(self)`
    -   `gear_contact(self)`

### Usage

-   Explains how the class is utilized within the RL environment.

## References and Resources

-   [Learning to Optimize with RL - Berkeley Blog](https://bair.berkeley.edu/blog/2017/09/12/learning-to-optimize-with-rl/)
-   [OpenAI Baselines: PPO](https://openai.com/research/openai-baselines-ppo)
-   [Quadcopter RL Projects on GitHub](https://chat.openai.com/c/c2f690ff-8500-48cd-a95b-2965d5d9d468#)
-   [Reinforcement Q Learning Tutorial - PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
-   [Research Paper: RL in Dynamic Environments](https://arxiv.org/pdf/2008.03162.pdf)

## Visualization Tool

Run `simple_visualization.py` to view machine learning plotting. Ensure dependencies are installed first.
