# Reinforcement Learning Agents 
Implemented for Tensorflow 2.0+ (use version 2.1 to avoid memory leak issues)
## Usage
- Install dependancies imported in python file
- Each file contains example code that runs training on CartPole env
- Example: `python3 TF2_DQN_LSTM.py`
## Agents
All agents tested using CartPole env with modified reward
- DQN (basic and with LSTM model)
- DDPG (basic and with LSTM model; supports discrete and continuous action space)
## Models
Each agent contain models that were trained for the CartPole env, q value and reward graphs that were generated after training
## Demos
DQN Basic, time step = 4, 500 reward
![DQN Basic](DQN/gifs/test_render_basic_time_step4_reward500.gif)

DQN LSTM, time step = 4, 500 reward
![DQN LSTM](DQN/gifs/test_render_lstm_time_step4_reward500.gif)