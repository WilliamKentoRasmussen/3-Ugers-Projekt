# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
#import mdock
import BreakOut
import pygame
import scipy.stats as stats 
import random


# %% Parameters
n_games = 10000
epsilon = 1
epsilon_min = 0.01
epsilon_reduction_factor = 0.9997
gamma = 0.99
batch_size = 512
buffer_size = 1000000
learning_rate = 0.0001
steps_per_gradient_update = 10
max_episode_step = 10000
input_dimension = 37
hidden_dimension = 128
output_dimension = 3

#Logging
DQN_scores = []
DQN_average_score_per_time_step = []
DQN_win_rate = []

# random seed
random.seed(42)

#Test Parameters
render = False
n_games = 100
n_DQN = 5
for DQN_idx in range(n_DQN):
    print(DQN_idx)

    # Neural network, optimizer and loss
    if DQN_idx==0: 
        #To hidden layers
        q_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dimension)
        )
    elif DQN_idx==1: 
        #Fire hidden layers
        q_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dimension)
        )
    elif DQN_idx==2: 
        #Seks hidden layers
        q_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dimension)
        )
    elif DQN_idx==3: 
        #Otte hidden layers
        q_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dimension)
        )
    elif DQN_idx==4: 
        #ti hidden layers
        q_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dimension)
        )
    
    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # %% State to input transformation
    # Convert environment state to neural network input by one-hot encoding the state
    def state_to_input(state):

        player_x, ball_x, ball_y, speed_x, speed_y, bricks = state

        #Translates the numeric values to normalised tensors
        player_x_tensor = torch.tensor([player_x/env.screen_width])
        ball_x_tensor = torch.tensor([ball_x/env.screen_width])
        ball_y_tensor = torch.tensor([ball_y/env.screen_height])
        speed_x_tensor = torch.tensor([speed_x/env.ball_speed_x])
        speed_y_tensor = torch.tensor([speed_y/env.ball_speed_y])
        brick_tensor = torch.FloatTensor(bricks)
        # brick_tensor_flat = brick_tensor.view(-1)
            

        return torch.hstack((player_x_tensor, ball_x_tensor, ball_y_tensor, speed_x_tensor, speed_y_tensor, brick_tensor))

    # %% Environment
    env = BreakOut.BreakOut()
    action_names = env.actions        # Actions the environment expects
    actions = np.arange(3)            # Action numbers


    # Play loop
    # Load model

  
    q_net.load_state_dict(torch.load(f'Eksperiment{DQN_idx}.pt',weights_only=True))

        # Create envionment
    env = BreakOut.BreakOut()
    env.reward_player = 0
    env.reward_death = 0
    env.reward_time = 0
    env.reward_brick = 1
    env.reward_brick2 = 2
    env.reward_brick3 = 3
    env.reward_brick4 = 4
    env.reward_won = 0


    #logging 
    scores = []
    wins = 0
    average_score_per_time_step = []

    #game loop
    for _ in range(n_games):    
            # Reset game
        score = 0
        step_count = 0
        done = False
        observation = env.reset()
        episode_step = 0

            # Play episode        
        with torch.no_grad(): 
            while (not done) and (episode_step < max_episode_step):
                pygame.event.get()
                    # Choose action and step environment
                action = np.argmax(q_net(state_to_input(observation)).detach().numpy())
                observation, reward, done = env.step(action_names[action])
                step_count += 1
                score += reward
                if render: env.render()
                episode_step += 1        
            if env.won == True:
                wins +=1
    
        scores.append(score)
        
        average_score_per_time_step.append(score/step_count)

        # Close and clean up
    env.close()

    DQN_scores.append(scores)
    DQN_win_rate.append(wins/n_games)
    DQN_average_score_per_time_step.append(average_score_per_time_step)

    
for i in range(n_DQN):
    # Print basic stats
    print("Win rate:", round(DQN_win_rate[i], 3), 
          #"Var Score per timestep:", np.var(DQN_average_score_per_time_step[i], ddof=1),
          "Mean score per timestep:", round(np.mean(DQN_average_score_per_time_step[i]),3),
          "Max score per timestep:", round(np.max(DQN_average_score_per_time_step[i]),3),
          "Mean score:", round(np.mean(DQN_scores[i]),3))
          # "Var score:", np.var(DQN_scores[i], ddof=1))
    
    # Confidence Interval for Average Mean Score per Timestep
    mean_score_per_time = np.mean(DQN_average_score_per_time_step[i])
    std_error = np.sqrt(np.var(DQN_average_score_per_time_step[i], ddof=1) / 100)
    t_value = stats.t.ppf(0.975, df=100 - 1)
    margin_of_error = t_value * std_error
    print(f"margin for average mean score per timestep score: {round(margin_of_error,3)}")
    
    # Confidence Interval for Average Mean Score
    mean_score = np.mean(DQN_scores[i])
    std_error = np.sqrt(np.var(DQN_scores[i], ddof=1) / 100)
    margin_of_error = t_value * std_error
    print(f"margin for average mean score: {round(margin_of_error,3)}")

    # Confidence Interval for Win Rate
    p_hat = DQN_win_rate[i]  # Ensure this is a proportion, not a percentage
    z_value = stats.norm.ppf(0.975)  # Z-value for 95% confidence
    win_rate_margin = z_value * np.sqrt((p_hat * (1 - p_hat)) / 100)
    print(f"margin for Win rate: {round(win_rate_margin,3)}")
    print("")
    

