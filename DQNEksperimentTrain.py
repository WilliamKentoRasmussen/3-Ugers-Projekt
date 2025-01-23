# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import BreakOut
import os 
import random
import pandas as pd

# random seed
random.seed(69420)
#Loggging
DQN_average_q_values = []
DQN_average_scores = []
DQN_data_to_save = []
DQN_data_to_save2 = []

n_DQN = 5
for DQN_idx in range(n_DQN):
    print(f"\n \nStarting time of DQN model nr. {DQN_idx} is: {datetime.datetime.now()}")
   
    # Parameters
    n_games = 10000 #3000
    epsilon = 1
    epsilon_min = 0.01
    epsilon_reduction_factor = 0.9996 #måske bedre med 0.9997
    gamma = 0.99
    batch_size = 512
    buffer_size = 1000000
    learning_rate = 0.0001
    steps_per_gradient_update = 10
    max_episode_step = 5000
    input_dimension = 37
    hidden_dimension = 128
    output_dimension = 3

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
        #Ti hidden layers
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

    # State to input transformation
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

    # Environment
    env = BreakOut.BreakOut()
    action_names = env.actions        # Actions the environment expects
    actions = np.arange(3)            # Action numbers

    # Buffers
    obs_buffer = torch.zeros((buffer_size, input_dimension))
    obs_next_buffer = torch.zeros((buffer_size, input_dimension))
    action_buffer = torch.zeros(buffer_size).long()
    reward_buffer = torch.zeros(buffer_size)
    done_buffer = torch.zeros(buffer_size)

    # Training loop

    # Logging
    scores = []
    losses = []
    episode_steps = []
    step_count = 0
    print_interval = 100
    average_scores = []
    #average_q_values = []
    #  Initialize a list to store the data for each DQN
    data_to_save = []
    data_to_save2 = []

    # Training loop
    for i in range(n_games):
        # Reset game
        score = 0
        #q_values = []
        episode_step = 0
        episode_loss = 0
        episode_gradient_step = 0
        done = False
        env_observation = env.reset()
        observation = state_to_input(env_observation)

        # Reduce exploration rate
        epsilon = (epsilon-epsilon_min)*epsilon_reduction_factor + epsilon_min
        
        # Episode loop
        while (not done) and (episode_step < max_episode_step):       
            # Choose action and step environment
            if np.random.rand() < epsilon:
                # Random action
                action = np.random.choice(actions)
            else:

                # Action according to policy
                action = np.argmax(q_net(observation).detach().numpy())
            

            #   Store the Q-value for the selected action
            #q_values.append(q_net(observation)[action].item())
            #Skal det istedet kun tælle hvis værdien bliver talt

            # step the env
            env_observation_next, reward, done = env.step(action_names[action])
            observation_next = state_to_input(env_observation_next)
            score += reward

            # Store to buffers
            buffer_index = step_count % buffer_size
            obs_buffer[buffer_index] = observation
            obs_next_buffer[buffer_index] = observation_next
            action_buffer[buffer_index] = action
            reward_buffer[buffer_index] = reward
            done_buffer[buffer_index] = done

            # Update to next observation
            observation = observation_next

            # Learn using minibatch from buffer (every steps_per_gradient_update)
            if step_count > batch_size and step_count%steps_per_gradient_update==0:

                # Choose a minibatch            
                batch_idx = np.random.choice(np.minimum(
                    buffer_size, step_count), size=batch_size, replace=False)

                # Compute loss function
                out = q_net(obs_buffer[batch_idx])
                val = out[np.arange(batch_size), action_buffer[batch_idx]]   # Explain this indexing
                with torch.no_grad():
                    out_next = q_net(obs_next_buffer[batch_idx])
                    target = reward_buffer[batch_idx] + \
                        gamma*torch.max(out_next, dim=1).values * \
                        (1-done_buffer[batch_idx])
                    

                loss = loss_function(val, target)

                #print(done)

                # Step the optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                episode_gradient_step += 1
                episode_loss += loss.item()

            # Update step counteres
            episode_step += 1
            step_count += 1

        
        # track rewards, losses, and Q-values
        scores.append(score)
        losses.append(episode_loss / (episode_gradient_step+1))
        episode_steps.append(episode_step)
        
        

        if (i+1) % print_interval == 0:
            # Print average score and number of steps in episode
            average_score = np.mean(scores[-print_interval:-1])
            #[-print_interval:-1])
            average_episode_steps = np.mean(episode_steps[-print_interval:-1])
            print(f'Episode={i+1}, Score={average_score:.1f}, Steps={average_episode_steps:.0f}, epsilon = {epsilon}')

            # Graf logging
            average_scores.append(average_score)
            #average_q_values.append(average_q_value)

            DQN_data_to_save.append([DQN_idx, i+1, average_score, average_episode_steps])

            # Save model
            torch.save(q_net.state_dict(), f'Eksperiment{DQN_idx}.pt')

    DQN_data_to_save2.append([DQN_idx, i+1, scores, losses, episode_steps])
    DQN_average_scores.append(average_scores)
    #DQN_average_q_values.append(average_q_values)

    
    env.close()
    print(f"Finish time of DQN model nr. {DQN_idx} is: {datetime.datetime.now()}")

# After finishing the loop, save all data to a CSV
df = pd.DataFrame(DQN_data_to_save, columns=['DQN_idx','Episode', 'Score', 'Episode Steps'])
df.to_csv('experiment_data.csv', index=False)

df = pd.DataFrame(DQN_data_to_save2, columns=['DQN_idx', 'Episode', 'Average Score', 'Episode Steps', 'Loss'])
df.to_csv('raw_experiment_data.csv', index=False)

#%% 

with open("Eksperiment.txt", "w") as file:
    file.write(str(DQN_average_scores))

colors = ['blue', 'green', 'red', 'purple', 'black']
x = np.arange(1, n_games/print_interval+1)
for idx in range(n_DQN):
    data = DQN_average_scores[idx]
    plt.plot(x, data, color = "green", marker = "o", linestyle = "solid")
    plt.plot(x, data, label=f"DQN nr. {idx}", color=colors[idx], marker='o')
plt.title(f"DQN'er average scores per 100 episodes")
plt.xlabel("Episoder")
plt.ylabel("Average score")
plt.legend()
plt.grid(True) 
plt.show()





'''
fig = plt.figure(figsize=(10, 8))
for idx in range(n_DQN):
    
    

    plt.subplot(n_DQN, 1, idx+1)
    data = DQN_average_scores[idx]
    x = np.arange(1, len(data)+1)
    plt.plot(x, data, color = "green", marker = "o", linestyle = "solid")

    plt.title(f"DQN nr. {idx} average scores per 100 episodes")
    plt.xlabel("100 episodes")
    plt.ylabel("Average score")

plt.tight_layout()  # Adjust spacing
plt.show()

'''
# %%
