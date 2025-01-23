# Play Breakout
import BreakOut
import inspect
print(inspect.getfile(BreakOut))
import pygame
import numpy as np
import scipy.stats as stats
scores = []
step_counts = []
score_per_step_counts = []
wins = []

def play():
    env = BreakOut.BreakOut()
    env.reset()
    action = 'none'
    exit_program = False

    #Normalize the rewards
    env.reward_player = 0
    env.reward_death = 0
    env.reward_time = 0

    env.reward_brick = 1

    env.reward_brick2 = 2 
    env.reward_brick3 = 3  
    env.reward_brick4 = 4    
    env.reward_won = 0
    score = 0
    count_step = 0 
    while not exit_program:
        # Render game
        count_step += 1
        env.render()
        
        # Process game events
        action = 'none'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    exit_program = True
                if event.key == pygame.K_RIGHT:
                    action = 'right'
                if event.key == pygame.K_LEFT:
                    action = 'left'
                if event.key == pygame.K_r:
                    env.reset()   

        # Step the enviroment
        _,reward,done = env.step(action)

        #print(env.player_pos.x)
        if done == True: 
            exit_program = True
            if env.won == True:
                wins.append(1)
            else:
                wins.append(0)

        score += reward
        

    env.close()
    scores.append(score)
    step_counts.append(count_step)
    score_per_step_counts.append(score/count_step)

antal_spil = 10
for i in range(antal_spil):
    play()
    
print(wins, score_per_step_counts, scores)    
#print(sum(wins)/antal_spil, np.mean(score_per_step_counts), np.max(score_per_step_counts), np.mean(scores))
