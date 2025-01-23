import numpy as np
import pygame
import random


class BreakOut():
    
    rendering = False
    actions = ['left', 'right', 'none']

    # Colors
    playerColor = (100, 192, 30)
    brickColor = (192, 30, 30)
    x = (192, 192, 192)
    ballColor = (255, 255, 255)
    scoreColor = (90,90,90)
    backgroundColor = (187,173,160)



    # Frames per second
    fps = 15

    #Game mechanics: 
    screen_width = 800                                              #Screen settings
    screen_height = 800
    spawn_intervals = 10

    player_pos_start = screen_width/ 2, screen_height - 100         #Player settings
    ball_pos_start = screen_width / 2, screen_height - 200
    player_length = 150
    player_height = 10
    paddle_speed = 25
    bounce_angle = 25     

    ball_speed_x = ball_speed_y = -15                               #Ball settings

    reward_brick = 50
    reward_brick2 = 75 
    reward_brick3 = 100  
    reward_brick4 = 125                                                  #Rewards settings
    reward_player = 100
    reward_death = -10
    reward_time = -1 #200
    reward_won = 5000

    #brick parameters 
    brick_clms = 8
    brick_rows = 4
    brick_width = screen_width // brick_clms
    brick_height = 50
    vertical_offset = 100
    

    """""
        bricks_layout =[
                [1, 2, 3, 2,2,3,4],
                [1, 2, 3, 2,2,3,4],
                [1, 2, 3, 2,2,3,4],
                [1, 2, 3]
            ]
        
        """""

    allow_bricks = True 
    # bricks = []


    def __init__(self):
        pygame.init()
        self.reset()

    def render(self):

        #if-statement makes sure init rendering is only called once
        if not self.rendering:
            self.init_render()
        
        # Limit fps
        self.clock.tick(self.fps)

        # Clears the screen, draws element and changes the display
        self.screen.fill((self.backgroundColor))
        self.draw_game_elements()
        pygame.display.flip()

    
    def draw_game_elements(self):

        # Draw ball
        pygame.draw.circle(self.screen, (self.ballColor), self.ballrect.center, 6)

        # Draw player
        pygame.draw.rect(self.screen, (self.playerColor), self.player_rect)

        #Draw bricks
        for row in range(self.brick_rows):
            for clm in range(self.brick_clms):
                if self.bricks[row, clm] == 1:
                    x = clm * self.brick_width
                    y = row * self.brick_height + self.vertical_offset
                    pygame.draw.rect(self.screen, self.brickColor, (x, y, self.brick_width, self.brick_height))

        # Draw score
        rendered_text = self.scorefont.render(f"Score: {self.score:0.2f}", True, (self.scoreColor))
        self.screen.blit(rendered_text, (self.screen_width/2 -60, 10))

    #Closes pygame
    def close(self):
        pygame.quit()
        
    def init_render(self):
        # Display mode
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        # Caption
        pygame.display.set_caption('BreakOut baby')

        # Clock
        self.clock = pygame.time.Clock()

        # Fonts
        self.scorefont = pygame.font.Font(None, 30)

        self.rendering = True

    
    
    def reset(self):

        #Player reset
        self.player_pos = pygame.Vector2(self.player_pos_start)
        self.player_size = (self.player_length, self.player_height)
        self.player_rect = pygame.Rect(self.player_pos, self.player_size)

        #Ball reset                                                                 can be added: random_dir = random.choice([-1,1])
        self.speed = [self.ball_speed_x, self.ball_speed_y]

        self.ballrect = pygame.Rect((self.screen_width/self.spawn_intervals)*random.choice(np.arange(1,self.spawn_intervals)), self.ball_pos_start[1], 30,30)

        
        #Bricks reset
        self.bricks = np.ones((self.brick_rows, self.brick_clms))
        """""if self.allow_bricks:
            self.bricks.clear()
            self.create_bricks()"""

        #Game value reset
        self.done = False
        self.won = False
        
        self.tick = 0        
        self.score = 0


        return self.get_state()
    
    def get_state(self):


        
        # return (self.player_pos.x + self.player_length/2, self.ballrect.center[0], self.ballrect.center[1], self.speed[0], self.speed[1])
        return (self.player_pos.x + self.player_length/2, self.ballrect.center[0], self.ballrect.center[1], self.speed[0], self.speed[1], self.bricks.flatten())
    
    def step(self, action):
        self.tick += 1
        reward = 0

        self.true_player_pos_x = self.player_pos.x+self.player_length/2
        
        
        #Update player based on action
        if action == 'left':
            self.player_pos.x = max(self.player_pos.x-self.paddle_speed, 0)
        elif action == 'right':
            self.player_pos.x = min(self.player_pos.x+self.paddle_speed, self.screen_width-(self.player_length))
        self.player_rect = pygame.Rect(self.player_pos, self.player_size)

        #Update ball
        if self.ballrect.left < 0 or self.ballrect.right > self.screen_width:
            self.speed[0] = -self.speed[0]
        if self.ballrect.top < 0:
            # print("yes")
            self.speed[1] = -self.speed[1]
        self.ballrect = self.ballrect.move(self.speed)

        #Handle collision with player and gives reward


        #Handle collision with player and gives reward
        if self.player_rect.colliderect(self.ballrect):
            if abs(self.ballrect.bottom - self.player_rect.top) < abs(self.speed[1]):
                self.speed[1] = -self.speed[1]
                reward += self.reward_player
          
                offset = (self.ballrect.center[0] - self.player_rect.center[0]) / (self.player_length / 2)
                new_x_speed = offset * self.bounce_angle

                if new_x_speed < 0 and new_x_speed > -2:
                    self.speed[0] = -2
                elif new_x_speed > 0 and new_x_speed < 2:
                    self.speed[0] = 2
                else:
                    self.speed[0] = new_x_speed
                    


        #Handle collision with bricks 
        for row in range(self.brick_rows):
            for clm in range(self.brick_clms):
                if self.bricks[row, clm] == 1:
                    brick_rect = pygame.Rect(clm * self.brick_width, row * self.brick_height + self.vertical_offset, self.brick_width, self.brick_height)
                    if self.ballrect.colliderect(brick_rect):
                        # print("collosion!")
                        self.bricks[row, clm] = 0

                        if abs(self.ballrect.right - brick_rect.left) < abs(self.speed[0]) or abs(self.ballrect.left - brick_rect.right) < abs(self.speed[0]):
                            # print("change horizontal")
                            self.speed[0] = -self.speed[0]
                        elif abs(self.ballrect.bottom - brick_rect.top) > abs(self.speed[1]) or abs(self.ballrect.top - brick_rect.bottom) > abs(self.speed[1]):
                            # print("change vertical")
                            self.speed[1] = -self.speed[1]
                        
                    
                        #self.score += 1
                        if row == 0:
                            reward += self.reward_brick4
                        elif row == 1:
                            reward += self.reward_brick3
                        elif row == 2:
                            reward += self.reward_brick2
                        else:
                            reward += self.reward_brick
                        break
                        # self.bricks.reshape(-1)
        
        #Handle collision with bricks and gives reward
        """""for i in range(len(self.bricks)): 

            brickrect = self.bricks[i].brickrect
            if brickrect.colliderect(self.ballrect):

                reward += self.reward_brick

                    # Determine collision side
                if abs(self.ballrect.bottom - brickrect.top) < abs(self.speed[1]) or abs(self.ballrect.top - brickrect.bottom) < abs(self.speed[1]):
                    self.speed[1] = -self.speed[1]  # Reverse vertical direction
                elif abs(self.ballrect.right - brickrect.left) < abs(self.speed[0]) or abs(self.ballrect.left - brickrect.right) < abs(self.speed[0]):
                    self.speed[0] = -self.speed[0]  # Reverse horizontal direction
                    
                else:
                    self.speed[1] = -self.speed[1]  # Reverse vertical direction
                self.bricks.remove(self.bricks[i])
                    
                    
                break"""

        #Gives rewards based on how far the player x pos is from the ball x pos
        if not self.done:
            reward += self.reward_time

            #if self.true_player_pos_x-self.ballrect.x == 0: 
            #    self.reward_time
            #else:
            #    reward += min(self.reward_time/(abs(self.true_player_pos_x-self.ballrect.x)), self.reward_time)
        
        #Check for loss
        if self.ballrect.bottom > self.screen_height-90:
            reward += self.reward_death*(abs(self.true_player_pos_x-self.ballrect.x))
            self.done = True 
    
        
        #Check for win 
        # if self.allow_bricks:
        if np.sum(self.bricks) == 0:
            #print("win")
            reward += self.reward_won
            self.done = True
            self.won = True
        
           
        
        self.score += reward

        return (self.get_state(), reward, self.done)


    #Appends all the brick objects to the list bricks with the given amount of lives 
    """""def create_bricks(self):
        self.brick_layout = self.bricks_layout
        self.brick_count = 0
        self.bricks = []
        for row_index, row in enumerate(self.brick_layout):
            for col_index, lives in enumerate(row):
                x = (self.screen_width / 2) - 50 * len(row) + self.brick_ver_space * col_index
                y = 100 + self.brick_hoz_space * row_index
                self.bricks.append(Brick(lives, x, y, self.brick_width, self.brick_height))
                self.brick_count += 1"""

"""""class Brick: 
    def __init__(self, lives, pos_x, pos_y, width, height):
        self.lives = lives
        self.brickrect = pygame.Rect(pos_x, pos_y, width, height)"""
    
