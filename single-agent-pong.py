import random
import numpy as np
from gym import spaces

class PongEnv(object):
    """
    This environment class handles all the interactions between agent and environment.
    """
    def __init__(
        self, 
        width, 
        height, 
        adv_mechanics, 
        max_speed=0.6, 
        ball_size=10, 
        paddle_width=10, 
        paddle_height=75, 
        speed=6, 
        scale = 1, 
        goal_scoring = False,
        score_limit = 0
        reward_scheme = "HEIGHT-BASED"
    ):   
        self.adv_mechanics=adv_mechanics
        self.goal_scoring = goal_scoring
        self.reward_scheme = reward_scheme
        self.score_limit = score_limit

        width = width * scale
        height = height * scale
        paddle_height = paddle_height * scale

        # Defining the observation space by their minimum and maximum values
        self.observation_space = spaces.Box(
            low= np.array([0, 0, -max_speed, -max_speed, 0, -2]), 
            high = np.array([width, height, max_speed, max_speed, height, 2]),
            dtype=np.float32)
        
        # Defining the action space as a continous one between 0 and 1
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        # init ball position
        self.ball_x = width / 2
        third_height = height / 3
        self.ball_y = random.uniform(third_height, third_height * 2)
        
        # init left paddle height (x is fixed) and vel
        self.left_y = height / 2
        self.left_vel = 0
        
        # init right paddle height (x is fixed) and vel
        self.right_y = height / 2
        self.right_vel = 0
        
        # decide serving direction of the ball
        initial = random.randint(0,1) == 1
        if initial:
            # start the ball going to the right
            self.ball_vel_x = random.uniform(0.2, 0.5)
        else:
            # start the ball going to the left
            self.ball_vel_x = random.uniform(-0.2, -0.5)
        
        # randomize the angle the ball starts with
        self.ball_vel_y = random.uniform(-0.5, 0.5)
        
        self.width = width
        self.height = height
        self.ball_size = ball_size
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.speed = speed
        self.max_speed = max_speed
        self.game_bottom = 0
        self.game_top = self.height
        self.total_score = 0
        self.score_tally = [0, 0]
        self.number_hits = 0
        self.viewer = None
        
    def seed(self, input_seed):
        """
        This method is needed for the algorithm and environment handler.
        """
        pass

    def update_right_paddle(self):
        """
        This method gets a basic opponent move for the right paddle and updates the position of the paddle
        """
        self.right_vel = self.get_basic_move(self.right_y) * 2

        # check bounds
        if self.right_y + (self.paddle_height / 2) > self.game_top:
            # checks top
            self.right_y = self.game_top - (self.paddle_height / 2)
            
        elif self.right_y - (self.paddle_height / 2) < self.game_bottom:
            # checks bot
            self.right_y = self.game_bottom + (self.paddle_height / 2)
        else:
            # update position
            self.right_y += self.right_vel

        
        
    def update_left_paddle(self, action=1.0):
        """
        This method takes an action and updates the position of the left paddle according to the action
        """
        self.left_vel = action * 2
        
        # check bounds
        if self.left_y + (self.paddle_height / 2) > self.game_top:
            # checks top
            self.left_y = self.game_top - (self.paddle_height / 2)
            
        elif self.left_y - (self.paddle_height / 2) < self.game_bottom:
            # checks bot
            self.left_y = self.game_bottom + (self.paddle_height / 2)
        else:
            # update position
            self.left_y += self.left_vel

    
    def check_speed(self, speed):
        """
        This method is used to ensure the ball does not go too quickly through the environment
        """
        return np.clip(speed, -self.max_speed, self.max_speed)
    
    def check_hit(self, ball_rect, paddle_center):
        """
        This method checks whether the ball has made contact with a paddle
        """
        paddle_top = paddle_center + (self.paddle_height / 2)
        paddle_bot = paddle_center - (self.paddle_height / 2)
        
        ball_hit_bot = (paddle_bot) <= ball_rect.bottom <= (paddle_top)
        ball_hit_top = (paddle_bot) <= ball_rect.top <= (paddle_top)
        
        hit = ball_hit_bot or ball_hit_top
        
        return hit

        
    def update_ball(self):
        """
        This method handles all the ball physics, updating the position and speed
        Also checks for goals scored or paddles hit
        """
        # flags for whether anything interesting happened
        self.hit_edge_left = False
        self.hit_edge_right = False
        self.hit_paddle_left = False
        self.hit_paddle_right = False

        # max speed check
        self.ball_vel_x = self.check_speed(self.ball_vel_x)
        self.ball_vel_y = self.check_speed(self.ball_vel_y)
        
        
        projected_x = self.ball_x + (self.speed * self.ball_vel_x)
        projected_y = self.ball_y + (self.speed * self.ball_vel_y)
                
        # check y position bounds
        if (projected_y + (self.ball_size / 2)) > self.game_top:
            self.ball_y = self.game_top - (self.ball_size / 2)
        elif (projected_y - (self.ball_size / 2)) < self.game_bottom:
            self.ball_y = self.game_bottom + (self.ball_size / 2)
        else:
            self.ball_y = projected_y
            
        # check x position bounds
        if (projected_x + (self.ball_size / 2)) > (self.width - self.paddle_width):
            self.ball_x = self.width - self.paddle_width - (self.ball_size / 2)
        elif (projected_x - (self.ball_size / 2)) < (self.paddle_width):
            self.ball_x = self.paddle_width + (self.ball_size / 2)
        else:
            self.ball_x = projected_x
	
        # rect = left top width height
        self.ball_rect = Rect(
            (self.ball_x - self.ball_size / 2),
            (self.ball_y + self.ball_size / 2),
            self.ball_size,
            self.ball_size,
        )
        
        
        # ball hit right side - paddle or wall
        if self.ball_rect.right >= self.width - self.paddle_width:
            # check if hits paddle
            if self.check_hit(self.ball_rect, self.right_y):
                # send ball back the other way (ensures -ve direction)
                self.ball_vel_x = -abs(self.ball_vel_x)

                if self.adv_mechanics:
                    # speed modifier stuff
                    y_distance = self.ball_y - self.right_y
                    normalised_distance = y_distance / (
                        (self.paddle_height / 2) + (self.ball_size / 2)
                    )
                    # this gives a sweet spot of 0.2 dist from center
                    # if hit in this area it will slow down the ball
                    speed_modifier = abs(normalised_distance) + 0.8
                    self.ball_vel_x = self.check_speed(self.ball_vel_x * speed_modifier)
                    self.ball_vel_y = self.check_speed(self.ball_vel_y * speed_modifier)
                    
                    # angle mod means that a distance < 0.15 from center
                    # will reduce angle of returns, anything else increases
                    angle_modifier = abs(normalised_distance) + 0.85
                    self.ball_vel_y = self.check_speed(self.ball_vel_y * angle_modifier)
  
                self.hit_paddle_right = True
                
            else:
                # ball hit right side
                self.ball_vel_x *= -1
                self.score_tally[0] += 1
                self.hit_edge_right = True
                if self.goal_scoring:
                  print("Agent  Scored!")
                  self.soft_reset()
                

        # ball hit left side - paddle or wall
        elif self.ball_rect.left <= self.paddle_width:
            # check if hits paddle
            if self.check_hit(self.ball_rect, self.left_y):
                # send ball back the other way (ensures +ve direction)
                self.ball_vel_x = abs(self.ball_vel_x)
                if self.adv_mechanics:
                    # speed modifier
                    y_distance = self.ball_y - self.left_y
                    normalised_distance = y_distance / (
                        (self.paddle_height / 2) + (self.ball_size / 2)
                    )
                    # this gives a sweet spot of 0.2 dist from center
                    # if hit in this area it will slow down the ball
                    speed_modifier = abs(normalised_distance) + 0.95
                    self.ball_vel_x = self.check_speed(self.ball_vel_x * speed_modifier)
                    self.ball_vel_y = self.check_speed(self.ball_vel_y * speed_modifier)
                    
                    # angle mod means that a distance < 0.15 from center
                    # will reduce angle of returns, anything else increases
                    angle_modifier = abs(normalised_distance) + 0.85
                    self.ball_vel_y = self.check_speed(self.ball_vel_y * angle_modifier)
                        
                self.hit_paddle_left = True

            else:
                self.ball_vel_x *= -1
                self.hit_edge_left = True
                self.score_tally[1] += 1
                if self.goal_scoring:
                  print("Default opponent scored!")
                  self.soft_reset()

        # ball hit top
        if self.ball_rect.top >= self.game_top:
            self.ball_vel_y = -abs(self.ball_vel_y)

        # ball hit bottom
        elif self.ball_rect.bottom <= self.game_bottom:
            self.ball_vel_y = abs(self.ball_vel_y)

    def check_states(self, list_of_states):
      """
      This method ensures that there is no unnecessary nesting of states being returned to the agent.
      """
      new_list=list_of_states
      for index, state in enumerate(list_of_states):
        if isinstance(state, np.ndarray):
          new_list[index]=state[0]
      return new_list


    def step(self, action, basic=False):
        """
        This method handles the physics of the environment at each time step, it is given an action and returns the observation.
        """ 
        
        self.update_left_paddle(action)  # (agent controls left paddle)
        self.update_right_paddle()
        self.update_ball()

        state = [
            self.ball_x,
            self.ball_y,
            self.ball_vel_x,
            self.ball_vel_y,
            self.left_y,
            self.left_vel,
        ]

        state = self.check_states(state)

        state = np.array(state)


        done = False
        if(self.score_tally[0] > self.score_limit):
          done = True
          print("Agent won the game!")
        elif(self.score_tally[1] > self.score_limit):
          done = True
          print("Default opponent won the game!")

        if self.reward_scheme =="HIT-BASED":
            # If using the hit based reward scheme, then reward the agent for hitting the ball
            reward = 10.0 if self.hit_paddle_left else 0.0
        elif self.reward_scheme =="HEIGHT-BASED":
            # If using the height based reward scheme, then reward the agent proportional to how close it is to the height of the ball
            reward = 1 + -abs(self.ball_y - self.left_y) / self.height

        self.number_hits = 1 if self.hit_paddle_left else 0.0
        

        return state, reward, self.number_hits, done, {}
        
    def get_basic_move(self, paddle_coords):
        """
        This method provides a way for a single agent to play against a hand-coded opponent.
        This opponent will always attempt to match the height of the ball with the paddle.
        """
        difference = self.ball_y - paddle_coords
        if difference > 0:
            curr_vel = 1
        else:
            curr_vel = -1
        return curr_vel

    def render(self, screen_width, screen_height, mode="human"):
        """
        This provides a way for the environment to be rendered on a screen to be seen by humans.
        Makes use of the OpenAI rendering tools.
        """
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            pitch = rendering.FilledPolygon(
                [(0, 0), (0, screen_height), (screen_width, screen_height), (screen_width, 0)]
            )
            pitch.set_color(0, 0, 0)
            self.viewer.add_geom(pitch)

            ball = rendering.make_circle(self.ball_size / 2)
            ball.set_color(1, 1, 1)
            self.ball_trans = rendering.Transform()
            ball.add_attr(self.ball_trans)
            self.viewer.add_geom(ball)
            
            l = -self.paddle_height // 2
            r = self.paddle_height // 2
            t = self.paddle_width // 2
            b = -self.paddle_width // 2
            
            left = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.left_trans = rendering.Transform()
            left.add_attr(self.left_trans)
            left.set_color(0.1, 0.5, 0.5)
            self.viewer.add_geom(left)
            
            right = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.right_trans = rendering.Transform()
            right.add_attr(self.right_trans)
            right.set_color(0.9, 0.5, 0.5)
            self.viewer.add_geom(right)

        self.ball_trans.set_translation(self.ball_x, self.ball_y)
        self.right_trans.set_translation((self.width - (self.paddle_width / 2)), self.right_y)
        self.left_trans.set_translation(self.paddle_width / 2, self.left_y)
        
        self.right_trans.set_rotation(1.5708)
        self.left_trans.set_rotation(1.5708)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    
    def soft_reset(self):
        """
        This method provides a way for the environment to reset the game state without resetting the environment itself, useful if goal scoring is enabled.
        """
        # init ball position
        self.ball_x = self.width // 2
        third_height = self.height / 3
        self.ball_y = random.uniform(third_height, third_height * 2)

        # init left paddle height (x is fixed) and vel
        self.left_y = self.height // 2
        self.left_vel = 0

        # init right paddle height (x is fixed) and vel
        self.right_y = self.height // 2
        self.right_vel = 0
        
        # decide serving direction of the ball
        initial = random.randint(0,1) == 1
        if initial:
            # start the ball going to the right
            self.ball_vel_x = random.uniform(0.2, 0.5)
        else:
            # start the ball going to the left
            self.ball_vel_x = -random.uniform(0.2, 0.5)
        
        # randomize the angle the ball starts with
        self.ball_vel_y = random.uniform(-0.5, 0.5)
        

        state = [
            self.ball_x,
            self.ball_y,
            self.ball_vel_x,
            self.ball_vel_y,
            self.left_y,
            self.left_vel,
        ]

        state = self.check_states(state)

        state = np.array(state)

        return state
    
    def reset(self):
        # reset score tally
        self.score_tally = [0, 0]
        self.number_hits = 0

        # init ball position
        self.ball_x = self.width // 2
        third_height = self.height / 3
        self.ball_y = random.uniform(third_height, third_height * 2)

        # init left paddle height (x is fixed) and vel
        self.left_y = self.height // 2
        self.left_vel = 0

        # init right paddle height (x is fixed) and vel
        self.right_y = self.height // 2
        self.right_vel = 0
        
        # decide serving direction of the ball
        initial = random.randint(0,1) == 1
        if initial:
            # start the ball going to the right
            self.ball_vel_x = random.uniform(0.2, 0.5)
        else:
            # start the ball going to the left
            self.ball_vel_x = -random.uniform(0.2, 0.5)
        
        # randomize the angle the ball starts with
        self.ball_vel_y = random.uniform(-0.5, 0.5)

        state = [
            self.ball_x,
            self.ball_y,
            self.ball_vel_x,
            self.ball_vel_y,
            self.left_y,
            self.left_vel,
        ]

        state = self.check_states(state)

        state = np.array(state)
        return state

class Rect(object):
    """
    This is a helper class used for detecting collisions between the ball and the paddle
    """
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = self.left + width
        self.bottom = self.top - height
