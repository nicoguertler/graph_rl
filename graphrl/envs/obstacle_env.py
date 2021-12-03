import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from .graphics_utils import ArrowConfig, get_default_subgoal_colors

class ObstacleEnv(gym.GoalEnv):  
    metadata = {'render.modes': ['human']}   

    def __init__(self, obstacle_radius=0.5, stage_dimension=6., 
            agent_speed=5e-2, max_episode_length=300, subgoal_radius=0.033):
        super().__init__()

        self.stage_dimension = stage_dimension
        self.obstacle_radius = obstacle_radius
        self.agent_speed = agent_speed
        self.max_episode_length = max_episode_length

        desired_goal_space = spaces.Box(
                low = -1., 
                high = 1., 
                shape = (2,),
                dtype = np.float32)
        achieved_goal_space = desired_goal_space
        obs_space = desired_goal_space

        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "desired_goal": desired_goal_space,
            "achieved_goal": achieved_goal_space
            })

        self.action_space = spaces.Box(
                low = -1., 
                high = 1., 
                shape = (2,),
                dtype = np.float32)

        self.window = None

        self.window_width = 800
        self.window_height = 800
        self.background_color = (1.0, 1.0, 1.0, 1.0)

        self.obstacle_color = (0.4, 0.4, 0.4)

        self.agent_position = np.array((0., -1.0))
        self.agent_radius = 0.1
        self.agent_color = (0.0, 0.0, 0.0)

        self._draw_goal()
        self.goal_radius = 0.1
        self.goal_color = (0.0, 0.0, 0.0)

        self.current_step = 0

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        self._subgoal_colors = get_default_subgoal_colors()
        self.subgoal_radius = float(subgoal_radius)*self.stage_dimension*0.5

        self.function_grid = None 
        self.color_low = np.array((0., 0., 1.))
        self.color_high = np.array((1., 0., 0.))
        self.value_low = -10.
        self.value_high = 0.

    def _draw_goal(self):
        while True:
            candidate = np.random.normal((0., 1.), 0.3, size = (2,))
            if np.linalg.norm(candidate) > self.agent_radius + self.obstacle_radius:
                self.goal = candidate
                break

    def update_function_grid(self, values, low, high):
        self.function_grid = values
        self.value_low = low
        self.value_high = high

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.linalg.norm(achieved_goal - desired_goal) <= \
                self.goal_radius/self.stage_dimension/0.5:
                    return 0.
        else:
            return -1.

    def _get_obs(self):
        obs = { 
            "observation": self.agent_position/self.stage_dimension*2.,
            "desired_goal" : self.goal/self.stage_dimension*2.,
            "achieved_goal": self.agent_position/self.stage_dimension*2.
            }
        return obs

    @classmethod
    def map_to_env_goal(self, partial_obs):
        return partial_obs

    def step(self, action):
        self.agent_position += self.agent_speed*np.array(action)
        self.agent_position = np.clip(self.agent_position,
                -0.5*self.stage_dimension, 0.5*self.stage_dimension)
        distance_to_center = np.linalg.norm(self.agent_position)
        if distance_to_center < self.obstacle_radius + self.agent_radius:
            self.agent_position *= (self.obstacle_radius + self.agent_radius)/distance_to_center
        info = {}
        obs = self._get_obs()
        reward = self.compute_reward(self.agent_position/self.stage_dimension*2., 
                self.goal/self.stage_dimension*2., info)
        self.current_step += 1
        done = reward == 0. or self.current_step >= self.max_episode_length
        return obs, reward, done, info
  
    def reset(self):
        self.agent_position = np.array((0., -1.0))
        self.current_step = 0
        self._draw_goal()
        return self._get_obs()

    def update_subgoals(self, subgoals):
        self._subgoals = [np.array(sg)*self.stage_dimension*0.5 for sg in subgoals]

    def update_timed_subgoals(self, timed_subgoals, tolerances):
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances
        for ts in self._timed_subgoals:
            if ts is not None:
                ts.goal = self.stage_dimension*0.5*ts.goal

    def render(self, mode='human', close=False):
        import pyglet
        import pyglet.gl as gl

        from .pyglet_utils import (draw_circle_sector, draw_box, draw_line, draw_vector, draw_vector_with_outline, 
                draw_circular_subgoal)

        if self.window is None:
            self.window = pyglet.window.Window(width = self.window_width,
                                               height = self.window_height,
                                               vsync = True,
                                               resizable = True)
            gl.glClearColor(*self.background_color)

        @self.window.event
        def on_resize(width, height):
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(-0.5*self.stage_dimension, 
                       0.5*self.stage_dimension, 
                       -0.5*float(height)/width*self.stage_dimension,
                       0.5*float(height)/width*self.stage_dimension, 
                       -1., 
                       1.)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            return pyglet.event.EVENT_HANDLED


        def draw_function_grid(values, a, b, color_low, color_high, value_low, value_high):
            diff = np.array(b) - np.array(a)
            diff[0] /= values.shape[0]
            diff[1] /= values.shape[1]
            v_diff = value_high - value_low
            for j in range(values.shape[1]):
                gl.glBegin(gl.GL_TRIANGLE_STRIP)
                for i in range(values.shape[0]):
                    value = np.clip(values[i, j], value_low, value_high)
                    v = (value - value_low)/v_diff
                    gl.glColor3f(*list(v*color_high + (1. - v)*color_low))
                    gl.glVertex2f(a[0] + diff[0]*i, a[1] + diff[1]*j)

                    value = np.clip(values[i, min(j + 1, values.shape[1] - 1)], value_low, value_high)
                    v = (value - value_low)/v_diff
                    gl.glColor3f(*list(v*color_high + (1. - v)*color_low))
                    gl.glVertex2f(a[0] + diff[0]*i, a[1] + diff[1]*(j + 1))
                gl.glVertex2f(a[0] + diff[0]*values.shape[0], a[1] + diff[1]*j)
                gl.glVertex2f(a[0] + diff[0]*values.shape[0], a[1] + diff[1]*(j + 1))
                gl.glEnd()


        def draw_timed_circular_subgoal(position, delta_t_ach, delta_t_comm, radius, color):
            draw_circular_subgoal(position, None, radius, color, None)
            # desired time until achievement
            draw_box(position + (0., radius + 0.05), delta_t_ach/100., 0.03, 0., color)
            # remaining commitment time
            draw_box(position + (0., radius + 0.02), delta_t_comm/100., 0.03, 0., (0., 0., 0.))


        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        gl.glLoadIdentity()

        n_triangles = 32

        if self.function_grid is not None:
            draw_function_grid(self.function_grid, -0.5*self.stage_dimension*np.ones(2), 
                    0.5*self.stage_dimension*np.ones(2), self.color_low, self.color_high, 
                    self.value_low, self.value_high)

        # obstacle
        draw_circle_sector([0., 0.], 
                0., 
                self.obstacle_radius,
                n_triangles, 
                self.obstacle_color,
                n_triangles)

        # subgoals
        for subgoal, color in zip(self._subgoals, self._subgoal_colors):
            draw_circle_sector(subgoal, 
                    0., 
                    self.subgoal_radius,
                    n_triangles, 
                    (0., 0., 0.),
                    n_triangles)
            draw_circle_sector(subgoal, 
                    0., 
                    0.8*self.subgoal_radius,
                    n_triangles, 
                    color,
                    n_triangles)

        # timed subgoals
        for ts, color, tol in zip(self._timed_subgoals, self._subgoal_colors, self._tolerances):
            if ts is not None:
                r = tol if tol is not None else self.subgoal_radius
                draw_timed_circular_subgoal(ts.goal, ts.delta_t_ach, 
                        ts.delta_t_comm, r, color)

        # goal
        draw_circle_sector(self.goal, 
                0., 
                self.goal_radius,
                n_triangles, 
                self.goal_color,
                n_triangles)
        draw_circle_sector(self.goal, 
                0., 
                self.goal_radius*0.8,
                n_triangles, 
                self.background_color[:3],
                n_triangles)

        # agent 
        draw_circle_sector(self.agent_position, 
                0., 
                self.agent_radius,
                n_triangles, 
                self.agent_color,
                n_triangles)


        self.window.flip()


class ObstacleEnvHAC(ObstacleEnv):
    """Version of obstalce environment that uses interface
    required by implementation of HAC by Andrew Levy."""

    def __init__(self):
        super().__init__()

        self.name = "ObstacleEnvHAC"

        self.max_actions = int(self.max_episode_length)
        self.visualize = False
        self.visualize_every_nth_episode = 10
        self.episode_counter = 0


        # Projection functions 
        self.project_state_to_end_goal = lambda s, s2: s
        self.project_state_to_subgoal = lambda s, s2: s

        # variables needed for HAC implementation
        self.state_dim = self.observation_space["observation"].low.shape[0]
        self.action_dim = self.action_space.low.shape[0] # low-level action dim
        self.action_bounds = np.ones(2) # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = self.observation_space["desired_goal"].low.shape[0]
        self.subgoal_dim = 2
        self.subgoal_bounds = np.array([
            [-1., 1.],
            [-1., 1.],
            ])
        self.subgoal_bounds_symmetric = np.array(
                [1., 1.])
        self.subgoal_bounds_offset = np.zeros((2))

        # End goal/subgoal thresholds
        self.subgoal_thresholds = np.ones(2)*(self.goal_radius/self.stage_dimension*2.)
        self.end_goal_thresholds = np.ones(2)*(self.goal_radius/self.stage_dimension*2.)


    def get_state(self):
        return self._get_obs()["observation"]

    def reset_sim(self):
        # save old goal
        old_goal = self.goal

        # reset (this also overwrites self._desired_goal)
        self.reset()

        # restore old self._desired_goal (in order to not mess up HAC implementation)
        # (Note: If this step is omitted, the goal in the HAC implementation and in 
        # the environment diverge and rendering and reward calculation in the environment
        # are off.)
        self.goal = old_goal

        self.episode_counter += 1

        # Return state
        return self.get_state()

    def render(self):
        super().render(self)

    def execute_action(self, action):
        obs, reward, done, info = self.step(action)
        self.done = done

        if self.visualize:
            if self.episode_counter % self.visualize_every_nth_episode == 0:
                self.render()

        # call the current state "sim" in order to trick the HAC implementation which 
        # expects an underlying Mujoco simulation
        self.sim = self.get_state()

        return self.sim

    def display_end_goal(self, end_goal):
        pass

    def get_next_goal(self, test):
        self._draw_goal()
        return self.goal/self.stage_dimension*2.0

    def display_subgoals(self, subgoals):
        self.update_subgoals(subgoals)

    def set_visualization(self, visualize, visualize_every_nth_episode):
        self.visualize = visualize
        self.visualize_every_nth_episode = visualize_every_nth_episode
