import os
import random

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HomeostaticVisionAntEnv(AntEnv, EzPickle):
    def __init__(
        self,
        xml_file="ant_vision.xml",
        frame_skip=4,  # Standard based on atari
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        image_size=(64, 64),
        hunger_decay=0.00015,
        thirst_decay=0.00015,
        action_heat_gain_rate=2e-5,
        heat_source_gain_rate=0.001,
        night_cooling_rate=0.001,
        sweat_cooling_rate=0.0005,
        replenish_rate=0.1,
        day_night_cycle_len=2_000,
        arena_size=15.0,
        num_food=5,
        num_water=5,
        num_heat=3,
        **kwargs,
    ):
        # Resolve absolute path for xml_file
        if not os.path.isabs(xml_file):
            xml_file = os.path.join(os.getcwd(), xml_file)

        self.image_size = image_size

        # Homeostatic variables
        self.hunger = 0.0
        self.thirst = 0.0
        self.temperature = 0.0
        self.current_step = 0
        self.sweat_ind = 0.0  # For HUD visualization of sweating

        self.hunger_decay = hunger_decay
        self.thirst_decay = thirst_decay
        self.action_heat_gain_rate = action_heat_gain_rate
        self.heat_source_gain_rate = heat_source_gain_rate
        self.night_cooling_rate = night_cooling_rate
        self.sweat_cooling_rate = sweat_cooling_rate
        self.replenish_rate = replenish_rate
        self.day_night_cycle_len = day_night_cycle_len
        self.arena_size = arena_size
        self.num_food = num_food
        self.num_water = num_water
        self.num_heat = num_heat

        # Initialize AntEnv
        # AntEnv v5 parameters
        AntEnv.__init__(
            self,
            xml_file=xml_file,
            frame_skip=frame_skip,  # Frame step is 0.01s
            default_camera_config=default_camera_config,
            width=image_size[0],
            height=image_size[1],
            **kwargs,
        )

        # Create action space: 8 for movement + 1 for sweating
        # The 8 movement have a range of (-1, 1)
        # Sweating is a binary action but uses a continuous space for simplicity. For the last action, more than zero means sweat, zero means no sweat.
        # Initialize the action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        # Override Observation Space to include vision and homeostatic states
        proprio_space = self.observation_space
        self.observation_space = spaces.Dict(
            {
                "proprioception": proprio_space,  # Proprioception - body location etc
                "vision": spaces.Box(  # Vision with RGB and depth
                    low=0,
                    high=255,
                    shape=(self.image_size[1], self.image_size[0], 4),
                    dtype=np.uint8,
                ),
                "environment": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_size[1], self.image_size[0], 4),
                    dtype=np.uint8,
                ),
                "internal_state": spaces.Box(  # Internal variables
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
            }
        )

        EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            image_size,
            hunger_decay,
            thirst_decay,
            action_heat_gain_rate,
            heat_source_gain_rate,
            night_cooling_rate,
            sweat_cooling_rate,
            replenish_rate,
            day_night_cycle_len,
            arena_size,
            num_food,
            num_water,
            num_heat,
            **kwargs,
        )

        # Resource Object Management
        self.food_names = [f"food_{i}" for i in range(self.num_food)]
        self.water_names = [f"water_{i}" for i in range(self.num_water)]
        self.heat_names = [f"heat_{i}" for i in range(self.num_heat)]

        self.food_ids = self._get_body_ids(self.food_names)
        self.water_ids = self._get_body_ids(self.water_names)
        self.heat_ids = self._get_body_ids(self.heat_names)
        self.ant_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    def _get_body_ids(self, names):
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx != -1:
                ids.append(idx)
        assert len(ids) == len(names), f"Some bodies not found for names: {names}"
        return ids

    def reset_model(self, is_training=False):
        # Override AntEnv.reset_model to handle homeostatic reset
        # This is called by AntEnv.reset
        if is_training:
            # During training, we can randomize the initial homeostatic state
            self.hunger = self.np_random.uniform(-(1/6), (1/6))
            self.thirst = self.np_random.uniform(-(1/6), (1/6))
            self.temperature = self.np_random.uniform(-(1/6), (1/6))
        else:
            self.hunger = 0.0
            self.thirst = 0.0
            self.temperature = 0.0
        self.current_step = 0
        self.sweat_ind = 0.0  # Uncomment if sweat visualization is needed

        # Standard Ant reset noise
        # self.init_qpos includes the x and y coordinates in the first 2 entries, different from observation space which does not
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        # Start at random positions in the arena, but not too close to the walls
        qpos[0] = self.np_random.uniform(-self.arena_size + 1, self.arena_size - 1)
        qpos[1] = self.np_random.uniform(-self.arena_size + 1, self.arena_size - 1)

        qvel = self.init_qvel  # Start with zero velocity
        self.set_state(qpos, qvel)

        # Randomize resources
        for body_id in self.food_ids + self.water_ids + self.heat_ids:
            self._randomize_object_pos(body_id)

        return self._get_obs()

    def _randomize_object_pos(self, body_id):
        if body_id == -1:
            return
        # Keep objects within arena bounds (accounting for their size of around 0.5)
        new_x = random.uniform(-self.arena_size + 0.75, self.arena_size - 0.75)
        new_y = random.uniform(-self.arena_size + 0.75, self.arena_size - 0.75)
        self.model.body_pos[body_id][:2] = [new_x, new_y]  # Use body_pos to set position, not qpos which is for the agent since the resources are static
        # pov_image, _ = self.mux_render(camera_name="pov")

    def _get_obs(self):
        # This is the basic proprioceptive observation from AntEnv
        # 105 shape about the body position, velocity, and joint angles
        proprio_obs = AntEnv._get_obs(self)

        # Render vision observations
        pov_image = self.mux_render(camera_name="pov")
        env_image_rgb, env_image_depth = self.mux_render(camera_name="environment")

        # Add visual HUD to the environment image for debugging/monitoring
        env_image_rgb = self._add_hud(env_image_rgb)

        # # Environmental state (Day/Night phase)
        # phase = (
        #     self.current_step % self.day_night_cycle_len
        # ) / self.day_night_cycle_len

        return {
            "proprioception": proprio_obs,
            "vision": pov_image,
            "environment":  (env_image_rgb, env_image_depth),
            "internal_state": np.array(
                [self.hunger, self.thirst, self.temperature], dtype=np.float32  # internal variables
            ),
        }

    def _add_hud(self, img):
        import cv2

        # Copy to avoid modifying the original if it's a view
        img = img.copy()
        h, w, _ = img.shape

        # HUD settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Day/Night Status
        is_night = (self.current_step % self.day_night_cycle_len) > (
            self.day_night_cycle_len / 2
        )
        time_text = "NIGHT" if is_night else "DAY"
        time_color = (
            (0, 255, 255) if is_night else (255, 255, 0)
        )

        stats = [
            (f"Hunger: {self.hunger:.2f}", (0, 255, 0)),  # Green
            (f"Thirst: {self.thirst:.2f}", (255, 0, 0)),  # Blue
            (f"Temp:   {self.temperature:.2f}", (0, 0, 255)),  # Red
            (f"Time:   {time_text}", time_color),
        ]

        for i, (text, color) in enumerate(stats):
            cv2.putText(img, text, (10, 20 + i * 20), font, scale, color, thickness)

        # Neon Sweat Indicator
        if self.sweat_ind > 0.0:
            # "Neon" effect: bright cyan text with a slight offset "glow" if intensity is high
            sweat_color = (255, 255, 0)
            sweat_text = "SWEATING"

            # Draw glow (thicker, same color but maybe slightly different position or just thicker)
            cv2.putText(img, sweat_text, (10, 20 + len(stats) * 20), font, scale, sweat_color, thickness + 2)
            # # Draw main text
            # cv2.putText(img, sweat_text, (10, 20 + len(stats) * 20), font, scale, (255, 255, 255), thickness)

        return img

    def mux_render(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return self.mujoco_renderer.render(render_mode="rgbd_tuple")

        old_cam_id = self.mujoco_renderer.camera_id
        self.mujoco_renderer.camera_id = cam_id

        # Adjust lighting based on day/night before rendering
        is_night = (self.current_step % self.day_night_cycle_len) > (
            self.day_night_cycle_len / 2
        )
        if is_night:
            self.model.light_diffuse[0] = [0.1, 0.1, 0.1]  # Dim the light
        else:
            # This is tricky because we don't want to keep multiplying by 0.2
            # Let's use a fixed value. Default is usually around [0.8, 0.8, 0.8]
            self.model.light_diffuse[0] = [0.9, 0.9, 0.9]

        img = self.mujoco_renderer.render(render_mode="rgbd_tuple")
        
        self.mujoco_renderer.camera_id = old_cam_id
        return img

    def step(self, action):
        physical_action = action[:8]
        sweat_action = action[8]

        # Apply physical action and simulate
        self.do_simulation(physical_action, self.frame_skip)
        self.current_step += (1 * self.frame_skip)

        # Homeostatic Dynamics
        is_night = (self.current_step % self.day_night_cycle_len) > (
            self.day_night_cycle_len / 2
        )

        # Resource and Heat Contact Detection
        contact_heat = 0
        respawned_bodies = set()
        ant_pos = self.data.xpos[self.ant_body_id]

        # Distance-based detection for all resources (Food, Water, Heat)
        # Check Food
        for body_id in self.food_ids:
            if body_id not in respawned_bodies:
                food_pos = self.data.xpos[body_id]
                if np.linalg.norm(ant_pos - food_pos) < 1.0:
                    self.hunger += self.replenish_rate
                    self._randomize_object_pos(body_id)
                    respawned_bodies.add(body_id)
        
        # Check Water
        for body_id in self.water_ids:
            if body_id not in respawned_bodies:
                water_pos = self.data.xpos[body_id]
                if np.linalg.norm(ant_pos - water_pos) < 1.0:
                    self.thirst += self.replenish_rate
                    self._randomize_object_pos(body_id)
                    respawned_bodies.add(body_id)

        # Pass-through detection (Heat)
        # Can get heated by multiple sources
        for heat_body_id in self.heat_ids:
            heat_pos = self.data.xpos[heat_body_id]
            if np.linalg.norm(ant_pos - heat_pos) < 1.0:
                contact_heat += 1

        # Passive decay/gain
        self.hunger -= (self.hunger_decay * self.frame_skip)
        self.thirst -= (self.thirst_decay * self.frame_skip)

        # Temperature dynamics
        action_magnitude = np.linalg.norm(physical_action)
        self.temperature += (action_magnitude * self.action_heat_gain_rate * self.frame_skip)
        if contact_heat:
            self.temperature += self.heat_source_gain_rate * contact_heat

        # Update sweat visualization (lingering effect for HUD)
        # Sweat is binary
        self.sweat_ind = sweat_action

        if sweat_action > 0.0:
            self.temperature -= self.sweat_cooling_rate
        if is_night:
            self.temperature -= self.night_cooling_rate

        # Clipping state variables
        self.hunger = np.clip(self.hunger, -1.0, 1.0)
        self.thirst = np.clip(self.thirst, -1.0, 1.0)
        self.temperature = np.clip(self.temperature, -1.0, 1.0)

        # Pure Homeostatic Reward
        reward = -(self.hunger**2 + self.thirst**2 + self.temperature**2)

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, self.terminated, False, {}

    @property
    def terminated(self):
        # Check if agent has flipped over
        z_pos = self.data.qpos[2]
        is_healthy = 0.27 <= z_pos <= 1.5

        # Homeostatic limits check (+/- 1.0)
        limit_reached = (
            abs(self.hunger) > 0.99
            or abs(self.thirst) > 0.99
            or abs(self.temperature) > 0.99
        )
        
        if not is_healthy:
            print(f"Not healthy at step {self.current_step}: z_pos={z_pos:.2f}")
        
        if limit_reached:
            print(f"Homeostatic limit reached at step {self.current_step}:")
            print(f"Hunger: {self.hunger:.2f}, Thirst: {self.thirst:.2f}, Temp: {self.temperature:.2f}")

        return (not is_healthy) or limit_reached  
