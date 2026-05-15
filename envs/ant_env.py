import random
import math
from pathlib import Path

import mujoco
import numpy as np
import scipy.spatial.transform as st
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

from config import OBS_SPACE_DIM, REWARD_SCALE

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    # theta = -np.pi/2 + 2*math.atan2(math.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1 - 2*(q[0]*q[2]-q[1]*q[3])))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


class HomeostaticAntEnv(AntEnv, EzPickle):
    def __init__(
        self,
        xml_file="ant_env.xml",
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        image_size=(64, 64),
        hunger_decay=0.00015,
        thirst_decay=0.00015,
        action_heat_gain_rate=0.00015 / 4,
        heat_source_gain_rate=0.0006,
        night_cooling_rate=0.0003,
        sweat_cooling_rate=0.00015,
        replenish_rate=0.1,
        day_night_cycle_len=1000,
        arena_size=6.0,
        posture_penalty_weight=0.005,
        num_food=4,
        num_water=4,
        num_heat=2,
        is_training=False,
        max_steps=40_000,
        render_mode="rgb_array",
        **kwargs,
    ):
        # Resolve absolute path for xml_file
        xml_file = str((Path(__file__).parent / xml_file).resolve())

        self.image_size = image_size
        self.is_training = is_training
        self.max_steps = max_steps

        # Homeostatic variables
        self.hunger = 0.0
        self.thirst = 0.0
        self.temperature = 0.0
        self.current_step = 0

        self.hunger_decay = hunger_decay
        self.thirst_decay = thirst_decay
        self.action_heat_gain_rate = action_heat_gain_rate
        self.heat_source_gain_rate = heat_source_gain_rate
        self.night_cooling_rate = night_cooling_rate
        self.sweat_cooling_rate = sweat_cooling_rate
        self.sweat_thirst_cost = 0.0  # FIXME: Remove for now
        self.replenish_rate = replenish_rate
        self.day_night_cycle_len = day_night_cycle_len
        self.arena_size = arena_size
        self.num_food = num_food
        self.num_water = num_water
        self.num_heat = num_heat
        self.posture_penalty_weight = posture_penalty_weight
        self.posture = 0.0

        self.food_consumed = 0
        self.water_consumed = 0
        self.heat_exposed_time = 0.0

        self.current_step = 0
        self.render_mode = render_mode

        # Check if heat should be added
        if self.num_heat == 0:
            print(
                "No heat sources defined. Removing heat dynamics. Ensure that XML file does not include heat bodies and that heat-related parameters are set to zero."
            )
            self.action_heat_gain_rate = 0.0
            self.heat_source_gain_rate = 0.0
            self.night_cooling_rate = 0.0
            self.sweat_cooling_rate = 0.0
            self.sweat_thirst_cost = 0.0

        # Initialize AntEnv
        # AntEnv v5 parameters
        AntEnv.__init__(
            self,
            xml_file=xml_file,
            default_camera_config=default_camera_config,
            width=image_size[0],
            height=image_size[1],
            render_mode=render_mode,
            **kwargs,
        )

        # Create action space: 8 for movement + 1 for sweating
        # The 8 movement have a range of (-1, 1)
        # Sweating is a binary action but uses a continuous space for simplicity. For the last action, more than zero means sweat, zero means no sweat.
        # Initialize the action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        # Override Observation Space to include vision and homeostatic states
        # Cannot use default observation space because we added resources in the XML file
        # Vision concatenates to RGBD but not transposing
        self.observation_space = spaces.Dict(
            {
                "proprioception": spaces.Box(
                    -np.inf, np.inf, (OBS_SPACE_DIM,), np.float32
                ),  # Proprioception - body location etc, excluding the resources
                "vision": spaces.Box(  # Vision with RGB and depth
                    low=-1.0,
                    high=1.0,
                    shape=(self.image_size[1], self.image_size[0], 4),
                    dtype=np.float32,
                ),
                "internal_state": spaces.Box(  # Internal variables
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "heat_sensor": spaces.Box(  # Heat sensor: [local_x, local_y]
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )

        EzPickle.__init__(
            self,
            xml_file,
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
            posture_penalty_weight,
            num_food,
            num_water,
            num_heat,
            is_training,
            render_mode,
            max_steps,
            **kwargs,
        )

        # Resource Object Management
        self.food_names = [f"food_{i}" for i in range(self.num_food)]
        self.water_names = [f"water_{i}" for i in range(self.num_water)]
        self.heat_names = [f"heat_{i}" for i in range(self.num_heat)]

        self.food_ids = self._get_body_ids(self.food_names)
        self.water_ids = self._get_body_ids(self.water_names)
        self.heat_ids = self._get_body_ids(self.heat_names)
        self.ant_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

    def _get_body_ids(self, names):
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx != -1:
                ids.append(idx)
        assert len(ids) == len(names), f"Some bodies not found for names: {names}"
        return ids

    def reset_model(self):
        # Override AntEnv.reset_model to handle homeostatic reset
        # This is called by AntEnv.reset
        if self.is_training:
            # During training, we can randomize the initial homeostatic state
            self.hunger = self.np_random.uniform(-(1 / 6), (1 / 6))
            self.thirst = self.np_random.uniform(-(1 / 6), (1 / 6))
            self.temperature = self.np_random.uniform(-(1 / 6), (1 / 6))
            # self.posture = 0.0  # Usually starts upright
        else:
            self.hunger = 0.0
            self.thirst = 0.0
            self.temperature = 0.0
            # self.posture = 0.0

        # Add if no heat
        if self.num_heat == 0:
            self.temperature = 0.0

        self.current_step = 0
        self.food_consumed = 0
        self.water_consumed = 0
        self.heat_exposed_time = 0.0

        # Standard Ant reset noise
        # self.init_qpos includes the x and y coordinates in the first 2 entries, different from observation space which does not
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        # Start at random positions in the arena, but not too close to the walls
        qpos[0] = self.np_random.uniform(-self.arena_size + 2, self.arena_size - 2)
        qpos[1] = self.np_random.uniform(-self.arena_size + 2, self.arena_size - 2)

        curr_w, curr_x, curr_y, curr_z = qpos[3:7]
        current_rot = st.Rotation.from_quat([curr_x, curr_y, curr_z, curr_w])
        random_yaw_angle = self.np_random.uniform(low=0, high=2 * np.pi)
        yaw_rot = st.Rotation.from_euler("z", random_yaw_angle)
        final_rot = yaw_rot * current_rot
        raw_quat = final_rot.as_quat()
        qpos[3:7] = [raw_quat[3], raw_quat[0], raw_quat[1], raw_quat[2]]

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        # Randomize resources
        for body_id in self.food_ids + self.water_ids + self.heat_ids:
            self._randomize_object_pos(body_id)

        # Initialize previous drive for the paper's reward formula
        self.prev_drive = self._calculate_drive()

        return self._get_obs()

    def _calculate_drive(self):
        
        # Posture dynamics - Deviation from upright
        # Using Euler angles (roll, pitch) to calculate tilt
        # data.qpos[3:7] is torso orientation (w, x, y, z)
        self.posture = np.square(qtoeuler(self.data.qpos[3:7])[:2] - qtoeuler([1.0, 0.0, 0.0, 0.0])[:2]).sum()
        # posture is the magnitude of roll and pitch deviation
        return self.hunger**2 + self.thirst**2 + self.temperature**2 + self.posture_penalty_weight * self.posture


    def _randomize_object_pos(self, body_id):
        if body_id == -1:
            return
        # Get ant position to avoid spawning resources too close to the ant
        pos_is_valid = False
        while not pos_is_valid:
            new_x = random.uniform(-self.arena_size + 0.75, self.arena_size - 0.75)
            new_y = random.uniform(-self.arena_size + 0.75, self.arena_size - 0.75)
            self.model.body_pos[body_id][:2] = [
                new_x,
                new_y,
            ]  # Use body_pos to set position, not qpos which is for the agent since the resources are static
            mujoco.mj_forward(
                self.model, self.data
            )  # Update the physics to reflect the new position before checking distances
            pos_is_valid = True
            for other_id in (
                self.food_ids + self.water_ids + self.heat_ids + [self.ant_body_id]
            ):
                if other_id == body_id:
                    continue
                dist = np.linalg.norm(
                    self.data.xpos[body_id][:2] - self.data.xpos[other_id][:2]
                )
                if dist < 2.0:  # Slightly larger buffer for 0.5 size objects
                    pos_is_valid = False
                    break

    def _get_heat_sensor_obs(self):
        """
        Detects the nearest heat source within 1.0m and returns the local direction.
        Returns [0, 0] if no heat source is within range.
        """
        ant_pos = self.data.xpos[self.ant_body_id][:2]
        nearest_heat_dist = 1.0
        nearest_heat_pos = None

        for body_id in self.heat_ids:
            heat_pos = self.data.xpos[body_id][:2]
            dist = np.linalg.norm(ant_pos - heat_pos)
            if dist < nearest_heat_dist:
                nearest_heat_dist = dist
                nearest_heat_pos = heat_pos

        if nearest_heat_pos is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Vector from Ant to Heat in world coordinates
        delta_world = nearest_heat_pos - ant_pos
        
        # Transform to local frame (torso's rotation)
        # xmat is a 9-element array (3x3 rotation matrix)
        rot_mat = self.data.xmat[self.ant_body_id].reshape(3, 3)
        # Local X (Forward) and Local Y (Left) projections
        local_x = np.dot(delta_world, rot_mat[:2, 0])
        local_y = np.dot(delta_world, rot_mat[:2, 1])
        
        # Normalize to unit vector for direction only
        direction = np.array([local_x, local_y], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            
        return direction

    def _get_obs(self):
        # This is the basic proprioceptive observation from AntEnv
        # OBS_SPACE_DIM shape about the body position, velocity, and joint angles
        # Only take the original proprioceptive part, not the resource positions we added in the XML file
        proprio_obs = AntEnv._get_obs(self)[:OBS_SPACE_DIM]

        # Render vision observations
        pov_image_rgb, pov_image_depth = self.mux_render(camera_name="pov")
        pov_image_rgb = 2.0 * (
            pov_image_rgb.astype(np.float32) / 255.0 - 0.5
        )  # Normalize RGB to [-1, 1]
        pov_image_depth = (
            2.0 * pov_image_depth.astype(np.float32) - 1.0
        )  # Normalize depth to [-1, 1]
        pov_image = np.concatenate(
            [pov_image_rgb, np.expand_dims(pov_image_depth, axis=-1)], axis=-1
        )

        return {
            "proprioception": proprio_obs,
            "vision": pov_image,
            "internal_state": np.array(
                [self.hunger, self.thirst, self.temperature],
                dtype=np.float32,  # internal variables
            ),
            "heat_sensor": self._get_heat_sensor_obs(),
        }

    def _add_hud(self, img):
        import cv2

        # Copy to avoid modifying the original if it's a view
        img = img.copy()

        # HUD settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Day/Night Status
        is_night = (self.current_step % self.day_night_cycle_len) >= (
            self.day_night_cycle_len / 2
        )
        time_text = "NIGHT" if is_night else "DAY"
        time_color = (0, 255, 255) if is_night else (255, 255, 0)

        stats = [
            (f"Hunger: {self.hunger:.2f}", (0, 255, 0)),  # Green
            (f"Thirst: {self.thirst:.2f}", (0, 0, 255)),  # Blue
            (f"Temp:   {self.temperature:.2f}", (255, 0, 0)),  # Red
            # (f"Posture:{self.posture:.2f}", (255, 255, 255)),  # White
            (f"Time:   {time_text}", time_color),
        ]

        for i, (text, color) in enumerate(stats):
            cv2.putText(img, text, (10, 20 + i * 20), font, scale, color, thickness)

        return img

    def mux_render(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return self.mujoco_renderer.render(render_mode="rgbd_tuple")

        old_cam_id = self.mujoco_renderer.camera_id
        self.mujoco_renderer.camera_id = cam_id

        # Adjust lighting based on day/night before rendering
        is_night = (self.current_step % self.day_night_cycle_len) >= (
            self.day_night_cycle_len / 2
        )
        if (
            self.num_heat > 0
        ):  # Only adjust lighting if heat sources are present, otherwise it can be confusing if the lighting changes but there are no heat dynamics
            if is_night:
                self.model.light_diffuse[0] = [0.1, 0.1, 0.1]  # Dim the light
            else:
                # This is tricky because we don't want to keep multiplying by 0.2
                # Let's use a fixed value. Default is usually around [0.8, 0.8, 0.8]
                self.model.light_diffuse[0] = [0.9, 0.9, 0.9]

        # Note that rgbd_tuple returns (rgb, depth) which is given by:
        # RGB: uint8 array of shape (height, width, 3) - this has a range of (0, 255) for each channel
        # Depth: float32 array of shape (height, width) with depth in meters - this has a range of (0, 1)
        # Uses Mujoco's depth rendering that it calculates on its own. Closer objects have smaller depth values
        img = self.mujoco_renderer.render(render_mode="rgbd_tuple")

        self.mujoco_renderer.camera_id = old_cam_id
        return img

    def step(self, action):
        physical_action = action[:8]
        sweat_action = action[8]

        # Apply physical action and simulate
        self.do_simulation(physical_action, self.frame_skip)
        self.current_step += 1

        # Homeostatic Dynamics
        is_night = (self.current_step % self.day_night_cycle_len) >= (
            self.day_night_cycle_len / 2
        )

        # Resource and Heat Contact Detection
        contact_heat = 0
        respawned_bodies = set()
        ant_pos = self.data.xpos[self.ant_body_id][:2]

        # Distance-based detection for all resources (Food, Water, Heat)
        # Check Food
        for body_id in self.food_ids:
            if body_id not in respawned_bodies:
                food_pos = self.data.xpos[body_id][:2]
                if (
                    np.linalg.norm(ant_pos - food_pos) < 1.0
                ):  #  and self._is_in_front(food_pos)
                    self.hunger += self.replenish_rate
                    self._randomize_object_pos(body_id)
                    respawned_bodies.add(body_id)
                    self.food_consumed += 1

        # Check Water
        for body_id in self.water_ids:
            if body_id not in respawned_bodies:
                water_pos = self.data.xpos[body_id][:2]
                if (
                    np.linalg.norm(ant_pos - water_pos) < 1.0
                ):  #  and self._is_in_front(water_pos):
                    self.thirst += self.replenish_rate
                    self._randomize_object_pos(body_id)
                    respawned_bodies.add(body_id)
                    self.water_consumed += 1

        # Pass-through detection (Heat)
        # Can get heated by multiple sources
        for heat_body_id in self.heat_ids:
            heat_pos = self.data.xpos[heat_body_id][:2]
            if (
                np.linalg.norm(ant_pos - heat_pos) < 1.0
            ):  #  and self._is_in_front(heat_pos):
                contact_heat += 1

        # Passive decay/gain
        self.hunger -= self.hunger_decay
        self.thirst -= self.thirst_decay

        # Temperature dynamics
        action_magnitude = np.linalg.norm(physical_action)
        self.temperature += action_magnitude**2 * self.action_heat_gain_rate  # quadratic so small movements gain less heat than moving wildly
        if contact_heat:
            self.temperature += self.heat_source_gain_rate * contact_heat
            self.heat_exposed_time += 1.0 * contact_heat

        # Update sweat visualization (lingering effect for HUD)
        if sweat_action > 0.0:
            self.temperature -= (sweat_action * self.sweat_cooling_rate)
            # self.thirst -= self.sweat_thirst_cost
            self.model.geom_rgba[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
            ] = [0.2, 0.6, 1.0, 1.0]  # Change color to indicate sweating
        else:
            self.model.geom_rgba[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
            ] = [0.8, 0.6, 0.4, 1.0]  # Default color
        if is_night:
            self.temperature -= self.night_cooling_rate

        # Check if agent has flipped
        up_vector_z = self.data.xmat[self.ant_body_id][8]
        z_pos = self.data.xpos[self.ant_body_id][2]

        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
            or abs(self.temperature) > 0.99999
        )
        is_flipped = up_vector_z < 0.5
        is_height_invalid = z_pos < 0.2 or z_pos > 1.0

        term_reason = 0  # Max episode...?
        if limit_reached:
            term_reason = 1  # homeostatic
        elif is_flipped:
            term_reason = 2  # flipped - just die immediately
            # self.hunger = -2
            # self.thirst = -2
            # self.temperature = -2
        elif is_height_invalid:
            term_reason = 3  # height
            # self.hunger = -2
            # self.thirst = -2
            # self.temperature = -2

        # Clipping state variables
        self.hunger = np.clip(self.hunger, -1.0, 1.0)
        self.thirst = np.clip(self.thirst, -1.0, 1.0)
        self.temperature = np.clip(self.temperature, -1.0, 1.0)

        # Homeostatic Reward (Paper: Reduction in Drive)
        current_drive = self._calculate_drive()

        reward = REWARD_SCALE * (self.prev_drive - current_drive)
        self.prev_drive = current_drive

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        info = {
            "time": {"timestep": self.current_step, "is_night": is_night},
            "internal_state": {
                "hunger": self.hunger,
                "thirst": self.thirst,
                "temperature": self.temperature,
            },
            "resources_consumed": {
                "food": self.food_consumed,
                "water": self.water_consumed,
                "heat_exposure_time": self.heat_exposed_time,
                "sweating": 1 if sweat_action > 0.0 else 0,
            },
            "stability": {
                "up_vector_z": up_vector_z,
                "z_pos": z_pos,
                "termination_reason": term_reason,
                "posture": self.posture,
            },
        }

        # Return the environment image in info for vieweing
        # Add visual HUD to the environment image for debugging/monitoring
        # Dont need depth for viewing
        # Only do it for debugging
        if not self.is_training:
            env_image_rgb, _ = self.mux_render(camera_name="environment")
            env_image_rgb = self._add_hud(env_image_rgb)

            # Also return POV in infor for recording and viewing
            # Current vision - dont have any normalize or frame stack etc.
            pov_image_rgb, _ = self.mux_render(camera_name="pov")

            info["vision"] = pov_image_rgb
            info["environment"] = env_image_rgb

        return obs, reward, self.terminated, self.truncated, info

    @property
    def terminated(self):
        # Homeostatic limits check (+/- 1.0)
        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
            or abs(self.temperature) > 0.99999
        )

        # Orientation check (True Flip Check)
        # xmat[8] is the world-Z component of the torso's local Z-axis (Up)
        # 1.0 = upright, 0.0 = on side, -1.0 = upside down
        # up_vector_z = self.data.xmat[self.ant_body_id][8]
        # is_flipped = up_vector_z < 0.5  # Tilted more than 60 degrees

        # # Height check
        # z_pos = self.data.xpos[self.ant_body_id][2]
        # is_too_low = z_pos < 0.2 or z_pos > 1.0

        return bool(limit_reached) #  or is_flipped or is_too_low

    @property
    def truncated(self):
        return self.current_step >= self.max_steps

    def _is_in_front(self, target_pos, fov_threshold=0.5):
        """
        Checks if target_pos is within the agent's forward-facing cone.
        0.5 corresponds to a +/- 60 degree FOV (total 120 degrees).
        """
        # 1. Get Ant's current position and rotation matrix
        ant_pos = self.data.xpos[self.ant_body_id][:2]
        # In MuJoCo, the first column of the xmat (rotation matrix) is the local X-axis (Forward)
        forward_vec = self.data.xmat[self.ant_body_id].reshape(3, 3)[:, 0]

        # 2. Vector from Ant to Resource (ignore Z for a flat arena check)
        target_vec = target_pos - ant_pos

        # 3. Normalize the target vector
        dist = np.linalg.norm(target_vec)
        if dist < 1e-6:
            return True  # If touching, count as in front
        target_vec /= dist

        # 4. Dot product check
        dot_product = np.dot(forward_vec[:2], target_vec[:2])

        return dot_product > fov_threshold
