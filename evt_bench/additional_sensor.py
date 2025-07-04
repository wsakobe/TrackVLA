#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from gym import spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import PointGoalSensor
from hydra.core.config_store import ConfigStore
import habitat_sim

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@dataclass
class MainHumanoidDetectorSensorConfig(LabSensorConfig):
    human_id: int = 0
    human_pixel_threshold: int = 60000
    return_image: bool = False
    is_return_image_bbox: bool = True

    type: str = "MainHumanoidDetectorSensor"

@dataclass
class OtherHumanoidDetectorSensorConfig(LabSensorConfig):
    human_id: int = 0
    human_pixel_threshold: int = 60000
    return_image: bool = False
    is_return_image_bbox: bool = True

    type: str = "OtherHumanoidDetectorSensor"


@registry.register_sensor
class MainHumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox
        self._first_init = True

        # Check the observation size
        jaw_panoptic_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_jaw_panoptic" in key:
                jaw_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if jaw_panoptic_shape is not None:
            self._height = jaw_panoptic_shape[0]
            self._width = jaw_panoptic_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "main_humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
        
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Dict({
            "facing": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "box": spaces.Box(
                shape=(4,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            )
        })

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        if self._first_init:
            self._human_id = episode.info["main_human_semantic_id"]
            self._first_init = False
        use_k = f"agent_{self.agent_id}_articulated_agent_jaw_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            if self._return_image:
                return{
                    "facing": np.zeros((1,), dtype=np.float32),
                    "box": np.zeros((4,), dtype=np.float32),
                    "mask": np.zeros(
                        (self._height, self._width, 1), dtype=np.float32
                    )
                } 
            else:
                return {
                    "facing": np.zeros((1,), dtype=np.float32),
                    "box": np.zeros((4,), dtype=np.float32),
                }
        result = {
            "facing": np.zeros((1,), dtype=np.float32),
            "box": np.zeros((4,), dtype=np.float32),
            "mask": panoptic
        }

        tgt_mask = np.float32(panoptic == self._human_id)
        
        if self._is_return_image_bbox and np.sum(tgt_mask) > 0:
            rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
            result["box"] = np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

        human_pixel_count = np.sum(panoptic == self._human_id)
        if (
            self._human_pixel_threshold < human_pixel_count < self._height * self._width * 0.3
        ):
            result["facing"] = np.ones((1,), dtype=np.float32)
        
        return result
    
@registry.register_sensor
class OtherHumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._all_human_ids = list(range(1000, 1101))
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox

        # Check the observation size
        jaw_panoptic_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_jaw_panoptic" in key:
                jaw_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if jaw_panoptic_shape is not None:
            self._height = jaw_panoptic_shape[0]
            self._width = jaw_panoptic_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "other_humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
        
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Dict({
            "facing": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "block": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "box": spaces.Box(
                shape=(4,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            )
        })

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        self._human_id = episode.info["main_human_semantic_id"]
        use_k = f"agent_{self.agent_id}_articulated_agent_jaw_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            return {
                "facing": np.zeros((1,), dtype=np.float32),
                "block": np.zeros((1,), dtype=np.float32),
                "box": np.zeros((4,), dtype=np.float32)
            }
        result = {
            "facing": np.zeros((1,), dtype=np.float32),
            "block": np.zeros((1,), dtype=np.float32),
            "box": np.zeros((4,), dtype=np.float32)
        }

        for other_human_id in self._all_human_ids:
            if other_human_id == self._human_id:
                continue
            tgt_mask = np.float32(panoptic == other_human_id)
            
            if self._is_return_image_bbox and np.sum(tgt_mask) > 0:
                rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
                result["box"] = np.array([rmin, cmin, rmax, cmax], dtype=np.float32)

            human_pixel_count = np.sum(panoptic == other_human_id)
            if (
                human_pixel_count > self._human_pixel_threshold
            ):
                result["facing"] = np.ones((1,), dtype=np.float32)
            if (
                human_pixel_count > self._height * self._width * 0.2
            ):
                result["block"] = np.ones((1,), dtype=np.float32)
        
        return result


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.main_humanoid_detector_sensor",
    group="habitat/task/lab_sensors",
    name="main_humanoid_detector_sensor",
    node=MainHumanoidDetectorSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.other_humanoid_detector_sensor",
    group="habitat/task/lab_sensors",
    name="other_humanoid_detector_sensor",
    node=OtherHumanoidDetectorSensorConfig,
)
