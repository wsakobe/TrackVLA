from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import attr
import numpy as np

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Episode

from habitat.core.logging import logger
from habitat.core.registry import registry
from hydra.core.config_store import ConfigStore
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
import habitat_sim
from dataclasses import dataclass
from habitat.config.default_structured_configs import MeasurementConfig

from habitat.tasks.rearrange.utils import rearrange_collision
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.core.embodied_task import Measure
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.utils import coll_name_matches
try:
    import magnum as mn
except ImportError:
    pass

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

cv2 = try_cv2_import()

MAP_THICKNESS_SCALAR: int = 128


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_measure
class DidMultiAgentsCollide(Measure):
    """
    Detects if the multi-agent ( more than 1 humanoids agents) in the scene 
    are colliding with each other at the current step. 
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_multi_agents_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        found_contact = False

        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in sim.agents_mgr.articulated_agents_iter
        ]
        main_human_id = agent_ids[0]
        robot_id = agent_ids[1]
        for cp in contact_points:
            if coll_name_matches(cp, main_human_id):
                if coll_name_matches(cp, robot_id):
                    found_contact = True
                    break  

        self._metric = found_contact


@registry.register_measure
class HumanCollision(Measure):

    cls_uuid: str = "human_collision"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._ever_collide = False
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToLeader.cls_uuid]
        )

        self._metric = 0.0
        self._ever_collide = False

    def update_metric(self, *args, episode, task, observations, **kwargs):
        collid = task.measurements.measures[
            DistanceToLeader.cls_uuid
        ].get_metric()
        if collid < 0.5 or self._ever_collide:
            self._metric = 1.0
            self._ever_collide = True
            # task.should_end = True
        else:
            self._metric = 0.0


@registry.register_measure
class DistanceToLeader(Measure):
    """The measure calculates a distance towards the leader human."""

    cls_uuid: str = "distance_to_leader"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_data(1).articulated_agent.base_pos
        main_human_id = 0
        human_pos = self._sim.get_agent_data(
            main_human_id
        ).articulated_agent.base_pos
        
        distance_to_target = np.linalg.norm(current_position - human_pos)
 
        self._metric = distance_to_target


@registry.register_measure
class HumanDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "human_distance_to_goal"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.num_goals = episode.info["num_goals_for_main_human"]
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        for goal in episode.goals:
            self.goals = [np.array(pos, dtype=np.float32) for pos in goal.position]
        
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_data(0).articulated_agent.base_pos
        
        path = habitat_sim.ShortestPath()
        path.requested_start = current_position
        path.requested_end = self.goals[-1]
        found_path = self._sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance

        # distance_to_target = self._sim.geodesic_distance(
        #     current_position,
        #     self.goals[-1],
        #     episode,
        # )
 
        self._metric = geodesic_distance


@registry.register_measure
class HumanFollowing(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "human_following"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        self._sim = sim
        self._config = config
        self._success_distance = self._config.success_distance

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _reach_agent(self, task, obs):
        """Check if the agent reaches the target agent or not"""
  
        facing = obs.get("agent_1_main_humanoid_detector_sensor")

        distance_to_target = task.measurements.measures[
            DistanceToLeader.cls_uuid
        ].get_metric()
        
        if  distance_to_target <= self._success_distance and facing["facing"]:
            return True
        
        return False

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToLeader.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if (self._reach_agent(task, observations)):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class HumanFollowingSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "human_following_success"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        self._sim = sim
        self._config = config
        self._success_following_distance_lower = self._config.success_following_distance_lower
        self._success_following_distance_upper = self._config.success_following_distance_upper

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _reach_agent(self, task, obs):
        """Check if the agent reaches the target agent or not"""

        facing = False
        facing = task.measurements.measures[
            HumanFollowing.cls_uuid
        ].get_metric()

        distance_to_target = task.measurements.measures[
            DistanceToLeader.cls_uuid
        ].get_metric()
        
        if distance_to_target <= self._success_following_distance_upper and distance_to_target >= self._success_following_distance_lower and facing:
            return True
        
        return False

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToLeader.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._reach_agent(task, observations)
        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and self._reach_agent(task, observations)
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class TopDownMapFollowing(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: List[Optional[Tuple[int, int]]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map_following"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                self.goals = [np.array(pos, dtype=np.float32) for pos in goal.position]
                for each_goal in self.goals:
                    if self._is_on_same_floor(each_goal[1]):
                        try:
                            if goal.view_points is not None:
                                for view_point in goal.view_points:
                                    self._draw_point(
                                        view_point.agent_state.position,
                                        maps.MAP_VIEW_POINT_INDICATOR,
                                    )
                        except AttributeError:
                            pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:
            for goal in episode.goals:
                self.goals = [np.array(pos, dtype=np.float32) for pos in goal.position]
                for each_goal in self.goals:
                    if self._is_on_same_floor(each_goal[1]):
                        try:
                            self._draw_point(
                                each_goal, maps.MAP_TARGET_POINT_INDICATOR
                            )
                        except AttributeError:
                            pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            for goal in episode.goals:
                self.goals = [np.array(pos, dtype=np.float32) for pos in goal.position]
                for each_goal in self.goals:
                    _shortest_path_points = (
                        self._sim.get_straight_shortest_path_points(
                            agent_position, each_goal
                        )
                    )
                    self._shortest_path_points = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                            sim=self._sim,
                        )
                        for p in _shortest_path_points
                    ]
                    maps.draw_path(
                        self._top_down_map,
                        self._shortest_path_points,
                        maps.MAP_SHORTEST_PATH_COLOR,
                        self.line_thickness,
                    )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._top_down_map = self.get_original_map()
        self._step_count = 0
        agent_position = self._sim.get_agent_state().position
        self._previous_xy_location = [
            None for _ in range(episode.info['human_num'] + 2)
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            # self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

        self.update_metric(episode, None)
        self._step_count = 0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        map_positions: List[Tuple[float]] = []
        map_angles = []
        for agent_index in range(2):
            agent_state = self._sim.get_agent_state(agent_index)
            map_positions.append(self.update_map(agent_state, agent_index))
            map_angles.append(TopDownMapFollowing.get_polar_angle(agent_state, agent_index))
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
        }

    @staticmethod
    def get_polar_angle(agent_state, agent_index):
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]

        if agent_index == 1:
            return np.array(phi)
        else:
            return np.array(phi) - np.pi / 2

    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.max_episode_steps, 245
            )

            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = TopDownMapFollowing.get_polar_angle(agent_state, agent_index)
 
        self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_fog_of_war_mask(self, agent_position, angle):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                angle,
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )


@dataclass
class DidMultiAgentsCollideConfig(MeasurementConfig):
    type: str = "DidMultiAgentsCollide"
    
@dataclass
class HumanCollisionMeasurementConfig(MeasurementConfig):
    type: str = "HumanCollision"

@dataclass
class DistanceToLeaderConfig(MeasurementConfig):
    type: str = "DistanceToLeader"

@dataclass
class HumanDistanceToGoalConfig(MeasurementConfig):
    type: str = "HumanDistanceToGoal"

@dataclass
class HumanFollowingConfig(MeasurementConfig):
    type: str = "HumanFollowing"
    success_distance: float = 3.0

@dataclass
class HumanFollowingSuccessConfig(MeasurementConfig):
    type: str = "HumanFollowingSuccess"
    success_following_distance_lower: float = 1.0
    success_following_distance_upper: float = 3.0
    max_episode_steps: int = 300

@dataclass
class FogOfWarConfig:
    draw: bool = True
    visibility_dist: float = 5.0
    fov: int = 90

@dataclass
class TopDownMapFollowingConfig(MeasurementConfig):
    type: str = "TopDownMapFollowing"
    max_episode_steps: int = 300
    map_padding: int = 3
    map_resolution: int = 1024
    draw_source: bool = True
    draw_border: bool = True
    draw_shortest_path: bool = True
    draw_view_points: bool = True
    draw_goal_positions: bool = True
    # axes aligned bounding boxes
    draw_goal_aabbs: bool = True
    fog_of_war: FogOfWarConfig = FogOfWarConfig()


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.measurements.human_collision",
    group="habitat/task/measurements",
    name="human_collision",
    node=HumanCollisionMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_multi_agents_collide",
    group="habitat/task/measurements",
    name="did_multi_agents_collide",
    node=DidMultiAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.distance_to_leader",
    group="habitat/task/measurements",
    name="distance_to_leader",
    node=DistanceToLeaderConfig,
)
cs.store(
    package="habitat.task.measurements.human_distance_to_goal",
    group="habitat/task/measurements",
    name="human_distance_to_goal",
    node=HumanDistanceToGoalConfig,
)
cs.store(
    package="habitat.task.measurements.human_following",
    group="habitat/task/measurements",
    name="human_following",
    node=HumanFollowingConfig,
)
cs.store(
    package="habitat.task.measurements.human_following_success",
    group="habitat/task/measurements",
    name="human_following_success",
    node=HumanFollowingSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.top_down_map_following",
    group="habitat/task/measurements",
    name="top_down_map_following",
    node=TopDownMapFollowingConfig,
)
