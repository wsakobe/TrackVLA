import habitat
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import trange

import os
import os.path as osp
import imageio
import json


def evaluate_agent(config, dataset_split, save_path) -> None:
    # robot definition
    robot_config = GTBBoxAgent(save_path)
    
    first_init = True
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            if first_init:
                instruction = env.current_episode.info['instruction']
                first_init = False

            action_dict = dict()
            finished = False
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            human_no_move = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()

            while not env.episode_over:
                record_info = {}
                
                obs = sim.get_sensor_observations()

                detector = env.task._get_observations(env.current_episode)
                action = robot_config.act(obs, detector, env.current_episode.episode_id)

                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {
                        "agent_1_base_vel" : action
                    }
                }
                
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0
                else:
                    print("Lost")

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
                
                print(f"========== ID: {env.current_episode.episode_id} Step now is: {iter_step} action is: {action} dis_to_main_human: {np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)} ============")

            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()
            robot_config.reset(env.current_episode)

            if env.episode_over:
                finished = True
            
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2) 
            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2) 


class GTBBoxAgent(AgentConfig):
    def __init__(self, result_path):
        super().__init__()
        print("Initialize gtbbox agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        
        self.rgb_list = []
        self.rgb_box_list = []

        self.kp_t = 2
        self.kd_t = 0
        self.kp_f = 1
        self.kd_f = 0

        self.prev_error_t = 0
        self.prev_error_f = 0

        self.first_inside = True

        self.reset()

    def reset(self, episode: NavigationEpisode = None):
        if len(self.rgb_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)

            print(f"Successfully save the episode video with episode id {episode.episode_id}")

            self.rgb_list = []
        
        self.first_inside = True

    def act(self, observations, detector, episode_id):
        self.episode_id = episode_id
        
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        image = np.asarray(rgb_[:, :, ::-1])
        height, width = image.shape[:2]
        
        if detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            best_box =  np.array([(box[0]+box[2])/(2*width), (box[1]+box[3])/(2*height), (box[2]-box[0])/width, (box[3]-box[1])/height], dtype=np.float32)
            
            center_x = best_box[0]

            error_t = 0.5 - center_x
            error_f = (30000 - (box[2]-box[0])*(box[3]-box[1])) / 10000
            if abs(error_f) < 0.5:
                error_f = 0

            derivative_t = error_t - self.prev_error_t
            derivative_f = error_f - self.prev_error_f

            yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
            move_speed = self.kp_f * error_f + self.kd_f * derivative_f

            self.prev_error_t = error_t
            self.prev_error_f = error_f

            action = [move_speed, 0, yaw_speed]
        else:
            action = [0, 0, 0]
        
        self.last_action = action
        self.rgb_list.append(rgb_)

        return action
