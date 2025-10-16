import torch

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

import habitat
import numpy as np
import os
import re
import cv2
import imageio
from tqdm import trange
import os.path as osp
import json
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat.sims.habitat_simulator.actions import HabitatSimActions

def evaluate_agent(config, model_path, dataset_split, save_path) -> None:
    agent = UniNaVid_Agent(model_path, save_path)
    
    first_init = True
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        agent.reset()
        
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
                action = agent.act(obs, info, instruction, env.current_episode.episode_id)

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
            agent.reset(env.current_episode)

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


class UniNaVid_Agent(Agent):
    def __init__(self, model_path, result_path, exp_save='video'):
        print("Initialize UniNaVid")

        self.result_path = result_path
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "video" in exp_save else False

        self.conv_mode = "vicuna_v1"
        
        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, back, or stop."

        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()


    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        return [video]


    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs


    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image


    def reset(self, episode: NavigationEpisode = None):
        if len(self.topdown_map_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.topdown_map_list)

            print(f"Successfully save the episode video with episode id {episode.episode_id}")

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.first_forward = False
        

    def act(self, observations, info, instruction, episode_id):
        self.episode_id = episode_id
        rgb = observations["agent_1_articulated_agent_jaw_rgb"][:,:,:3]
        self.rgb_list.append(rgb)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_following"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, instruction, "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}

        navigation_qs = self.promt_template.format(instruction)
        navigation = self.predict_inference(navigation_qs)
        print("Output actions: ", navigation)
        if self.require_map:
            img = self.addtext(output_im, instruction, navigation)
            self.topdown_map_list.append(img)

        action_list = navigation.split(" ")
        
        if action_list[0] == "stop":
            action = [0, 0, 0]
        elif action_list[0] == "forward":
            action = [0.5, 0, 0]
        elif action_list[0] == "left":
            action = [0, 0, 1]
        elif action_list[0] == "right":
            action = [0, 0, -1]
        elif action_list[0] == "back":
            action = [-0.67, 0, 0]
        else:
            raise ValueError("wrong actions!, please check the code and data")
        
        return action

        
