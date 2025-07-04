import json
import os
import numpy as np
import math
import argparse

def read_json_file(file_path):
    """ Read and return the content of a JSON file. """
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_averages(folder_path):
    """ Calculate and return the averages of specified keys in all JSON files within the given folder. """
    keys = ['status', 'finish', 'success', 'following_rate',  'following_step', 'total_step', 'collision']
    count = 0
    succ_count = 0

    # Traverse through all files in the folder
    following_step_list = []
    following_step_revise_list = []
    total_step_list = []
    following_rate_list = []
    success_list = []
    collision_list = []
    output_file = []
    file_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            record_now = {}
            if "info" in filename:
                continue
            
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    
    for file_path in file_paths:
        data = read_json_file(file_path)
        
        record_now['file_path'] = file_path
        record_now['result'] = data
        output_file.append(record_now.copy())
        
        # Accumulate values
        for key in keys:
            if key in data:
                result = data[key]
            else:
                continue
            
            if key == "success":
                if result:
                    succ_count += 1
            
            if key == "success" and not math.isinf(result) and not math.isnan(result) and result:
                success_list.append(file_path)
        
            if key == "following_rate" and not math.isinf(result) and not math.isnan(result):
                following_rate_list.append(result)
                
            if key == "following_step" and not math.isinf(result) and not math.isnan(result):
                following_step_list.append(result)
                
            if key == "total_step" and  not math.isinf(result) and not math.isnan(result):
                total_step_list.append(result)
                total_step_path = os.path.join('track_episode_step', os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path))
                if os.path.exists(total_step_path):
                    total_revised_steps = read_json_file(total_step_path)
                    following_step_revise_list.append(max(total_revised_steps['total_step'], result))
                else:
                    following_step_revise_list.append(result)

            if key == "collision" and not math.isinf(result) and not math.isnan(result):
                collision_list.append(result)     

        count += 1
    
    with open("following_info.json", "w") as f:
        json.dump(output_file, f, indent=2)

    total_steps = sum(following_step_revise_list)
    total_following_steps = sum(following_step_list)
    return  {"episode count:": len(file_paths), "success rate": succ_count / count, "following rate:": total_following_steps / total_steps, "collision rate:": sum(collision_list) / count}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="The path store eval results.",
    )
    args = parser.parse_args()
    folder_path = args.path

    averages = calculate_averages(folder_path)
    print(averages)

if __name__ == "__main__":
    main()
