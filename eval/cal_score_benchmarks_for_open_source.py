import json
import argparse
import re, os
from tqdm import tqdm
import pandas as pd
from collections import deque, defaultdict
from math_verify import parse, verify

def get_score_clevr(content, sol):
    reward = 0
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    try:
        if match:
            answer = match.group(1).strip()
            if sol == answer.strip():
                reward = 1.0
            elif float(verify(parse(answer), parse(sol))) > 0:
                reward = 1.0
        else:
            if sol.strip() == content.strip():
                reward = 1.0
            elif float(verify(parse(content), parse(sol))) > 0:
                reward = 1.0
    except:
        reward = 0.0
    
    return reward

def get_score_from_json_clevr(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    type_scores = defaultdict(list)
    
    # Process data to calculate scores
    for item in data:
        item_type = "default"
        score = get_score_clevr(item["pred"], item["answer"])
        item["score"] = score
        type_scores[item_type].append(score)
    
    for item_type, scores in type_scores.items():
        avg_score = 100 * sum(scores) / len(scores) if scores else 0.0
        print(f"Type: {item_type}, Average Score: {avg_score:.2f}, Count: {len(scores)}")
    
    with open(file_path.replace(".json", "_with_score.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def get_score_geomath(content, sol, enhance=False):
    reward = 0
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    try:
        if match:
            answer = match.group(1).strip()
            if sol == answer.strip():
                reward = 1.0
            elif float(verify(parse(answer), parse(sol))) > 0:
                reward = 1.0
        else:
            if sol.strip() == content.strip():
                reward = 1.0
            elif float(verify(parse(content), parse(sol))) > 0:
                reward = 1.0
            
            if enhance and reward == 0:
                if sol.strip() in content.strip():
                    reward = 1.0
    except:
        reward = 0.0
    
    return reward

def get_score_from_json_geomath(file_path, enhance=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    type_scores = defaultdict(list)
    
    # Process data to calculate scores
    for item in data:
        item_type = "choice" if item["answer"] in ["A", "B", "C", "D"] else "non_choice"
        score = get_score_geomath(item["pred"], item["answer"], enhance)
        item["score"] = score
        type_scores[item_type].append(score)
    
    for item_type, scores in type_scores.items():
        avg_score = 100 * sum(scores) / len(scores) if scores else 0.0
        print(f"Type: {item_type}, Average Score: {avg_score:.2f}, Count: {len(scores)}")
    
    with open(file_path.replace(".json", "_with_score.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def extract_items(text):
    pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
    matches = pattern.findall(text)
    filtered_matches = list(set(matches))
    return filtered_matches

def get_score_trance(content, sol):
    reward = 0.0
    
    content_match = re.search(r'<answer>(.*?)</answer>', content)
    content_match = content_match.group(1).strip() if content_match else content.strip()
    pred_list = extract_items(content_match)
    sol_list = extract_items(sol)
    
    if not sol_list:
        return 0.0
    
    item_score = 1.0 / max(len(pred_list), len(sol_list)) if pred_list else 0
    
    pred_queue = deque(pred_list)
    sol_queue = deque(sol_list)
    
    # full mapping
    full_mapping_num = 0
    exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
    for p, s in exact_matches:
        if p in pred_queue and s in sol_queue:
            full_mapping_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += full_mapping_num * item_score

    # (func, object_id) mapping
    partial_matches_1_num = 0
    partial_matches_1 = [(p, s) for p in pred_queue for s in sol_queue if p[:2] == s[:2]]
    for p, s in partial_matches_1:
        if p in pred_queue and s in sol_queue:
            partial_matches_1_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_1_num * item_score * 0.5
    
    # (func, value) mapping
    partial_matches_2_num = 0
    partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
    for p, s in partial_matches_2:
        if p in pred_queue and s in sol_queue:
            partial_matches_2_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_2_num * item_score * 0.5
    
    # only-func mapping
    func_matches_num = 0
    func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
    for p, s in func_matches:
        if p in pred_queue and s in sol_queue:
            func_matches_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += func_matches_num * item_score * 0.25
    
    return reward

def get_score_from_json_trance(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    type_scores = defaultdict(list)
    
    # Process data to calculate scores
    for item in data:
        item_type = "default"
        score = get_score_trance(item["pred"], item["answer"])
        item["score"] = score
        type_scores[item_type].append(score)
    
    avg_score_list = []
    for item_type, scores in type_scores.items():
        avg_score = 100 * sum(scores) / len(scores) if scores else 0.0
        print(f"Type: {item_type}, Average Score: {avg_score:.2f}, Count: {len(scores)}")
        avg_score_list.append(f"{avg_score:.2f}")
    
    with open(file_path.replace(".json", "_with_score.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)


def get_score_geometry3k(content, sol, enhance=False):
    reward = 0
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    try:
        if match:
            answer = match.group(1).strip()
            if sol == answer.strip():
                reward = 1.0
            elif float(verify(parse(answer), parse(sol))) > 0:
                reward = 1.0
        else:
            if sol.strip() == content.strip():
                reward = 1.0
            elif float(verify(parse(content), parse(sol))) > 0:
                reward = 1.0

            if enhance and reward == 0:
                if sol.strip() in content.strip():
                    reward = 1.0
    except:
        reward = 0.0
    
    return reward

def get_score_from_json_geometry3k(file_path, enhance=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    type_scores = defaultdict(list)
    
    # Process data to calculate scores
    for item in tqdm(data):
        item_type = "choice" if item["answer"] in ["A", "B", "C", "D"] else "non_choice"
        score = get_score_geometry3k(item["pred"], item["answer"], enhance)
        item["score"] = score
        type_scores[item_type].append(score)
    
    for item_type, scores in type_scores.items():
        avg_score = 100 * sum(scores) / len(scores) if scores else 0.0
        print(f"Type: {item_type}, Average Score: {avg_score:.2f}, Count: {len(scores)}")
    
    with open(file_path.replace(".json", "_with_score.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset with CoT generation")
    parser.add_argument("--ckpt_path", default=None, type=str, help="path to checkpoint to be evaluated")
    
    args = parser.parse_args()

    print(f"*********************************************************************************************************************")
    print(args.ckpt_path)
    print(f"*********************************************************************************************************************\n")

    print(f"==============================================================")
    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "clevr-math.json")
        print(json_path)
        get_score_from_json_clevr(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "super-clevr.json")
        print(json_path)
        get_score_from_json_clevr(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "geomath.json")
        print(json_path)
        get_score_from_json_geomath(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "geometry3k.json")
        print(json_path)
        get_score_from_json_geometry3k(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "trance.json")
        print(json_path)
        get_score_from_json_trance(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "trance-left.json")
        print(json_path)
        get_score_from_json_trance(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")

    try:
        json_path = os.path.join(args.ckpt_path, "vision-r1-result", "trance-right.json")
        print(json_path)
        get_score_from_json_trance(json_path)
    except:
        print(f"[ERROR] occured when evaluating {json_path} !!!")
        pass
    print(f"==============================================================")
