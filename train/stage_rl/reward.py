
import os
import re
import copy
import math

from datetime import datetime
from math_verify import parse, verify
from collections import deque

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 1.0
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail         
        
        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def math_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    epsilon1=0.05
    epsilon2=0.20
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Try to convert both answers to float for numerical comparison
            try:
                a_pred = float(student_answer)
                a_gt = float(ground_truth)
                
                # Calculate absolute difference
                diff = abs(a_pred - a_gt)
                abs_gt = abs(a_gt)
                
                # Handle exact match case
                if diff < epsilon1 * abs_gt:
                    reward = 1.0
                # Handle completely incorrect case
                elif diff > epsilon2 * abs_gt:
                    reward = 0.0
                # Handle partial match case with smooth transition
                else:
                    normalized_diff = (diff - epsilon1 * abs_gt) / ((epsilon2 - epsilon1) * abs_gt)
                    reward = 0.5 * (math.cos(math.pi * normalized_diff) + 1)
                    
            except ValueError:
                # If conversion to float fails, do exact string matching
                if student_answer == ground_truth:
                    reward = 1.0
                    
        except Exception:
            pass  # Keep reward as 0.0 if string matching fails
        
        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails           

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
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

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
                    f.write(f"Func-Object Mapping: {partial_matches_1}\n")
                    f.write(f"Func-Value Mapping: {partial_matches_2}\n")
                    f.write(f"Func-Only: {func_matches}\n")
            except:
                pass
    return rewards


def only_full_func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
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

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
            except:
                pass
    return rewards


def penalty_func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
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
            reward += partial_matches_1_num * item_score * -0.25
            
            # (func, value) mapping
            partial_matches_2_num = 0
            partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
            for p, s in partial_matches_2:
                if p in pred_queue and s in sol_queue:
                    partial_matches_2_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += partial_matches_2_num * item_score * -0.25
            
            # only-func mapping
            func_matches_num = 0
            func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
            for p, s in func_matches:
                if p in pred_queue and s in sol_queue:
                    func_matches_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += func_matches_num * item_score * -0.5

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
                    f.write(f"Func-Object Mapping: {partial_matches_1}\n")
                    f.write(f"Func-Value Mapping: {partial_matches_2}\n")
                    f.write(f"Func-Only: {func_matches}\n")
            except:
                pass
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def caption_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<summary>.*?</summary>\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]

LENGTH_REWARD_START_POINT = 800

def len_reward(completions, solution, current_step, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solution

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        correct = False
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Compare the extracted answers
            if student_answer == ground_truth:
                correct = True
            elif float(student_answer) == float(ground_truth):
                correct = True
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
        
        # If symbolic verification failed, try symbolic verification
        if correct is False and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    correct = True
            except Exception:
                pass  # Continue to next verification method if this fails
        correctness.append(copy.deepcopy(correct))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        if current_step < LENGTH_REWARD_START_POINT:
            reward = 0.0
        else:
            reward = 0.05 * reward

        rewards.append(float(reward))

    return rewards