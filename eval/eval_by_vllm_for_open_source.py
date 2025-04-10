import json, os
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
import argparse

# ====================================
#  COT PROMPT
# ====================================

COT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provxided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Reasoning process**: Enclosed within `<think>...</think>`  
- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`  

Now, it's your turn!

{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

COT_CLEVR_MATH_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

COT_GEOMATH_QUESTION_PROMPT = "{Question}  Output the thinking process in <think> </think> and final answer (number or choice) in <answer> </answer> tags."

COT_GEOMETRY_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number or choice) in <answer> </answer> tags."


COT_TRANCE_QUESTION_WITH_CAPTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Summary process**: Summary how you will approach the problem and explain the steps you will take to reach the answer, enclosed within `<summary>...</summary>`

- **Caption process**: Provide a detailed description of the image, particularly emphasizing the aspects related to the question, enclosed within `<caption>...</caption>`

- **Reasoning process**: Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning, enclosed within `<think>...</think>`  

- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`

Now, it's your turn!

{Question} Output the summary process in <summary> </summary>, caption process in <caption>...</caption>, thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

# ====================================
#  SFT PROMPT
# ====================================

SFT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.

Now, it's your turn!

{Question}
'''

SFT_CLEVR_MATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMETRY_QUESTION_PROMPT = "{Question}"

# ====================================
#  Zero-Shot PROMPT
# ====================================

ZERO_SHOT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.

Now, it's your turn!

{Question} Please output the answer only with a sequence of functions for transformation.
'''

ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT = "Please answer in Arabic numerals. For example, if the answer is 3, please respond with 3. {Question}"

ZERO_SHOT_GEOMATH_QUESTION_PROMPT = "Please answer the question with only numbers (either integer or float, such as 1, 2, 5.2, etc.) or options (such as A, B, C, or D). If it is an option, please provide your answer as a single letter (A, B, C, or D). For example, if the answer is A, just respond with A. Do not include any explanations or additional text. {Question}"

class VL_Evaluator():
    def __init__(self, model_name_or_path, max_image_num=2):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image"} for _ in images],
                            {"type": "text", "text": self.prompt.format(Question=sample['problem_no_prompt'])},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image"} for _ in images],
                            {"type": "text", "text": self.prompt.format(Question=sample['problem'])},
                        ],
                    }
                ]

            vllm_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list
    
    def run(self, task_name="trance", eval_type="cot-sft", batch=16):

        assert task_name in ["trance", "trance-left", "trance-right", "clevr-math", "super-clevr", "geomath", "geometry3k"], \
            f"Task ({task_name}) is not supported. Please choose in ['trance', 'trance-left', 'trance-right', 'clevr-math', 'super-clevr', 'geomath', 'geometry3k']"
        
        assert eval_type in ["zero-shot", "sft", "cot-sft", "caption-cot"], f"Type ({eval_type}) is not supported. Please choose in ['zero-shot', 'sft', 'cot-sft']"

        # Prompt
        if eval_type == "cot-sft":
            if task_name in ["trance", "trance-left", "trance-right"]:
                self.prompt = COT_TRANCE_QUESTION_PROMPT
            elif task_name in ["clevr-math", "super-clevr"]:
                self.prompt = COT_CLEVR_MATH_QUESTION_PROMPT
            elif task_name in ["geomath"]:
                self.prompt = COT_GEOMATH_QUESTION_PROMPT
            elif task_name in ["geometry3k"]:
                self.prompt = COT_GEOMETRY_QUESTION_PROMPT
        elif eval_type == "sft":
            if task_name in ["trance", "trance-left", "trance-right"]:
                self.prompt = SFT_TRANCE_QUESTION_PROMPT
            elif task_name in ["clevr-math"]:
                self.prompt = SFT_CLEVR_MATH_QUESTION_PROMPT
            elif task_name in ["geomath"]:
                self.prompt = SFT_GEOMATH_QUESTION_PROMPT
            elif task_name in ["geometry3k"]:
                self.prompt = SFT_GEOMETRY_QUESTION_PROMPT
        elif eval_type == "zero-shot":
            if task_name in ["trance", "trance-left", "trance-right"]:
                self.prompt = ZERO_SHOT_TRANCE_QUESTION_PROMPT
            elif task_name in ["clevr-math", "super-clevr"]:
                self.prompt = ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT
            elif task_name in ["geomath", "geometry3k"]:
                self.prompt = ZERO_SHOT_GEOMATH_QUESTION_PROMPT
        elif eval_type == "caption-cot":
            self.prompt = COT_TRANCE_QUESTION_WITH_CAPTION_PROMPT

        # Path to benchmark
        if task_name == "trance":
            self.benchmark_json = "/path/to/your/benchmarks/spatial_transformation/trance.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "trance-left":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_left.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "trance-right":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_right.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "geomath":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geomath.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "geometry3k":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geometry3k.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "clevr-math":
            self.benchmark_json = "/path/to/your/benchmarks/visual_counting/clevr_math.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        elif task_name == "super-clevr":
            self.benchmark_json = "/path/to/your/benchmarks/visual_counting/super_clevr.json"
            self.image_dir = "/path/to/your/benchmarks/image"
        
        
        self.task_name = task_name
        self.eval_type = eval_type

        with open(self.benchmark_json, 'r') as file:
            data = json.load(file)

        sample_batch = []
        data_with_pred = []
        pred_times = 0

        for idx, sample in tqdm(enumerate(data), desc=f"{self.task_name}-{self.eval_type}", total=len(data)):
            sample_batch.append(sample)

            if idx % batch != batch - 1 and idx != len(data) - 1:
                continue
            else:
                sample_batch_with_pred = self.eval_batch(sample_batch, self.image_dir)
                data_with_pred += sample_batch_with_pred
                pred_times += 1
                sample_batch = []

            if pred_times % 10 == 0:
                self.path_to_save = os.path.join(self.model_name_or_path, "vision-r1-result")
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
                
                with open(os.path.join(self.path_to_save, f"{self.task_name}.json"), 'w', encoding='utf-8') as outfile:
                    json.dump(data_with_pred, outfile, indent=4)

        with open(os.path.join(self.path_to_save, f"{self.task_name}.json"), 'w', encoding='utf-8') as outfile:
            json.dump(data_with_pred, outfile, indent=4)

        print(f"Save to {os.path.join(self.path_to_save, f'{self.task_name}.json')}")


class QWEN_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2, min_pixels=3136, max_pixels=480000):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        try:
            self.processor.pad_token_id = self.processor.tokenizer.pad_token_id
            self.processor.eos_token_id = self.processor.tokenizer.eos_token_id
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels
        except:
            pass
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image"} for _ in images],
                            {"type": "text", "text": self.prompt.format(Question=sample['problem_no_prompt'])},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image"} for _ in images],
                            {"type": "text", "text": self.prompt.format(Question=sample['problem'])},
                        ],
                    }
                ]

            vllm_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list
    

class Mllama_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            max_model_len=4096,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_num_seqs=16,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                placeholders = "<|image|>" * len(images)
                prompt = f"{placeholders}<|begin_of_text|>{self.prompt.format(Question=sample['problem_no_prompt'])}"
                
            else:
                placeholders = "<|image|>" * len(images)
                prompt = f"{placeholders}<|begin_of_text|>{self.prompt.format(Question=sample['problem'])}"

            vllm_prompt = prompt

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list
    
class PHI3V_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(images, start=1))
                prompt = f"<|user|>\n{placeholders}\n{self.prompt.format(Question=sample['problem_no_prompt'])}<|end|>\n<|assistant|>\n"
                
            else:
                placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(images, start=1))
                prompt = f"<|user|>\n{placeholders}\n{self.prompt.format(Question=sample['problem'])}<|end|>\n<|assistant|>\n"

            vllm_prompt = prompt

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list
    

class Pixtral_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                placeholders = "[IMG]" * len(images)
                prompt = f"<s>[INST]{self.prompt.format(Question=sample['problem_no_prompt'])}\n{placeholders}[/INST]"
                
            else:
                placeholders = "[IMG]" * len(images)
                prompt = f"<s>[INST]{self.prompt.format(Question=sample['problem'])}\n{placeholders}[/INST]"

            vllm_prompt = prompt

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list


class Internvl_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list, image_dir):
            
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            if isinstance(sample["image"], list):
                for image in sample["image"]:
                    images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            else:
                images = [Image.open(os.path.join(image_dir, sample["image"])) if isinstance(sample["image"], str) else sample["image"]]

            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(images, start=1))
                messages = [{'role': 'user', 'content': f"{placeholders}\n{self.prompt.format(Question=sample['problem_no_prompt'])}"}]
                
            else:
                placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(images, start=1))
                messages = [{'role': 'user', 'content': f"{placeholders}\n{self.prompt.format(Question=sample['problem'])}"}]

            vllm_prompt = self.tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list


if __name__ == "__main__":

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Evaluate a model on different benchmarks with specified strategies.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation (default: 16)")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--benchmark_list', type=str, nargs='+', default=["trance", "clevr-math", "super-clevr", "geomath"], help="List of benchmarks to evaluate on.")
    parser.add_argument('--stratage_list', type=str, nargs='+', default=["cot-sft", "cot-sft", "cot-sft", "cot-sft"], help="List of strategies for each benchmark.")

    # Parse the arguments
    args = parser.parse_args()

    print(f"Benchmark List: {args.benchmark_list}")
    print(f"Stratage List: {args.stratage_list}")

    print(f"Loading Model Path from {args.model_name_or_path} ...")
    if 'qwen' in args.model_name_or_path.lower():
        print("======== Using QWEN_VL_Evaluator ==========")
        evaluator = QWEN_VL_Evaluator(args.model_name_or_path)
    elif 'llama' in args.model_name_or_path.lower() and 'vision' in args.model_name_or_path.lower():
        print("======== Using Mllama_VL_Evaluator ==========")
        evaluator = Mllama_VL_Evaluator(args.model_name_or_path)
    elif 'phi' in args.model_name_or_path.lower():
        print("======== Using PHI3V_VL_Evaluator ==========")
        evaluator = PHI3V_VL_Evaluator(args.model_name_or_path)
    elif 'internvl' in args.model_name_or_path.lower():
        print("======== Using Internvl_VL_Evaluator ==========")
        evaluator = Internvl_VL_Evaluator(args.model_name_or_path)
    elif 'pixtral' in args.model_name_or_path.lower():
        print("======== Using Pixtral_VL_Evaluator ==========")
        evaluator = Pixtral_VL_Evaluator(args.model_name_or_path)
    else:
        print("======== Using Default VL_Evaluator ==========")
        evaluator = VL_Evaluator(args.model_name_or_path)


    for benchmark, stratage in zip(args.benchmark_list, args.stratage_list):
        print(f"================== Evaluating {benchmark}-{stratage} ==================")
        evaluator.run(benchmark, stratage, args.batch_size)