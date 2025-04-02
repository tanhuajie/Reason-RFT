import argparse
import base64
import concurrent.futures
import json
import os
import random
import time
from functools import partial
from io import BytesIO

from tqdm import tqdm

from openai import AzureOpenAI, OpenAI
from PIL import Image
from cal_score_benchmarks_for_open_source import (
    get_score_from_json_geometry3k,
    get_score_from_json_clevr,
    get_score_from_json_geomath,
    get_score_from_json_trance,
)

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

Now, it's your turn!

{Question} Output the final answer (sequence of functions only) within `<answer>...</answer>` 
'''

ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT = "Please answer in Arabic numerals. For example, if the answer is 3, please respond with 3. {Question}"

ZERO_SHOT_GEOMATH_QUESTION_PROMPT = "Please answer the question with only numbers (either integer or float, such as 1, 2, 5.2, etc.) or options (such as A, B, C, or D). If it is an option, please provide your answer as a single letter (A, B, C, or D). For example, if the answer is A, just respond with A. Do not include any explanations or additional text. {Question}"


class Evaluator_Baseline():
    def __init__(self, task_name, model_type):
        # Path to benchmark
        if task_name == "trance":
            self.benchmark_json = "/path/to/your/benchmarks/spatial_transformation/trance.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_TRANCE_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/trance_{model_type}.json"
            self.score_func = get_score_from_json_trance
        elif task_name == "trance-left":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_left.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_TRANCE_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/trance_left_{model_type}.json"
            self.score_func = get_score_from_json_trance
        elif task_name == "trance-right":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_right.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_TRANCE_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/trance_right_{model_type}.json"
            self.score_func = get_score_from_json_trance
        elif task_name == "geomath":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geomath.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_GEOMATH_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/geomath_{model_type}.json"
            self.score_func = partial(get_score_from_json_geomath, enhance=True)
        elif task_name == "geometry3k":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geometry3k.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_GEOMATH_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/geometry3k_{model_type}.json"
            self.score_func = partial(get_score_from_json_geometry3k, enhance=True)
        elif task_name == "clevr-math":
            self.benchmark_json = "/path/to/your/benchmarks/visual_counting/clevr_math.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/clevr_math_{model_type}.json"
            self.score_func = get_score_from_json_clevr
        elif task_name == "super-clevr":
            self.benchmark_json = "/path/to/your/benchmarks/visual_counting/super_clevr.json"
            self.image_dir = "/path/to/your/benchmarks/image"
            self.prompt = ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT
            self.output_json = f"/path/to/close_source/result/super_clevr_{model_type}.json"
            self.score_func = get_score_from_json_clevr
        
        self.task_name = task_name
        self.model_type = model_type

        self.model_version = "gemini-1.5-pro-002" if self.model_type == "gemini" else "gpt-4o-2024-08-06"
        self.client_type = "Default" 

    def run(self, only_score=False):
        print(f"==============================================================")
        print(f"Evaluating {self.task_name} for {self.model_type}")
        if not only_score:
            self.eval(self.image_dir, self.benchmark_json, self.output_json)
        print(f"Saving evaluation result to {self.output_json}")
        self.score_func(self.output_json)
        print(f"Calculating score from {self.output_json}")
        print(f"==============================================================\n")

    def get_image_data_url(self, image_input):
        if isinstance(image_input, str):
            image_input = Image.open(image_input)

        if not isinstance(image_input, Image.Image):
            raise ValueError("Unsupported image input type")

        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"


    def gpt4o_query(self, image_list, prompt, max_retries=5, initial_delay=3):
        if image_list is None:
            return None

        data_url_list = [self.get_image_data_url(image) for image in image_list]
        if self.client_type == "Azure":
            client = AzureOpenAI(
                azure_endpoint="YOUR-AZURE-ENDPOINT",
                azure_deployment="gpt-4o",
                api_version="2024-08-06",
                api_key="YOUR-API-KEY",
            )
        else:
            base_url="YOUR-API-BASE-URL"
            api_key="YOUR-API-KEY"
            client = OpenAI(base_url=base_url, api_key=api_key)

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image_url", "image_url": {"url": img}} for img in data_url_list],
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                response = client.chat.completions.create(
                    model=self.model_version,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=8192,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed after {max_retries} attempts. Last error: {str(e)}"
                    )
                delay = initial_delay * (2**attempt) + random.uniform(
                    0, 0.1 * initial_delay * (2**attempt)
                )
                time.sleep(delay)


    def process_item(self, item, image_dir, progress_bar):
        try:
            if isinstance(item["image"], list):
                image_path = [os.path.join(image_dir, image_path) for image_path in item["image"]]
            elif isinstance(item["image"], str):
                image_path = [os.path.join(image_dir, item["image"])]

            if self.task_name == "geomath":
                formatted_prompt = self.prompt.format(
                    Question=item["problem_no_prompt"]
                )
            else:
                formatted_prompt = self.prompt.format(
                    Question=item["problem"]
                )

            response = self.gpt4o_query(image_path, formatted_prompt)
            item["pred"] = response

            progress_bar.update(1)
            return item, None  # Successful result
        except Exception as e:
            error_message = f"Error processing item: {item.get('id', 'Unknown')}, error: {e}\n"
            print(error_message.strip())

            with open("error.txt", "a", encoding="utf-8") as error_file:
                error_file.write(error_message)

            progress_bar.update(1)
            return None, error_message  # Return error message


    def eval(self, image_dir, input_json, output_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        if os.path.exists(output_json):
            with open(output_json, 'r', encoding='utf-8') as f:
                data_with_pred = json.load(f)
        else:
            data_with_pred = []

        exist_id = []
        for exist_item in data_with_pred:
            exist_id.append(exist_item["id"])

        data_no_process = []
        for item_xx in data:
            if item_xx["id"] not in exist_id:
                data_no_process.append(item_xx)

        processed_count = len(data_with_pred)
        print(f"Resuming from item {processed_count}/{len(data)}...")

        total_items = len(data) - processed_count
        with tqdm(total=total_items) as progress_bar:
            # Prepare the partial function to use in ThreadPoolExecutor
            process_item_partial = partial(self.process_item, image_dir=image_dir, progress_bar=progress_bar)

            with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
                futures = []
                for i, item in enumerate(tqdm(data_no_process, total=len(data_no_process))):
                    futures.append(executor.submit(process_item_partial, item))

                for future in concurrent.futures.as_completed(futures):
                    result, error = future.result()

                    if result:
                        data_with_pred.append(result)
                    else:
                        # Log the error if needed
                        print(f"Error occurred for an item: {error}")

                    # Save checkpoint every 50 items
                    if len(data_with_pred) % 50 == 0:
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(data_with_pred, f, ensure_ascii=False, indent=4)
                        # print(f"Checkpoint saved at {len(data_with_pred)} items.")

        # Final save after processing all items
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data_with_pred, f, ensure_ascii=False, indent=4)

        print(f"Processed dataset saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with CoT generation")
    parser.add_argument("--task_name", default=None, type=str, help="task_name")
    parser.add_argument("--model_type", default=None, type=str, help="model_type")
    parser.add_argument("--only_score", action='store_true', help="whether only calculate score")
    
    args = parser.parse_args()

    Evaluator_Baseline(task_name=args.task_name, model_type=args.model_type).run(only_score=args.only_score)
