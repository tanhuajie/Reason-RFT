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

from openai import AzureOpenAI
from PIL import Image

PROMPT_FORMAT = """I will provide you with two images, an original question, and its answer related to the image. Your task is to answer it requiring step-by-step Chain-of-Thought (CoT) reasoning. The reasoning process can include expressions like "let me think," "oh, I see," or other natural language thought expressions.

The task is to complete the spatial visual reasoning task according to the following rules.  

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


Input Format:
Question: {original_question}
Original Answer: {original_answer}

Output Format:
Answer: [answer with reasoning steps]
<think>step-by-step Chain-of-Thought (CoT) reasoning process here</think>
<answer>Original Answer here</answer>
"""


def get_image_data_url(image_input):
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


def gpt4o_query(image_list, prompt, max_retries=5, initial_delay=3):
    if image_list is None:
        return None

    data_url_list = [get_image_data_url(image) for image in image_list]
    client = AzureOpenAI(
        azure_endpoint="YOUR-AZURE-ENDPOINT",
        azure_deployment="gpt-4o",
        api_version="2024-08-06",
        api_key="YOUR-API-KEY",
    )

    for attempt in range(max_retries):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url_list[0]}},
                        {"type": "image_url", "image_url": {"url": data_url_list[1]}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
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


def process_item(item, image_dir, progress_bar):
    try:
        image_path = [os.path.join(image_dir, image_path) for image_path in item["image"]]
        formatted_prompt = PROMPT_FORMAT.format(
            original_question=item["problem"], original_answer=item["solution"]
        )

        response = gpt4o_query(image_path, formatted_prompt)
        if "<think>" in response:
            item["cot"] = "<think>" + response.split("<think>")[-1]
        else:
            item["cot"] = response
        progress_bar.update(1)
        return item, None  # Successful result
    except Exception as e:
        error_message = f"Error processing item: {item.get('id', 'Unknown')}, error: {e}\n"
        print(error_message.strip())

        with open("error.txt", "a", encoding="utf-8") as error_file:
            error_file.write(error_message)

        progress_bar.update(1)
        return None, error_message  # Return error message


def main(image_dir, input_json, output_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            data_with_cot = json.load(f)
    else:
        data_with_cot = []

    processed_count = len(data_with_cot)
    print(f"Resuming from item {processed_count}/{len(data)}...")

    total_items = len(data) - processed_count
    with tqdm(total=total_items) as progress_bar:
        # Prepare the partial function to use in ThreadPoolExecutor
        process_item_partial = partial(process_item, image_dir=image_dir, progress_bar=progress_bar)

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for i, item in enumerate(tqdm(data[processed_count:], initial=processed_count, total=len(data))):
                futures.append(executor.submit(process_item_partial, item))

            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()

                if result:
                    data_with_cot.append(result)
                else:
                    # Log the error if needed
                    print(f"Error occurred for an item: {error}")

                # Save checkpoint every 200 items
                if len(data_with_cot) % 200 == 0:
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(data_with_cot, f, ensure_ascii=False, indent=4)
                    print(f"Checkpoint saved at {len(data_with_cot)} items.")

    # Final save after processing all items
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data_with_cot, f, ensure_ascii=False, indent=4)

    print(f"Processed dataset saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with CoT generation")
    parser.add_argument("--image_dir", default=None, type=str, help="Path to the image directory")
    parser.add_argument("--input_json", default=None, type=str, help="Path to the input JSON file")
    parser.add_argument("--output_json", default=None, type=str, help="Path to the output JSON file")
    
    args = parser.parse_args()

    args.image_dir = "/home/vlm/finetune_json/cot/image"
    args.input_json = "/home/vlm/workspace/vision-open-r1-spatial/data/trance_train_60k.json"
    args.output_json = "/home/vlm/workspace/vision-open-r1-spatial/data/trance_train_60k_cot.json"

    main(args.image_dir, args.input_json, args.output_json)
