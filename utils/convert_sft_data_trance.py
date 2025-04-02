import json
from tqdm import tqdm
import argparse

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

Now, it's your turn!

{Question}
'''

COT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

- **Reasoning process**: Enclosed within `<think>...</think>`  
- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`  

Now, it's your turn!

{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

TRANCE_QUESTION_CAPTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions where each function is represented as **'func(object_id, value)'**.  

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
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer.

- **Summary process**: Summary how you will approach the problem and explain the steps you will take to reach the answer, enclosed within `<summary>...</summary>`

- **Caption process**: Provide a detailed description of the image, particularly emphasizing the aspects related to the question, enclosed within `<caption>...</caption>`

- **Reasoning process**: Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning, enclosed within `<think>...</think>`  

- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`

Now, it's your turn!

{Question} 
'''

def process_json_sft(input_file, output_file):
   with open(input_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
   item_list = []
   for item in tqdm(data):
      item_save = {
         "id": item["id"],
         "images": item["image"],
         "messages": [
            {
               "content": SFT_TRANCE_QUESTION_PROMPT.format(Question=f"<image>\n<image>\n{item['problem']}"),
               "role": "user"
            },
            {
               "content": item["solution"],
               "role": "assistant"
            }
         ]
      },
      item_list.append(item_save[0])

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(item_list, f, ensure_ascii=False, indent=4)

def process_json_cot_sft(input_file, output_file):
   with open(input_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
   item_list = []
   for item in tqdm(data):
      item_save = {
         "id": item["id"],
         "images": item["image"],
         "messages": [
            {
               "content": COT_TRANCE_QUESTION_PROMPT.format(Question=f"<image>\n<image>\n{item['problem']}"),
               "role": "user"
            },
            {
               "content": item["cot"],
               "role": "assistant"
            }
         ]
      },
      item_list.append(item_save[0])

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(item_list, f, ensure_ascii=False, indent=4)

def process_json_caption_cot_sft(input_file, output_file):
   with open(input_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
   item_list = []
   for item in tqdm(data):
      item_save = {
         "id": item["id"],
         "images": item["image"],
         "messages": [
            {
               "content": TRANCE_QUESTION_CAPTION_PROMPT.format(Question=f"<image>\n<image>\n{item['problem']}"),
               "role": "user"
            },
            {
               "content": item["cot"],
               "role": "assistant"
            }
         ]
      },
      item_list.append(item_save[0])

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(item_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Process dataset with CoT generation")
   parser.add_argument("--input_json", default=None, type=str, help="Path to the input JSON file")
   parser.add_argument("--output_json", default=None, type=str, help="Path to the output JSON file")
   args = parser.parse_args()

   process_json_sft(args.input_json, args.output_json)
   process_json_cot_sft(args.input_json, args.output_json)
   process_json_caption_cot_sft(args.input_json, args.output_json)