# General system prompt copied from open-r1
GENERAL_SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
    'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
    'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>'
)


# Special system prompt for spatial visual reasoning task (trance)
TRANCE_SYSTEM_PROMPT = '''You are a spatial visual reasoning assistant, and you need to complete the spatial reasoning task according to the following rules.  

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

''' 

# Special system prompt for spatial visual reasoning task (trance)
TRANCE_SYSTEM_CAPTION_PROMPT = '''You are a spatial visual reasoning assistant, and you need to complete the spatial reasoning task according to the following rules.  

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
''' 


CLEVR_MATH_SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT

GEOQA_SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT


# General question prompt copied from R1-V
GENERAL_QUESTION_PROMPT = '{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'


# Special question prompt for spatial visual reasoning task (trance)
TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

# Special question prompt for spatial visual reasoning task (trance)

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


TEMP_TRANCE_QUESTION_CAPTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

CLEVR_MATH_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

GEOQA_QUESTION_PROMPT = "{Question}"