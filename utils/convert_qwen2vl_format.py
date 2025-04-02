import json
import argparse
from tqdm import tqdm


def transform_item(item):
    """
    Transform a single item into the desired format.
    """
    # Ensure images are a list
    images = item.get("image", [])
    if isinstance(images, str):
        images = [images]

    # Transform conversations into messages
    messages = [
        {
            "content": conversation["value"],
            "role": "user" if conversation["from"] == "human" else "assistant"
        }
        for conversation in item["conversations"]
    ]

    return {
        "messages": messages,
        "images": images
    }


def process_json(input_file, output_file):
    """
    Read the input JSON file, process each item, and write the transformed items to the output file.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    transformed_data = []
    for item in tqdm(data, desc="Processing items"):
        transformed_data.append(transform_item(item))
        # print(transformed_data[-1])

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)

    print(f"Transformation complete! Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform JSON items into the desired format.")
    parser.add_argument("--input", default="/home/vlm/finetune_json/llava_v1_5_mix665k.json", help="Path to the input JSON file.")
    parser.add_argument("--output", default="/home/vlm/finetune_json/llava_v1_5_mix665k_qwen2vl_format.json", help="Path to the output JSON file.")
    args = parser.parse_args()

    process_json(args.input, args.output)
