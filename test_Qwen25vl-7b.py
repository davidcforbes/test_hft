# Install required packages
# pip install transformers accelerate torch pillow requests

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import time
from datetime import datetime
from io import BytesIO

print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
init_start = time.time()

# Load model and processor
print("Loading Qwen2.5-VL-7B-Instruct model and processor...")
model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

init_end = time.time()
init_duration = init_end - init_start
print(f"Model loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Duration: {init_duration:.2f} seconds")

def analyze_image_from_url(url: str, prompt: str = "What do you see in this image?"):
    """Analyze an image from URL using Qwen2.5-VL-7B"""
    
    # Download and open image
    print(f"Downloading image from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image downloaded successfully, size: {image.size}")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None
    
    # Prepare messages in the correct format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    print("Generating response...")
    generation_start = time.time()
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    generation_end = time.time()
    generation_duration = generation_end - generation_start
    print(f"Response generated in: {generation_duration:.2f} seconds")
    
    return output_text[0] if output_text else None

def analyze_local_image(image_path: str, prompt: str = "What do you see in this image?"):
    """Analyze a local image file using Qwen2.5-VL-7B"""
    print(f"Loading local image: {image_path}")
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully, size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Prepare messages in the correct format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    print("Generating response...")
    generation_start = time.time()
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    generation_end = time.time()
    generation_duration = generation_end - generation_start
    print(f"Response generated in: {generation_duration:.2f} seconds")
    
    return output_text[0] if output_text else None

def analyze_multiple_images(image_paths: list, prompt: str = "Extract all of the text from the attached image files and generate a single output markdown file that combines all extracted text and matches the original image format, style, look, and feel. Ensure you preserve bullet points, lists, and tables, but do not include graphics. Do not include your own disclaimers or comments in your response. Do not hallucinate."):
    """Analyze multiple local image files using Qwen2.5-VL-7B"""
    print(f"Loading {len(image_paths)} images: {', '.join(image_paths)}")
    
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            print(f"Image {image_path} loaded successfully, size: {image.size}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    # Prepare messages with all images
    content = []
    for image in images:
        content.append({
            "type": "image",
            "image": image,
        })
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response with increased token limit for multiple images
    print("Generating response...")
    generation_start = time.time()
    
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    generation_end = time.time()
    generation_duration = generation_end - generation_start
    print(f"Response generated in: {generation_duration:.2f} seconds")
    
    return output_text[0] if output_text else None

def process_vision_info(messages):
    """Extract vision information from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for content_item in message["content"]:
                if content_item["type"] == "image":
                    image_inputs.append(content_item["image"])
                elif content_item["type"] == "video":
                    video_inputs.append(content_item["video"])
    
    return image_inputs, video_inputs

# Example usage
if __name__ == "__main__":
    # Process all four image files
    print("\n" + "="*50)
    print("Testing Qwen2.5-VL-7B with multiple images")
    print("="*50)
    
    image_files = ["input_page_1.png", "input_page_2.png", "input_page_3.png", "input_page_4.png"]
    prompt = "Extract all of the text from the attached image files and generate a single output markdown file that combines all extracted text and matches the original image format, style, look, and feel. Ensure you preserve bullet points, lists, and tables, but do not include graphics. Do not include your own disclaimers or comments in your response. Do not hallucinate."
    
    analysis_start = time.time()
    print(f"Starting multiple image analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = analyze_multiple_images(image_files, prompt)
    
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"Multiple image analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Analysis duration: {analysis_duration:.2f} seconds")
    
    if result:
        print(f"\nMultiple Image Analysis Result Preview:")
        print("-" * 50)
        print(f"Result length: {len(str(result))} characters")
        print(f"Result preview: {str(result)[:300]}...")
        print("-" * 50)
        
        # Save result to markdown file
        with open("Qwen25-VL-7b.md", "w", encoding="utf-8") as f:
            f.write(str(result))
        
        print(f"\nResult saved to: Qwen25-VL-7b.md")
    else:
        print("Failed to analyze multiple images")
    
    total_end = time.time()
    total_duration = total_end - init_start
    print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration:.2f} seconds")


''''' 
PS C:\Development\test_hft> python test_Qwen25vl-7b.py
Script started at: 2025-06-03 12:37:46
Loading Qwen2.5-VL-7B-Instruct model and processor...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:06<00:00,  1.31s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
Model loaded at: 2025-06-03 12:37:56, Duration: 9.82 seconds

==================================================
Testing Qwen2.5-VL-7B with multiple images
==================================================
Starting multiple image analysis at: 2025-06-03 12:37:56
Loading 4 images: input_page_1.png, input_page_2.png, input_page_3.png, input_page_4.png
Image input_page_1.png loaded successfully, size: (1224, 1584)
Image input_page_2.png loaded successfully, size: (1224, 1584)
Image input_page_3.png loaded successfully, size: (1224, 1584)
Image input_page_4.png loaded successfully, size: (1224, 1584)
Generating response...
Response generated in: 1531.24 seconds
Multiple image analysis completed at: 2025-06-03 13:03:28, Analysis duration: 1531.90 seconds

Multiple Image Analysis Result Preview:
--------------------------------------------------
Result length: 5307 characters
Result preview: ```markdown
# PURCHASE AGREEMENT

**Purchase Agreement**
D1176564
Page 1

This Purchase Agreement Number must appear on all order acknowledgements, packing lists, cartons, and correspondence.

**Ship To:**
Receiving: D1176564
*** MULTIPLE "Ship to" LOCATIONS MAY EXIST ***
Please review line items fo...
--------------------------------------------------

Result saved to: Qwen25-VL-7b.md

Script completed at: 2025-06-03 13:03:28
Total duration: 1541.72 seconds
PS C:\Development\test_hft> 
'''