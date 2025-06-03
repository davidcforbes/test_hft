# Install required packages
# pip install transformers accelerate bitsandbytes>=0.39.0 torch pillow

from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
from PIL import Image
import time
from datetime import datetime
from typing import List, Dict, Union, Any

print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
init_start = time.time()

# Initialize 8-bit quantization config with CPU offload support
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load model using pipeline approach as recommended
model_id = "google/gemma-3-12b-it"
pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    model_kwargs={
        "quantization_config": quant_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16
    },
    use_fast=True
)

def analyze_images(image_paths: List[str], prompt: str) -> str:
    """Analyze multiple images with Gemma 3 12B using 8-bit quantization via pipeline"""
    # Load all images and convert to RGB
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # Create content list with text prompt and all images
    # Using proper typing to handle mixed content types
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image", "image": image})
    
    # Use the chat format with proper structure for Gemma 3 (based on the sample)
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Use pipeline for inference - pass messages directly
    result = pipe(
        messages,
        max_new_tokens=512  # Increased for multiple images
    )
    
    # Debug: Print the structure of the result to understand what we're getting
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if isinstance(result, list) else 'N/A'}")
    
    # Handle the chat completion format returned by the pipeline
    # The pipeline returns the full conversation including the assistant's response
    if isinstance(result, list) and len(result) > 0:
        print(f"Processing list result with {len(result)} items")
        
        # Look through all messages for assistant response
        for i, message in enumerate(result):
            print(f"Message {i}: type={type(message)}")
            if isinstance(message, dict):
                print(f"Message {i} keys: {list(message.keys())}")
                print(f"Message {i} role: {message.get('role')}")
                
                if message.get('role') == 'assistant':
                    response_content = message.get('content', '')
                    print(f"Found assistant response in message {i}")
                    print(f"Content type: {type(response_content)}")
                    print(f"Content length: {len(str(response_content))}")
                    print(f"Content preview: {str(response_content)[:100]}...")
                    
                    # Make sure we return just the content string, not the whole structure
                    return str(response_content)
        
        # If we get here, no assistant message was found
        print("ERROR: No assistant role found in any message!")
        print("Full result structure:")
        for i, item in enumerate(result):
            print(f"Item {i}: {type(item)} - {item}")
        
        # Return empty string instead of the whole structure
        return "ERROR: No assistant response found in pipeline result"
    else:
        print("Result is not a list or is empty")
        print(f"Full result: {result}")
        return "ERROR: Pipeline returned unexpected result format"

# Example usage with timing
init_end = time.time()
init_duration = init_end - init_start
print(f"Initialization completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Initialization duration: {init_duration:.2f} seconds")

analysis_start = time.time()
print(f"Starting image analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Process all four image files
image_files = ["input_page_1.png", "input_page_2.png", "input_page_3.png", "input_page_4.png"]
response = analyze_images(
    image_paths=image_files,
    prompt="Extract all of the text from the attached image files and generate a single output markdown file that combines all extracted text and matches the original image format, style, look, and feel. Ensure you preserve bullet points, lists, and tables, but do not include graphics. Do not include your own disclaimers or comments in your response. Do not hallucinate."
)

analysis_end = time.time()
analysis_duration = analysis_end - analysis_start
print(f"Image analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Analysis duration: {analysis_duration:.2f} seconds")

# Save response to file
print(f"Response type before saving: {type(response)}")
print(f"Response length: {len(str(response))}")
print(f"Response preview: {str(response)[:200]}...")

with open("gemma3-12b.md", "w", encoding="utf-8") as f:
    # The analyze_images function should return a clean string
    if isinstance(response, list):
        print("WARNING: Response is still a list! Attempting to extract assistant content...")
        # Try to extract assistant content from the list
        for item in response:
            if isinstance(item, dict) and item.get('role') == 'assistant':
                content = item.get('content', '')
                print(f"Found assistant content in list, writing to file...")
                f.write(str(content))
                break
        else:
            print("ERROR: No assistant content found in list, writing raw response...")
            f.write(str(response))
    else:
        # Normal case - response should be a string
        print("Response is a string, writing to file...")
        f.write(str(response))

end_time = time.time()
total_duration = end_time - init_start
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Total script duration: {total_duration:.2f} seconds")
print(f"Response saved to: gemma3-12b.md")
