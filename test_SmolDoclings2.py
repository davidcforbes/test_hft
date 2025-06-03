# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "docling-core",
#     "transformers",
#     "pillow",
#     "requests",
# ]
# ///
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from typing import Any, List, Dict  # Added for clearer static typing
from PIL import Image
from transformers.pipelines import pipeline
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

# Settings
SHOW_IN_BROWSER = True  # Export output as HTML and open in webbrowser.

# Load the SmolDocling model via transformers pipeline
pipe = pipeline(
    "image-text-to-text",
    model="ds4sd/SmolDocling-256M-preview",
    max_new_tokens=1024,
    do_sample=False
)

# Prepare input - using chat format as recommended
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    }
]
image_sources = ["input_page_1.png", "input_page_2.png", "input_page_3.png", "input_page_4.png"]

# Load image resources
pil_images = []
for image_source in image_sources:
    try:
        if urlparse(image_source).scheme != "":  # it is a URL
            response = requests.get(image_source, stream=True, timeout=10)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
        else:
            pil_image = Image.open(image_source)
        pil_images.append(pil_image)
        print(f"Loaded image: {image_source}")
    except (requests.RequestException, IOError, FileNotFoundError) as e:
        print(f"Warning: Failed to load image '{image_source}': {e}")
        continue

if not pil_images:
    raise RuntimeError("No images could be loaded successfully.")

# Process each image and collect outputs
all_outputs = []
all_images = []

for i, pil_image in enumerate(pil_images):
    print(f"\nProcessing image {i+1}/{len(pil_images)}...")
    # `transformers` may yield results as a generator when streaming.
    # Convert to a list so that we can safely index and satisfy static type-checkers.
    # Generate DocTags using the SmolDocling model
    # Create messages with actual image content for chat format
    chat_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        }
    ]
    
    try:
        pipeline_output = pipe(chat_messages, max_new_tokens=1024)
    except Exception as e:
        print(f"Error processing image {i+1}: {e}")
        continue
    if pipeline_output is None:
        print(f"Warning: Pipeline returned None for image {i+1}")
        continue
    
    results = list(pipeline_output)

    if not results or "generated_text" not in results[0]:
        print(f"Warning: Pipeline did not return expected output for image {i+1}")
        continue

    # Extract the generated DocTags from the chat response
    output_data = results[0]["generated_text"]  # type: ignore[index]

    if isinstance(output_data, list):
        # Handle list of chat messages - get the assistant's response
        if output_data and len(output_data) > 1:
            assistant_response = output_data[-1]  # Last message should be assistant
            if isinstance(assistant_response, dict) and "content" in assistant_response:
                output = str(assistant_response["content"])  # type: ignore[index]
            else:
                output = str(assistant_response)
        else:
            output = ""
    else:
        output = str(output_data)

    all_outputs.append(output)
    all_images.append(pil_image)
    
    print(f"DocTags for page {i+1}:")
    print(output)
    print("\n" + "="*50 + "\n")

# Populate document with all pages
print("Creating combined document...")

# Extract text content from the DocTags format
extracted_texts = []
for i, output in enumerate(all_outputs):
    # Extract text content from <text>...</text> tags
    import re
    text_matches = re.findall(r'<text[^>]*>([^<]+)</text>', output)
    
    if text_matches:
        # Clean up the text by removing coordinate prefixes
        cleaned_texts = []
        for text in text_matches:
            # Remove coordinate patterns like "303>35>364>41>" from the beginning
            cleaned_text = re.sub(r'^\d+>\d+>\d+>\d+>', '', text)
            if cleaned_text.strip():
                cleaned_texts.append(cleaned_text.strip())
        
        # Join all cleaned text with newlines
        page_text = '\n'.join(cleaned_texts)
        extracted_texts.append(page_text)
        print(f"Debug: Extracted and cleaned text from page {i+1}: {len(cleaned_texts)} text elements")
        print(f"Preview: {page_text[:200]}...")
    else:
        # Fallback: use the raw output
        extracted_texts.append(output.strip())
        print(f"Debug: No text tags found in page {i+1}, using raw output")

print(f"Debug: Number of extracted texts: {len(extracted_texts)}")

# Create markdown output directly from extracted text
print("Markdown:\n\n")
markdown_output = ""
for i, text_content in enumerate(extracted_texts):
    markdown_output += f"# Page {i+1}\n\n"
    
    # Split into lines and format as markdown
    lines = text_content.split('\n')
    for line in lines:
        if line.strip():  # Skip empty lines
            markdown_output += f"{line.strip()}\n\n"
    
    markdown_output += "---\n\n"  # Page separator

print(f"Debug: Generated markdown length: {len(markdown_output)}")
print(markdown_output)

# Also try the original DocTags approach for comparison
try:
    output_strs = [str(output) for output in all_outputs]
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(output_strs, all_images)
    doc = DoclingDocument(name="SampleDocument")
    doc.load_from_doctags(doctags_doc)
    print(f"Debug: Original DocTags approach - Document pages: {len(doc.pages) if hasattr(doc, 'pages') else 'No pages'}")
    
except Exception as e:
    print(f"Error creating document: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    doc = None

# HTML
if SHOW_IN_BROWSER and doc is not None:
    import webbrowser

    out_path = Path("./output.html")
    doc.save_as_html(out_path, image_mode=ImageRefMode.EMBEDDED)
    webbrowser.open(f"file:///{str(out_path.resolve())}")