"""
Process input.pdf file by converting each page to images and analyzing them with InternVL3 model.
Saves all responses to output.md file.
"""

import os
from pathlib import Path
from typing import List, Any
from datetime import datetime

# PDF processing imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

# Model imports
from transformers import AutoTokenizer, AutoModel
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch


def extract_images_from_pdf(pdf_path: str) -> List[Any]:
    """
    Extract images from PDF pages using PyMuPDF.
    Returns a list of PIL Images, one per page.
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: uv add pymupdf")
    
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for image processing. Install with: uv add pillow")
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Get the base filename without extension
    pdf_name = Path(pdf_path).stem
    
    images = []
    pdf_document = fitz.open(pdf_path)  # type: ignore
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {pdf_document.page_count}")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        # Render page as image with high DPI for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # type: ignore
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        from io import BytesIO
        img = Image.open(BytesIO(img_data))  # type: ignore
        
        # Save image to output folder with proper naming
        output_filename = f"{pdf_name}_page_{page_num + 1}.png"
        output_path = output_dir / output_filename
        img.save(output_path)
        
        images.append(img)
        
        print(f"Extracted page {page_num + 1}/{pdf_document.page_count} as image -> {output_path}")
    
    pdf_document.close()
    return images


def load_internvl3_model(use_8bit=True):
    """
    Load InternVL3 model with optional 8-bit quantization.
    Falls back to CPU-compatible loading if CUDA is not available.
    """
    model_name = "OpenGVLab/InternVL3-8B"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Configure model loading
    if device == "cuda" and use_8bit:
        print("Loading model with 8-bit quantization...")
        print("This will significantly reduce memory usage while maintaining good performance.")
        
        try:
            # Configure 8-bit quantization with CPU offloading support
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for large models
            )
            
            # Try to load with automatic device mapping
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                max_memory={0: "20GB", "cpu": "30GB"}  # Limit GPU memory usage
            )
            print("✅ Successfully loaded model with 8-bit quantization!")
            
            # Print memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
        except Exception as e:
            print(f"⚠️ Failed to load with 8-bit quantization: {e}")
            print("Falling back to bfloat16 precision...")
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).cuda()
            
    elif device == "cuda":
        print("Loading model with bfloat16 precision (8-bit disabled)...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda()
    else:
        print("⚠️ CUDA not available. 8-bit quantization requires GPU.")
        print("Loading model for CPU (float32) - this will use more memory and be slower...")
        print("For better performance, consider using a GPU-enabled environment.")
        
        # For CPU, we might want to use a smaller model
        # For CPU, we'll use the 8B model which is already smaller than 14B
        print("Using InternVL3-8B model for CPU processing...")
        
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def analyze_image_with_model(model, tokenizer, image, question="Describe what you see in this image in detail."):
    """
    Analyze an image using the InternVL3 model.
    """
    try:
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Try the standard chat method first
        try:
            pixel_values = model.extract_feature(image)
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config=dict(max_new_tokens=1024, do_sample=False)
            )
            return response
        except Exception as e:
            # Fallback to alternative method
            print(f"Standard method failed, trying alternative: {e}")
            messages = [{'role': 'user', 'content': f'<image>\n{question}'}]
            response = model.chat(
                tokenizer,
                image,
                messages,
                generation_config=dict(max_new_tokens=1024, do_sample=False)
            )
            return response
            
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {e}"


def save_output_to_markdown(responses: List[str], pdf_name: str) -> str:
    """
    Save all responses to a markdown file called output.md.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# PDF Analysis Output

**Source File**: {pdf_name}.pdf
**Generated**: {timestamp}
**Total Pages**: {len(responses)}

---

## Analysis Results

"""
    
    for i, response in enumerate(responses, 1):
        markdown_content += f"""### Page {i}

{response}

---

"""
    
    markdown_content += """
*Generated by process_input_pdf.py using InternVL3 model*
"""
    
    # Write to output.md
    output_path = "output.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return output_path


def main():
    """
    Main function to process input.pdf file.
    """
    pdf_path = "input.pdf"
    
    # Check if input.pdf exists
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found in current directory")
        return
    
    try:
        # Step 1: Extract images from PDF
        print("Step 1: Extracting images from PDF...")
        images = extract_images_from_pdf(pdf_path)
        
        if not images:
            print("No images extracted from PDF")
            return
        
        # Step 2: Load the model with 8-bit quantization
        print("\nStep 2: Loading InternVL3 model with 8-bit quantization...")
        model, tokenizer = load_internvl3_model(use_8bit=True)
        
        # Step 3: Analyze each image
        print("\nStep 3: Analyzing images with model...")
        responses = []
        question = "Describe what you see in this image in detail. Include any text, diagrams, charts, or other visual elements you can identify."
        
        for i, image in enumerate(images, 1):
            print(f"\nAnalyzing page {i}/{len(images)}...")
            response = analyze_image_with_model(model, tokenizer, image, question)
            responses.append(response)
            print(f"Page {i} analysis complete")
        
        # Step 4: Save all responses to output.md
        print("\nStep 4: Saving results to output.md...")
        pdf_name = Path(pdf_path).stem
        output_file = save_output_to_markdown(responses, pdf_name)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_file}")
        print(f"Total pages processed: {len(responses)}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()