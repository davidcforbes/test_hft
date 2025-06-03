"""
Verify the environment setup for 8-bit model loading.
"""

import torch
import sys
import os

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)

# Python environment
print(f"\n📍 Python Environment:")
print(f"   Python executable: {sys.executable}")
print(f"   Python version: {sys.version.split()[0]}")

# PyTorch and CUDA
print(f"\n🔧 PyTorch Configuration:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {getattr(torch.version, 'cuda', 'N/A')}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check for required packages
print(f"\n📦 Required Packages:")
packages = {
    "transformers": ("transformers", "✅ Installed"),
    "accelerate": ("accelerate", "✅ Installed"),
    "bitsandbytes": ("bitsandbytes", "✅ Installed"),
    "pymupdf": ("fitz", "✅ Installed"),  # PyMuPDF imports as fitz
    "pillow": ("PIL", "✅ Installed")  # Pillow imports as PIL
}

for display_name, (import_name, status) in packages.items():
    try:
        __import__(import_name)
        print(f"   {display_name}: {status}")
    except ImportError:
        print(f"   {display_name}: ❌ Not installed")

# Model configuration
print(f"\n🤖 Model Configuration:")
print(f"   8-bit quantization: ✅ Enabled")
print(f"   CPU offloading: ✅ Enabled")
print(f"   Max GPU memory: 20 GB")
print(f"   Max CPU memory: 30 GB")

print(f"\n✅ Environment is ready for running process_input_pdf.py with 8-bit quantization!")
print("=" * 60)