# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hugging Face Transformers testing project focused on document analysis using Vision-Language Models (VLMs). The primary purpose is to process PDF documents by converting them to images and analyzing them with various state-of-the-art models.

## Key Architecture

The project has a modular architecture with:
- **Main pipeline** (`test_pipeline.py`): Unified interface supporting multiple AI tasks (text-generation, image-to-text, visual-question-answering, sentiment-analysis, question-answering)
- **Model-specific scripts**: Individual test files for Qwen2.5-VL, InternVL3, SmolDocling, and Gemma3 models
- **PDF processing**: Automatic conversion of PDFs to images using PyMuPDF for vision model input
- **Authentication**: HuggingFace API key authentication module (`hf_auth.py`) for accessing private/gated models

## Essential Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
uv venv
uv sync

# Quick environment check
uv run python test_env.py

# Comprehensive environment verification
uv run python verify_environment.py
```

### Running the Pipeline
```bash
# Text generation
uv run python test_pipeline.py --task text-generation --prompt "Hello world" --model gpt2

# Image captioning from PDF
uv run python test_pipeline.py --task image-to-text --pdf input.pdf --save-output

# Visual question answering
uv run python test_pipeline.py --task visual-question-answering --pdf input.pdf --prompt "What is this document about?"

# With specific model and optimizations
uv run python test_pipeline.py --task image-to-text --pdf input.pdf --model Salesforce/blip-image-captioning-large --fp16 --save-output
```

### Common Flags
- `--fp16`: Enable half-precision for memory efficiency
- `--no-auth`: Skip HuggingFace authentication (for public models only)
- `--save-output`: Save results to markdown files in ./output/
- `--max-new-tokens`: Control generation length (default: 512)

### Testing Specific Models
```bash
# Process PDF with InternVL3
uv run python test_process_pdf.py

# Test Qwen2.5-VL-7B
uv run python test_Qwen25vl-7b.py
```

## Authentication Setup

1. Copy `.env.example` to `.env`
2. Add your HuggingFace token from https://huggingface.co/settings/tokens
3. The pipeline will automatically authenticate when loading models

## Important Technical Details

- **GPU Support**: PyTorch with CUDA 12.8 is configured for GPU acceleration
- **Memory Optimization**: 8-bit quantization via bitsandbytes for large models
- **Package Manager**: Uses `uv` (not pip) for dependency management
- **Python Version**: Requires Python >= 3.12
- **Output Format**: Results are saved as markdown files with timestamps in ./output/