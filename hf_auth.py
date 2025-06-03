"""
HuggingFace authentication configuration module.
Handles API key authentication for accessing private models and increasing rate limits.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import login, HfFolder


def load_hf_token(env_file: Optional[str] = None) -> Optional[str]:
    """
    Load HuggingFace API token from environment or .env file.
    
    Args:
        env_file: Path to .env file. Defaults to '.env' in current directory.
        
    Returns:
        The HuggingFace API token if found, None otherwise.
    """
    # First check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    
    # Try to load from .env file
    env_path = Path(env_file or ".env")
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
                            return value
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}")
    
    # Check if already logged in via huggingface-cli
    stored_token = HfFolder.get_token()
    if stored_token:
        return stored_token
    
    return None


def authenticate_hf(token: Optional[str] = None, use_auth_token: bool = True) -> bool:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HuggingFace API token. If None, will try to load from environment.
        use_auth_token: Whether to use authentication. Set to False to skip auth.
        
    Returns:
        True if authentication successful or skipped, False otherwise.
    """
    if not use_auth_token:
        return True
    
    # Load token if not provided
    if token is None:
        token = load_hf_token()
    
    if not token:
        print("Warning: No HuggingFace token found. Running without authentication.")
        print("To use authentication, set HF_TOKEN environment variable or create .env file.")
        return False
    
    try:
        # Login to HuggingFace Hub
        login(token=token, add_to_git_credential=False)
        print("Successfully authenticated with HuggingFace Hub")
        return True
    except Exception as e:
        print(f"Warning: HuggingFace authentication failed: {e}")
        return False


def get_model_kwargs(use_auth_token: bool = True) -> dict:
    """
    Get model loading kwargs with authentication if available.
    
    Args:
        use_auth_token: Whether to include authentication token.
        
    Returns:
        Dictionary of kwargs to pass to model/tokenizer loading functions.
    """
    if not use_auth_token:
        return {}
    
    token = load_hf_token()
    if token:
        return {"use_auth_token": token}
    
    return {}