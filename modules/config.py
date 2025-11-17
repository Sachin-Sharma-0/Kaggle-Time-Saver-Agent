"""
Configuration Module - Converted from Block 1
Handles API keys and environment setup
"""

import os
import json
from pathlib import Path


def setup_environment():
    """
    Setup environment variables and Kaggle credentials.
    Replaces Colab's userdata.get() with local .env file or environment variables.
    """
    # Load from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    # Check for required API keys
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")
    
    if not google_api_key:
        raise ValueError("❌ Missing GOOGLE_API_KEY. Set it in .env file or environment variables.")
    
    if not kaggle_username or not kaggle_key:
        raise ValueError("❌ Missing Kaggle credentials (KAGGLE_USERNAME or KAGGLE_KEY).")
    
    # Set Google API Key for Gemini
    os.environ["GOOGLE_API_KEY"] = google_api_key
    print("✓ Google API Key configured.")
    
    # Setup Kaggle credentials
    setup_kaggle_credentials(kaggle_username, kaggle_key)
    
    return {
        "google_api_key": google_api_key,
        "kaggle_username": kaggle_username,
        "kaggle_key": kaggle_key
    }


def setup_kaggle_credentials(username: str, key: str):
    """
    Write Kaggle credentials file to ~/.kaggle/kaggle.json
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json_path = kaggle_dir / "kaggle.json"
    
    with open(kaggle_json_path, "w") as f:
        json.dump({
            "username": username,
            "key": key
        }, f)
    
    # Set permissions (600) - read/write for owner only
    if os.name != 'nt':  # Unix-like systems
        os.chmod(kaggle_json_path, 0o600)
    
    print("✓ kaggle.json created and permissions set.")

