"""
Kaggle Utilities Module - Converted from Block 2
Helper functions for interacting with Kaggle API
"""

import subprocess
import os


def execute_kaggle_command(command_list):
    """
    Execute a Kaggle CLI command.
    
    Args:
        command_list: List of command arguments (e.g., ["competitions", "list", "-s", "titanic"])
    
    Returns:
        str: Command output or error message
    """
    try:
        result = subprocess.run(
            ["kaggle"] + command_list,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


def search_competitions(search_term: str):
    """
    Search for Kaggle competitions.
    
    Args:
        search_term: Search query
    
    Returns:
        str: List of competitions or error message
    """
    output = execute_kaggle_command(["competitions", "list", "-s", search_term])
    if "Error:" in output:
        return output
    return output if output.strip() else f"No competitions found for search: {search_term}"


def get_competition_details(competition_name: str):
    """
    Get details about a specific competition.
    
    Args:
        competition_name: Name of the competition
    
    Returns:
        str: Competition files/details or error message
    """
    output = execute_kaggle_command(["competitions", "files", competition_name])
    if "error" in output.lower():
        return f"Could not retrieve details for competition '{competition_name}'."
    return output


def download_competition(competition_name: str, path: str = "./data"):
    """
    Download a competition dataset.
    
    Args:
        competition_name: Name of the competition
        path: Directory to save the dataset
    
    Returns:
        str: Download output or error message
    """
    print(f"Downloading competition: {competition_name}")
    os.makedirs(path, exist_ok=True)
    output = execute_kaggle_command([
        "competitions", "download",
        "-c", competition_name,
        "-p", path
    ])
    return output

