"""
Kaggle Time-Saver Agent - Main Application
Interactive CLI interface for the AI agent
"""

import os
from modules.config import setup_environment
from modules.agent import initialize_agent, ask_agent, load_dataset, train_df
from modules.dataset_loader import download_competition_dataset, load_dataset_from_path, download_kaggle_dataset_simple


def main():
    """Main application entry point."""
    print("=" * 60)
    print("Kaggle Time-Saver Agent")
    print("=" * 60)
    
    # Setup environment (API keys, etc.)
    try:
        config = setup_environment()
        print("✓ Environment setup complete.")
    except ValueError as e:
        print(f"❌ {e}")
        print("\nPlease create a .env file with the following variables:")
        print("GOOGLE_API_KEY=your_google_api_key")
        print("KAGGLE_USERNAME=your_kaggle_username")
        print("KAGGLE_KEY=your_kaggle_key")
        return
    
    # Initialize Gemini model
    try:
        model = initialize_agent(config["google_api_key"])
        print("✓ Gemini AI agent initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Load initial dataset (optional)
    print("\n" + "=" * 60)
    print("Dataset Setup")
    print("=" * 60)
    print("\nOptions:")
    print("1. Download Titanic competition dataset")
    print("2. Download a Kaggle dataset by ID (e.g., uciml/iris)")
    print("3. Load dataset from local file")
    print("4. Skip (load dataset later via agent)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if download_competition_dataset("titanic"):
            train_df_local, test_df, gender_df = load_dataset_from_path()
            # Register with agent
            result = load_dataset("titanic", "./data/train.csv")
            print(f"\n{result.get('message', result.get('error', 'Unknown result'))}")
    elif choice == "2":
        dataset_id = input("Enter Kaggle dataset ID (e.g., uciml/iris): ").strip()
        download_kaggle_dataset_simple(dataset_id)
        # Try to find and load the CSV file
        csv_files = [f for f in os.listdir("./data") if f.endswith('.csv')]
        if csv_files:
            dataset_name = input(f"Enter a name for this dataset: ").strip() or dataset_id.split('/')[-1]
            filepath = os.path.join("./data", csv_files[0])
            result = load_dataset(dataset_name, filepath)
            print(f"\n{result.get('message', result.get('error', 'Unknown result'))}")
    elif choice == "3":
        filepath = input("Enter file path to CSV: ").strip()
        dataset_name = input("Enter a name for this dataset: ").strip() or "dataset"
        result = load_dataset(dataset_name, filepath)
        print(f"\n{result.get('message', result.get('error', 'Unknown result'))}")
    else:
        print("Skipping dataset loading. You can load a dataset later by asking the agent.")
    
    # Interactive agent loop
    print("\n" + "=" * 60)
    print("AI Agent Ready!")
    print("=" * 60)
    print("\nYou can now ask questions about your dataset.")
    print("The agent can:")
    print("  - List columns and describe data")
    print("  - Generate charts and visualizations")
    print("  - Train ML models and make predictions")
    print("  - Run SQL queries")
    print("  - Generate HTML/PDF reports")
    print("  - Load and switch between multiple datasets")
    print("\nType 'exit' or 'quit' to end the session.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Agent: ", end="", flush=True)
            response = ask_agent(user_input, model)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
