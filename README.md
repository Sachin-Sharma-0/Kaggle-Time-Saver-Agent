# Kaggle Time-Saver Agent

An AI-powered agent that helps you analyze Kaggle datasets quickly using Google Gemini AI with function calling capabilities.

## Features

- 🤖 **AI-Powered Analysis**: Ask natural language questions about your dataset
- 📊 **Data Exploration**: List columns, describe statistics, find missing values
- 📈 **Visualizations**: Generate histograms, bar charts, and correlation heatmaps
- 🧠 **Machine Learning**: Train models (Logistic Regression, Random Forest) and make predictions
- 🗄️ **SQL Queries**: Run SQL queries on your dataset
- 📄 **Report Generation**: Create HTML/PDF reports with statistics and charts
- 📁 **Multi-Dataset Support**: Load and switch between multiple datasets
- 🔽 **Kaggle Integration**: Download datasets directly from Kaggle

## Project Structure

```
Kaggle_Time_Saver_Agent/
├── app.py                      # Main application entry point
├── modules/
│   ├── __init__.py
│   ├── config.py              # Configuration and environment setup (Block 1)
│   ├── kaggle_utils.py        # Kaggle API helpers (Block 2)
│   ├── dataset_loader.py      # Dataset download/loading (Blocks 3, 4, 10.5)
│   └── agent.py               # Core AI agent with all tools (Blocks 5-10)
├── data/                      # Dataset storage directory
├── reports/                   # Generated reports directory
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Kaggle API credentials
- Google Gemini API key

### 2. Install Dependencies

```bash
# Activate your virtual environment (if using one)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

**Note**: For PDF generation, you may also need to install `wkhtmltopdf`:
- **Windows**: Download from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)
- **Linux**: `sudo apt-get install wkhtmltopdf`
- **Mac**: `brew install wkhtmltopdf`

### 3. Configure API Keys

1. Copy the example environment file:
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env    # Linux/Mac
   ```

2. Edit `.env` and add your credentials:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

   **Getting API Keys:**
   - **Google Gemini API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Kaggle API**: Get from [Kaggle Settings](https://www.kaggle.com/settings) → API section

### 4. Run the Application

```bash
python app.py
```

## Usage

### Running the Streamlit Web App (Recommended)

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Running the CLI Version

```bash
python app.py
```

### Loading a ZIP Dataset (e.g., `titanic.zip`)

1. Place the ZIP file inside the `data/` folder.
2. Open the Streamlit app (`streamlit run streamlit_app.py`).
3. Go to `📁 Load Dataset` → `📦 Zipped Datasets Detected`.
4. Select the ZIP file, choose a dataset name, and click **Extract & Load ZIP Dataset**.

When you run `app.py`, you'll be prompted to:
1. Download Titanic competition dataset
2. Download a Kaggle dataset by ID (e.g., `uciml/iris`)
3. Load a dataset from a local file
4. Skip and load a dataset later via the agent

### Example Queries

Once the agent is running, you can ask questions like:

```
🤖 You: What columns are in the dataset?
🤖 You: Show me statistics for the Age column
🤖 You: How many missing values are there?
🤖 You: Create a histogram for the Fare column
🤖 You: Generate a correlation heatmap
🤖 You: Train a random_forest model
🤖 You: Predict survival for Pclass=3, Sex=male, Age=30, Fare=7.25
🤖 You: Run SQL query: SELECT * FROM dataset LIMIT 10
🤖 You: Generate a report with title "Titanic Analysis"
🤖 You: Download the iris dataset from Kaggle
🤖 You: Load the dataset from ./data/iris.csv with name "iris"
```

## Code Block Mapping

This project is organized based on your Google Colab notebook blocks:

- **Block 1** → `modules/config.py`: Environment setup and API key configuration
- **Block 2** → `modules/kaggle_utils.py`: Kaggle API helper functions
- **Block 3** → `modules/dataset_loader.py`: Competition dataset download
- **Block 4** → `modules/dataset_loader.py`: Dataset loading and preview
- **Block 4.5** → `modules/config.py`: Alternative API key loading
- **Block 5** → `modules/agent.py`: Base tools (list_columns, describe_column, missing_values)
- **Block 6** → `modules/agent.py`: Chart tools (histogram, bar chart, correlation)
- **Block 7** → `modules/agent.py`: ML tools (train_model, predict, list_models)
- **Block 8** → `modules/agent.py`: SQL query tool
- **Block 9** → `modules/agent.py`: Report generation (HTML/PDF)
- **Block 10** → `modules/agent.py`: Multi-dataset support
- **Block 10.5** → `modules/dataset_loader.py`: Simple Kaggle dataset download

## Key Differences from Colab

1. **Environment Variables**: Uses `.env` file instead of Colab's `userdata.get()`
2. **Display**: Uses `print()` instead of Colab's `display()` for images
3. **File Paths**: Uses local file system paths instead of Colab's `/content/`
4. **Kaggle Credentials**: Stored in `~/.kaggle/kaggle.json` instead of Colab secrets

## Troubleshooting

### Kaggle API Issues
- Ensure your Kaggle credentials are correct in `.env`
- Check that `kaggle.json` exists in `~/.kaggle/` with correct permissions (600)
- For some datasets, you may need to accept terms on Kaggle website first

### PDF Generation Issues
- Ensure `wkhtmltopdf` is installed and in your PATH
- On Windows, you may need to specify the path to `wkhtmltopdf` in your code

### API Key Issues
- Verify your Google API key is valid and has access to Gemini models
- Check that `.env` file is in the project root directory

## Development

To extend the agent with new tools:

1. Add your function to `modules/agent.py`
2. Create a function declaration dictionary
3. Add it to the appropriate declarations list
4. Update `rebuild_tool_schema()` to include your new tool

## License

This project is part of the Google x Kaggle 5-day intensive AI training capstone project.

## Acknowledgments

- Google Gemini AI for the function calling capabilities
- Kaggle for dataset access
- All the open-source libraries that make this project possible
