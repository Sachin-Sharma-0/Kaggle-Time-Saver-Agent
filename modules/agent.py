"""
Core Agent Module - Converted from Blocks 5-10
Main Gemini AI agent with function calling capabilities
"""

import pandas as pd
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import sqlite3
import zipfile
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from jinja2 import Template
import base64
from io import BytesIO
from xhtml2pdf import pisa


# ============================================
# Global State Management
# ============================================
_ALL_LOADED_DFS = {}
_CURRENT_DF_NAME = None
train_df = None  # Will be set when dataset is loaded

# ML Model Storage
MODELS = {}
ENCODERS = {}
FEATURES = {}

# SQL Database
conn = None
cursor = None


# ============================================
# Block 5: Base Tools
# ============================================
def list_columns():
    """Return all column names."""
    if train_df is None:
        return {"error": "No dataset loaded. Please load a dataset first."}
    return {"data": list(train_df.columns)}


def describe_column(column: str):
    """Return stats for a numeric column."""
    if train_df is None:
        return {"error": "No dataset loaded."}
    if column not in train_df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if pd.api.types.is_numeric_dtype(train_df[column]):
        return train_df[column].describe().to_dict()
    return {"error": f"Column '{column}' is not numeric."}


def missing_values():
    """Return columns with missing value counts."""
    if train_df is None:
        return {"error": "No dataset loaded."}
    return train_df.isnull().sum().to_dict()


BASE_FUNCTION_DECLARATIONS = [
    {
        "name": "list_columns",
        "description": "List all columns in the dataset.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "describe_column",
        "description": "Get statistics for a specific numeric column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"}
            },
            "required": ["column"]
        }
    },
    {
        "name": "missing_values",
        "description": "Count missing values in each column.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

BASE_TOOL_FUNCTIONS = {
    "list_columns": list_columns,
    "describe_column": describe_column,
    "missing_values": missing_values,
}


# ============================================
# Block 6: Chart Tools
# ============================================
def generate_filename(prefix="plot"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}.png"


def plot_histogram(column: str):
    """Generate histogram of a numeric column."""
    if train_df is None:
        return {"error": "No dataset loaded."}
    if column not in train_df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(train_df[column]):
        return {"error": f"Column '{column}' is not numeric."}
    
    filename = generate_filename("hist")
    plt.figure(figsize=(6, 4))
    plt.hist(train_df[column].dropna(), bins=20)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return {"image_path": filename}


def plot_bar(column: str):
    """Generate bar chart from a categorical column."""
    if train_df is None:
        return {"error": "No dataset loaded."}
    if column not in train_df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if pd.api.types.is_numeric_dtype(train_df[column]):
        return {"error": f"Column '{column}' is numeric, bar chart not useful."}
    
    filename = generate_filename("bar")
    plt.figure(figsize=(6, 4))
    train_df[column].value_counts().plot(kind="bar")
    plt.title(f"Bar Chart of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return {"image_path": filename}


def plot_correlation():
    """Generate correlation heatmap for numeric columns."""
    if train_df is None:
        return {"error": "No dataset loaded."}
    numeric_df = train_df.select_dtypes(include=['number'])
    
    filename = generate_filename("corr")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return {"image_path": filename}


chart_function_declarations = [
    {
        "name": "plot_histogram",
        "description": "Generate histogram of a numeric column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"}
            },
            "required": ["column"]
        }
    },
    {
        "name": "plot_bar",
        "description": "Generate bar chart from a categorical column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"}
            },
            "required": ["column"]
        }
    },
    {
        "name": "plot_correlation",
        "description": "Generate correlation heatmap for numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

chart_tool_functions = {
    "plot_histogram": plot_histogram,
    "plot_bar": plot_bar,
    "plot_correlation": plot_correlation,
}


# ============================================
# Block 7: ML Tools
# ============================================
def preprocess_df(df):
    """Preprocess dataframe for ML."""
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = df[col].fillna(df[col].median())
    return df, encoders


def train_model(model_type: str):
    """Train an ML model."""
    global train_df, MODELS, ENCODERS, FEATURES
    
    if train_df is None:
        return {"error": "No dataset loaded."}
    
    target = "Survived"
    if target not in train_df.columns:
        return {"error": f"'{target}' column not found in dataset."}
    
    df_processed, encs = preprocess_df(train_df)
    cols_to_drop = ['Name', 'Cabin', 'Ticket']
    df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns])
    
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    feature_cols = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=120)
    else:
        return {"error": "Invalid model_type. Use 'logistic' or 'random_forest'."}
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    model_name = f"{model_type}_{len(MODELS) + 1}"
    MODELS[model_name] = model
    ENCODERS[model_name] = encs
    FEATURES[model_name] = feature_cols
    
    return {
        "model_name": model_name,
        "accuracy": float(acc)
    }


def predict(model_name: str, input_values: dict):
    """Make predictions using a trained model."""
    if model_name not in MODELS:
        return {"error": f"Model '{model_name}' does not exist."}
    
    model = MODELS[model_name]
    encs = ENCODERS[model_name]
    feature_cols = FEATURES[model_name]
    
    template_df = train_df.copy()
    template_df, _ = preprocess_df(template_df)
    template_df = template_df[feature_cols]
    
    sample = []
    for col in feature_cols:
        if col in input_values:
            val = input_values[col]
            if col in encs:
                try:
                    val = encs[col].transform([str(val)])[0]
                except ValueError:
                    val = 0
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = template_df[col].median()
        else:
            if col in encs:
                val = 0
            else:
                val = template_df[col].median()
        sample.append(val)
    
    X = np.array(sample).reshape(1, -1)
    pred = model.predict(X)[0]
    
    confidence = None
    try:
        confidence = model.predict_proba(X)[0].max()
    except AttributeError:
        pass
    
    result = {"prediction": int(pred)}
    if confidence is not None:
        result["confidence"] = float(confidence)
    return result


def list_models():
    """List all trained models."""
    return {"models": list(MODELS.keys())}


ml_tool_declarations = [
    {
        "name": "train_model",
        "description": "Train an ML model (logistic or random_forest). Returns model name and accuracy.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_type": {"type": "string", "enum": ["logistic", "random_forest"]}
            },
            "required": ["model_type"]
        }
    },
    {
        "name": "predict",
        "description": "Predict output using a trained model. Requires model_name and input_values (dict).",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "input_values": {
                    "type": "object",
                    "description": "Dictionary of feature names and their values."
                }
            },
            "required": ["model_name", "input_values"]
        }
    },
    {
        "name": "list_models",
        "description": "List all trained models available in memory.",
        "parameters": {"type": "object", "properties": {}}
    }
]


# ============================================
# Block 8: SQL Tools
# ============================================
def init_sql_database():
    """Initialize SQLite database with current dataset."""
    global conn, cursor, train_df
    
    if train_df is None:
        return False
    
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create table based on train_df structure
    # This is a simplified version - you may need to adjust based on your dataset
    train_df.to_sql("dataset", conn, if_exists="replace", index=False)
    print("✓ Dataset loaded into in-memory SQLite database.")
    return True


def sql_query(query: str, limit: int = 20):
    """Execute a SQL SELECT query."""
    global conn
    
    if conn is None:
        return {"error": "SQL database not initialized. Load a dataset first."}
    
    query_lower = query.strip().lower()
    if not query_lower.startswith("select") and not query_lower.startswith("pragma"):
        return {"error": "Only SELECT and PRAGMA queries are allowed."}
    
    try:
        df = pd.read_sql_query(query, conn)
        if limit and not query_lower.startswith("pragma"):
            df = df.head(limit)
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


sql_tool_declaration = {
    "name": "sql_query",
    "description": "Run a SQL SELECT query on the dataset.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The SQL SELECT query to execute."},
            "limit": {"type": "integer", "description": "Maximum number of rows to return."}
        },
        "required": ["query"]
    }
}


# ============================================
# Block 9: Report Generation
# ============================================
REPORT_DIR = "./reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def plot_to_base64(plot_func, *args, **kwargs):
    """Generate plot and return base64-encoded image."""
    buf = BytesIO()
    plt.figure()
    plot_func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_report(title="Data Report", include_ml=False, columns=None, report_filename=None, generate_pdf=False):
    """Generate HTML/PDF report."""
    global train_df, MODELS, FEATURES
    
    if train_df is None:
        return {"error": "No dataset loaded."}
    
    current_report_columns = train_df.columns
    if columns:
        current_report_columns = [col for col in train_df.columns if col in columns]
    
    column_stats = {col: describe_column(col) for col in current_report_columns if pd.api.types.is_numeric_dtype(train_df[col])}
    missing = missing_values()
    
    hist_imgs = {}
    for col in [c for c in current_report_columns if pd.api.types.is_numeric_dtype(train_df[c])]:
        hist_imgs[col] = plot_to_base64(lambda c=col: plt.hist(train_df[c].dropna(), bins=20))
    
    bar_imgs = {}
    for col in [c for c in current_report_columns if not pd.api.types.is_numeric_dtype(train_df[c])]:
        bar_imgs[col] = plot_to_base64(lambda c=col: train_df[c].value_counts().plot(kind='bar'))
    
    corr_img = plot_to_base64(lambda: sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="coolwarm"))
    
    ml_results = []
    if include_ml and MODELS:
        try:
            df_processed, _ = preprocess_df(train_df)
            target_col = "Survived"
            if target_col in df_processed.columns:
                X_full = df_processed.drop(columns=[target_col])
                y_full = df_processed[target_col]
                for model_name, model in MODELS.items():
                    feature_cols = FEATURES.get(model_name)
                    X_used = X_full
                    if feature_cols:
                        missing_cols = [col for col in feature_cols if col not in X_full.columns]
                        if missing_cols:
                            ml_results.append({
                                "model_name": model_name,
                                "accuracy": f"Skipped (missing columns: {missing_cols})"
                            })
                            continue
                        X_used = X_full[feature_cols]
                    try:
                        y_pred = model.predict(X_used)
                        accuracy = accuracy_score(y_full, y_pred)
                        ml_results.append({
                            "model_name": model_name,
                            "accuracy": round(float(accuracy), 4)
                        })
                    except Exception as model_error:
                        ml_results.append({
                            "model_name": model_name,
                            "accuracy": f"Error: {model_error}"
                        })
            else:
                ml_results.append({"model_name": "N/A", "accuracy": "Target column 'Survived' not found."})
        except Exception as ml_exception:
            ml_results.append({"model_name": "N/A", "accuracy": f"ML summary failed: {ml_exception}"})
    
    template_str = """
    <html>
    <head><title>{{ title }}</title></head>
    <body>
        <h1>{{ title }}</h1>
        <h2>Column Statistics</h2>
        {% for col, stats in column_stats.items() %}
            <h3>{{ col }}</h3>
            <pre>{{ stats }}</pre>
        {% endfor %}
        <h2>Missing Values</h2>
        <pre>{{ missing }}</pre>
        <h2>Histograms</h2>
        {% for col, img in hist_imgs.items() %}
            <h4>{{ col }}</h4>
            <img src="data:image/png;base64,{{ img }}">
        {% endfor %}
        <h2>Bar Charts</h2>
        {% for col, img in bar_imgs.items() %}
            <h4>{{ col }}</h4>
            <img src="data:image/png;base64,{{ img }}">
        {% endfor %}
        <h2>Correlation Heatmap</h2>
        <img src="data:image/png;base64,{{ corr_img }}">
        
        {% if ml_results %}
        <h2>ML Model Performance</h2>
        <ul>
        {% for res in ml_results %}
            <li>{{ res.model_name }} - Accuracy: {{ res.accuracy }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </body>
    </html>
    """
    
    template = Template(template_str)
    html_content = template.render(
        title=title,
        column_stats=column_stats,
        missing=missing,
        hist_imgs=hist_imgs,
        bar_imgs=bar_imgs,
        corr_img=corr_img,
        ml_results=ml_results
    )
    
    final_report_filename_html = report_filename if report_filename else f"report_{uuid.uuid4().hex[:6]}.html"
    filename_path_html = os.path.join(REPORT_DIR, final_report_filename_html)
    with open(filename_path_html, "w") as f:
        f.write(html_content)
    
    report_paths = {"html_report_file": filename_path_html}
    
    if generate_pdf:
        try:
            pdf_filename = final_report_filename_html.replace(".html", ".pdf")
            filename_path_pdf = os.path.join(REPORT_DIR, pdf_filename)
            with open(filename_path_pdf, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file, encoding="UTF-8")
            if pisa_status.err:
                report_paths["pdf_generation_error"] = "Failed to generate PDF report."
            else:
                report_paths["pdf_report_file"] = filename_path_pdf
        except Exception as e:
            report_paths["pdf_generation_error"] = str(e)
    
    return report_paths


report_tool_declarations = [
    {
        "name": "generate_report",
        "description": "Generate automated HTML report for dataset including stats, charts, and optional ML results. Can also generate a PDF version.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "include_ml": {"type": "boolean"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "report_filename": {"type": "string"},
                "generate_pdf": {"type": "boolean", "default": False}
            },
            "required": []
        }
    }
]


# ============================================
# Block 10: Multi-Dataset Support
# ============================================
def load_dataset(name: str, filepath: str):
    """Load a CSV dataset and make it active."""
    global _ALL_LOADED_DFS, _CURRENT_DF_NAME, train_df
    
    if not os.path.exists(filepath):
        return {"error": f"File '{filepath}' does not exist."}
    
    try:
        df = pd.read_csv(filepath)
        _ALL_LOADED_DFS[name] = df
        _CURRENT_DF_NAME = name
        train_df = df
        init_sql_database()  # Reinitialize SQL with new dataset
        return {"message": f"Dataset '{name}' loaded with {len(df)} rows and {len(df.columns)} columns. Now active."}
    except Exception as e:
        return {"error": f"Failed to load dataset '{name}': {str(e)}"}


def switch_dataset(name: str):
    """Switch the current active dataset."""
    global _ALL_LOADED_DFS, _CURRENT_DF_NAME, train_df
    
    if name not in _ALL_LOADED_DFS:
        return {"error": f"Dataset '{name}' is not loaded. Load it first using load_dataset()."}
    
    _CURRENT_DF_NAME = name
    train_df = _ALL_LOADED_DFS[name]
    init_sql_database()  # Reinitialize SQL with new dataset
    return {"message": f"Switched to dataset '{name}'. '{name}' is now active."}


def list_datasets():
    """List all loaded datasets."""
    return {"loaded_datasets": list(_ALL_LOADED_DFS.keys()), "current_active_dataset": _CURRENT_DF_NAME}


def _execute_kaggle_command_download(command_list):
    """Execute Kaggle download command."""
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


def download_kaggle_dataset(kaggle_dataset_id: str, download_path: str = "./data", extract_dir: str = "./data"):
    """Download and extract a Kaggle dataset."""
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    zip_filename = kaggle_dataset_id.split('/')[-1].replace('-', '_') + '.zip'
    zip_filepath = os.path.join(download_path, zip_filename)
    
    print(f"Attempting to download Kaggle dataset '{kaggle_dataset_id}' to {download_path}...")
    download_output = _execute_kaggle_command_download([
        "datasets", "download",
        "-d", kaggle_dataset_id,
        "-p", download_path
    ])
    
    if "Error:" in download_output:
        return {"error": f"Kaggle download failed: {download_output}"}
    
    print(f"Unzipping '{zip_filename}' to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_filepath, "r") as z:
            z.extractall(extract_dir)
        os.remove(zip_filepath)
        extracted_files = os.listdir(extract_dir)
        return {"message": f"Dataset '{kaggle_dataset_id}' downloaded and extracted. Files: {extracted_files}"}
    except Exception as e:
        return {"error": f"Failed to unzip dataset: {str(e)}"}


dataset_tool_declarations = [
    {
        "name": "load_dataset",
        "description": "Load a CSV dataset by providing a name and file path. This dataset will become the active dataset for analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A unique name for the dataset."},
                "filepath": {"type": "string", "description": "The file path to the CSV dataset (e.g., './data/iris.csv')."}
            },
            "required": ["name", "filepath"]
        }
    },
    {
        "name": "switch_dataset",
        "description": "Switch the current active dataset for analysis to a previously loaded dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the dataset to switch to (must be already loaded)."}
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_datasets",
        "description": "List all loaded datasets and indicate the currently active one.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "download_kaggle_dataset",
        "description": "Downloads and extracts a Kaggle dataset by ID into the local './data' directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "kaggle_dataset_id": {"type": "string", "description": "The Kaggle dataset ID (e.g., 'uciml/iris')."},
                "download_path": {"type": "string", "description": "Optional: Directory for the zip file (defaults to './data')."},
                "extract_dir": {"type": "string", "description": "Optional: Directory for extracted files (defaults to './data')."}
            },
            "required": ["kaggle_dataset_id"]
        }
    }
]


# ============================================
# Global Tool Management
# ============================================
def _remove_default_from_dict(d):
    """Recursively remove 'default' keys from dict."""
    if not isinstance(d, dict):
        return d
    new_dict = {}
    for k, v in d.items():
        if k == 'default':
            continue
        if isinstance(v, dict):
            new_dict[k] = _remove_default_from_dict(v)
        elif isinstance(v, list):
            new_dict[k] = [_remove_default_from_dict(item) for item in v]
        else:
            new_dict[k] = v
    return new_dict


def clean_declarations(declarations_list):
    """Clean function declarations for API."""
    cleaned = []
    for decl in declarations_list:
        decl_copy = decl.copy()
        decl_copy.pop('prompt_engineering', None)
        if 'parameters' in decl_copy:
            decl_copy['parameters'] = _remove_default_from_dict(decl_copy['parameters'])
        cleaned.append(decl_copy)
    return cleaned


def rebuild_tool_schema():
    """Rebuild the complete tool schema from all components."""
    all_declarations_to_combine = []
    
    all_declarations_to_combine.extend(BASE_FUNCTION_DECLARATIONS)
    all_declarations_to_combine.extend(chart_function_declarations)
    all_declarations_to_combine.extend(ml_tool_declarations)
    all_declarations_to_combine.append(sql_tool_declaration)
    all_declarations_to_combine.extend(report_tool_declarations)
    all_declarations_to_combine.extend(dataset_tool_declarations)
    
    ALL_FUNCTION_DECLARATIONS = clean_declarations(all_declarations_to_combine)
    
    ALL_TOOL_FUNCTIONS = {}
    ALL_TOOL_FUNCTIONS.update(BASE_TOOL_FUNCTIONS)
    ALL_TOOL_FUNCTIONS.update(chart_tool_functions)
    ALL_TOOL_FUNCTIONS['train_model'] = train_model
    ALL_TOOL_FUNCTIONS['predict'] = predict
    ALL_TOOL_FUNCTIONS['list_models'] = list_models
    ALL_TOOL_FUNCTIONS['sql_query'] = sql_query
    ALL_TOOL_FUNCTIONS['generate_report'] = generate_report
    ALL_TOOL_FUNCTIONS['load_dataset'] = load_dataset
    ALL_TOOL_FUNCTIONS['switch_dataset'] = switch_dataset
    ALL_TOOL_FUNCTIONS['list_datasets'] = list_datasets
    ALL_TOOL_FUNCTIONS['download_kaggle_dataset'] = download_kaggle_dataset
    
    tools_schema = [{
        "function_declarations": ALL_FUNCTION_DECLARATIONS
    }]
    
    return ALL_FUNCTION_DECLARATIONS, ALL_TOOL_FUNCTIONS, tools_schema


# ============================================
# Agent Functions
# ============================================
def initialize_agent(api_key: str):
    """Initialize the Gemini model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model


def tool_executor(function_name: str, arguments: dict):
    """Dynamically execute tool functions."""
    _, ALL_TOOL_FUNCTIONS, _ = rebuild_tool_schema()
    
    if function_name in ALL_TOOL_FUNCTIONS:
        func = ALL_TOOL_FUNCTIONS[function_name]
        result = func(**arguments)
        if isinstance(result, str):
            return {"error_message": result}
        elif isinstance(result, list):
            return {"data": result}
        return result
    return {"error": f"Unknown tool: {function_name}", "function_name": function_name}


def ask_agent(prompt: str, model):
    """Send user query to Gemini with tools enabled."""
    _, _, tools_schema = rebuild_tool_schema()
    
    response = model.generate_content(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        tools=tools_schema
    )
    
    tool_call = None
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call'):
                tool_call = part.function_call
                break
    
    if tool_call:
        name = tool_call.name
        args = tool_call.args or {}
        
        tool_output = tool_executor(name, args)
        
        followup = model.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": prompt}]},
                {
                    "role": "function",
                    "parts": [{"function_response": {"name": name, "response": tool_output}}]
                }
            ]
        )
        return followup.text
    
    return response.text

