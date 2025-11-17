"""
Kaggle Time-Saver Agent - Streamlit Web Application
Web interface for the AI agent
"""

import streamlit as st
import os
import zipfile
from pathlib import Path
from modules.config import setup_environment
from modules.agent import (
    initialize_agent, ask_agent, load_dataset, switch_dataset, 
    list_datasets, download_kaggle_dataset, train_df, generate_report
)
from modules.dataset_loader import (
    download_competition_dataset, load_dataset_from_path, 
    download_kaggle_dataset_simple
)

# Page configuration
st.set_page_config(
    page_title="Kaggle Time-Saver Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'report_paths' not in st.session_state:
    st.session_state.report_paths = None


def initialize_agent_session():
    """Initialize the agent if not already done."""
    if not st.session_state.agent_initialized:
        try:
            with st.spinner("Initializing AI agent..."):
                config = setup_environment()
                model = initialize_agent(config["google_api_key"])
                st.session_state.config = config
                st.session_state.model = model
                st.session_state.agent_initialized = True
                st.success("✓ Agent initialized successfully!")
                return True
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.info("Please check your .env file or environment variables.")
            return False
    return True


def open_report_in_new_tab(report_path, report_type="html"):
    """Generate JavaScript to open report in new tab."""
    if report_type == "html":
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Use Streamlit's components to display HTML
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        # For PDF, we'll provide a download link that opens in new tab
        with open(report_path, 'rb') as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )


def main():
    """Main Streamlit application."""
    
    # Title and header
    st.title("📊 Kaggle Time-Saver Agent")
    st.markdown("""
    An AI-powered agent that helps you analyze Kaggle datasets quickly:
    - 🤖 Ask natural language questions about your data
    - 📊 Explore datasets (columns, missing values, statistics)
    - 📈 Generate charts and visualizations
    - 🧠 Train ML models and make predictions
    - 🗄️ Run SQL queries
    - 📄 Generate HTML/PDF reports
    - 📁 Manage multiple datasets
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["🏠 Home", "📁 Load Dataset", "🤖 AI Assistant", "📄 Generate Report", "⚙️ Settings"]
    )
    
    # Initialize agent
    if not initialize_agent_session():
        st.stop()
    
    # Route to different pages
    if page == "🏠 Home":
        home_page()
    elif page == "📁 Load Dataset":
        load_dataset_page()
    elif page == "🤖 AI Assistant":
        ai_assistant_page()
    elif page == "📄 Generate Report":
        report_page()
    elif page == "⚙️ Settings":
        settings_page()


def home_page():
    """Home page with overview."""
    st.header("Welcome to Kaggle Time-Saver Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Dataset")
        if train_df is not None:
            st.success(f"✓ Dataset loaded: {st.session_state.current_dataset or 'Active Dataset'}")
            st.metric("Rows", len(train_df))
            st.metric("Columns", len(train_df.columns))
            st.dataframe(train_df.head(10), use_container_width=True)
        else:
            st.info("No dataset loaded. Go to 'Load Dataset' to get started.")
    
    with col2:
        st.subheader("🚀 Quick Actions")
        if train_df is not None:
            if st.button("📋 List Columns"):
                st.write(list(train_df.columns))
            
            if st.button("📊 Show Statistics"):
                st.dataframe(train_df.describe(include="all").transpose())
            
            if st.button("🔍 Check Missing Values"):
                missing_df = train_df.isnull().sum().reset_index()
                missing_df.columns = ["Column", "Missing Values"]
                missing_df["Percent"] = (missing_df["Missing Values"] / len(train_df) * 100).round(2)
                st.dataframe(missing_df)
        else:
            st.info("Load a dataset first to use quick actions.")


def load_dataset_page():
    """Page for loading datasets."""
    st.header("📁 Load Dataset")
    
    tab1, tab2, tab3 = st.tabs(["Upload File", "From Kaggle", "From Local Path"])
    
    with tab1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset CSV file"
        )
        
        if uploaded_file is not None:
            dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.replace('.csv', ''))
            
            if st.button("Load Uploaded Dataset"):
                # Save uploaded file temporarily
                temp_path = os.path.join("./data", uploaded_file.name)
                os.makedirs("./data", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                result = load_dataset(dataset_name or "uploaded_dataset", temp_path)
                if "error" not in result:
                    st.session_state.current_dataset = dataset_name or "uploaded_dataset"
                    st.success(result.get("message", "Dataset loaded successfully!"))
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed to load dataset"))
    
    with tab2:
        st.subheader("Download from Kaggle")
        
        download_option = st.radio(
            "Download Type",
            ["Competition Dataset", "Regular Dataset"]
        )
        
        if download_option == "Competition Dataset":
            competition_name = st.text_input("Competition Name", value="titanic", help="e.g., titanic")
            if st.button("Download Competition Dataset"):
                with st.spinner("Downloading dataset..."):
                    if download_competition_dataset(competition_name):
                        st.success("✓ Dataset downloaded successfully!")
                        # Try to load train.csv
                        train_path = f"./data/train.csv"
                        if os.path.exists(train_path):
                            result = load_dataset(competition_name, train_path)
                            if "error" not in result:
                                st.session_state.current_dataset = competition_name
                                st.success(result.get("message", "Dataset loaded!"))
                                st.rerun()
        else:
            dataset_id = st.text_input("Kaggle Dataset ID", value="uciml/iris", help="e.g., uciml/iris")
            dataset_name = st.text_input("Dataset Name", value=dataset_id.split('/')[-1] if '/' in dataset_id else "dataset")
            
            if st.button("Download Dataset"):
                with st.spinner("Downloading dataset..."):
                    download_kaggle_dataset_simple(dataset_id)
                    # Find CSV files
                    csv_files = [f for f in os.listdir("./data") if f.endswith('.csv')]
                    if csv_files:
                        filepath = os.path.join("./data", csv_files[0])
                        result = load_dataset(dataset_name, filepath)
                        if "error" not in result:
                            st.session_state.current_dataset = dataset_name
                            st.success(result.get("message", "Dataset loaded!"))
                            st.rerun()
    
    with tab3:
        st.subheader("Load from Local Path")
        file_path = st.text_input("File Path", value="./data/train.csv", help="Path to your CSV file")
        dataset_name = st.text_input("Dataset Name", value="local_dataset")
        
        if st.button("Load Dataset"):
            if os.path.exists(file_path):
                result = load_dataset(dataset_name, file_path)
                if "error" not in result:
                    st.session_state.current_dataset = dataset_name
                    st.success(result.get("message", "Dataset loaded successfully!"))
                    st.rerun()
            else:
                st.error(f"File not found: {file_path}")
    
    # Detect zipped datasets in data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    zip_files = [f.name for f in data_dir.glob("*.zip")]
    if zip_files:
        st.subheader("📦 Zipped Datasets Detected")
        selected_zip = st.selectbox("Select a ZIP file to inspect", zip_files)
        if selected_zip:
            zip_path = data_dir / selected_zip
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                csv_members = [member for member in zip_ref.namelist() if member.lower().endswith(".csv")]
            if not csv_members:
                st.warning("No CSV files found inside this ZIP.")
            else:
                preferred_csv = next((name for name in csv_members if "train" in Path(name).stem.lower()), csv_members[0])
                selected_csv = st.selectbox(
                    "Select which CSV inside the ZIP to load",
                    options=csv_members,
                    index=csv_members.index(preferred_csv),
                    key="zip_internal_csv"
                )
                default_name = Path(selected_csv).stem
                zip_dataset_name = st.text_input("Dataset name for extracted CSV", value=default_name, key="zip_dataset_name")
                if st.button("Extract & Load ZIP Dataset"):
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extract(selected_csv, data_dir)
                        extracted_csv_path = data_dir / selected_csv
                        if extracted_csv_path.exists():
                            result = load_dataset(zip_dataset_name or default_name, str(extracted_csv_path))
                            if "error" not in result:
                                st.session_state.current_dataset = zip_dataset_name or default_name
                                st.success(result.get("message", "Dataset extracted and loaded!"))
                                st.rerun()
                            else:
                                st.error(result.get("error"))
                        else:
                            st.warning("Failed to locate extracted CSV file.")
                    except Exception as e:
                        st.error(f"Failed to extract ZIP file: {e}")
    else:
        st.info("Place ZIP files (e.g., titanic.zip) into the ./data folder to have them detected here.")
    
    # Show loaded datasets
    st.subheader("📋 Loaded Datasets")
    datasets_info = list_datasets()
    if datasets_info.get("loaded_datasets"):
        st.json(datasets_info)
        current = datasets_info.get("current_active_dataset")
        if current:
            st.info(f"Current active dataset: **{current}**")
        
        # Switch dataset
        if len(datasets_info.get("loaded_datasets", [])) > 1:
            switch_to = st.selectbox("Switch to dataset:", datasets_info["loaded_datasets"])
            if st.button("Switch Dataset"):
                result = switch_dataset(switch_to)
                if "error" not in result:
                    st.session_state.current_dataset = switch_to
                    st.success(result.get("message", "Switched successfully!"))
                    st.rerun()
    else:
        st.info("No datasets loaded yet.")


def ai_assistant_page():
    """AI Assistant page."""
    st.header("🤖 AI Assistant")
    
    if train_df is None:
        st.warning("⚠️ Please load a dataset first from the 'Load Dataset' page.")
        return
    
    st.info("💡 Ask questions about your dataset in natural language. The AI can analyze data, create charts, train models, and more!")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your dataset..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_agent(prompt, st.session_state.model)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def report_page():
    """Report generation page."""
    st.header("📄 Generate Report")
    
    if train_df is None:
        st.warning("⚠️ Please load a dataset first from the 'Load Dataset' page.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", value="Data Analysis Report")
        include_ml = st.checkbox("Include ML Model Results", value=False)
        generate_pdf = st.checkbox("Generate PDF Version", value=True)
    
    with col2:
        # Column selection
        all_columns = list(train_df.columns)
        selected_columns = st.multiselect(
            "Select Columns (leave empty for all)",
            all_columns,
            default=[]
        )
        columns = selected_columns if selected_columns else None
    
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                report_paths = generate_report(
                    title=report_title,
                    include_ml=include_ml,
                    columns=columns,
                    generate_pdf=generate_pdf
                )
                
                st.session_state.report_paths = report_paths
                
                if "error" in report_paths:
                    st.error(report_paths["error"])
                else:
                    st.success("✅ Report generated successfully!")
                    
                    # HTML Report
                    if "html_report_file" in report_paths:
                        html_path = report_paths["html_report_file"]
                        st.subheader("📄 HTML Report")
                        
                        # Read HTML content
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Create a URL-friendly path for the report
                        import base64
                        html_base64 = base64.b64encode(html_content.encode()).decode()
                        
                        # Display HTML preview
                        st.components.v1.html(html_content, height=600, scrolling=True)
                        
                        # Auto-open in new tab using JavaScript
                        import urllib.parse
                        html_encoded = urllib.parse.quote(html_content)
                        st.markdown(
                            f"""
                            <script>
                                var htmlContent = decodeURIComponent('{html_encoded}');
                                var blob = new Blob([htmlContent], {{type: 'text/html'}});
                                var url = URL.createObjectURL(blob);
                                window.open(url, '_blank');
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Download button
                        with open(html_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download HTML Report",
                                data=f.read(),
                                file_name=os.path.basename(html_path),
                                mime="text/html"
                            )
                    
                    # PDF Report
                    if "pdf_report_file" in report_paths:
                        st.subheader("📄 PDF Report")
                        pdf_path = report_paths["pdf_report_file"]
                        
                        # Read PDF data
                        with open(pdf_path, 'rb') as f:
                            pdf_data = f.read()
                        
                        # Display PDF preview
                        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" style="border:none;"></iframe>'
                        st.components.v1.html(pdf_display, height=600)
                        
                        # Auto-open PDF in new tab
                        st.markdown(
                            f"""
                            <script>
                                var pdfWindow = window.open('', '_blank');
                                pdfWindow.document.write('<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100%" style="border:none;"></iframe>');
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                    
                    if "pdf_generation_error" in report_paths:
                        st.warning(f"PDF generation had an issue: {report_paths['pdf_generation_error']}")
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def settings_page():
    """Settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("Environment Status")
    if st.session_state.config:
        st.success("✓ Environment variables loaded")
        st.json({
            "Google API Key": "✓ Configured" if st.session_state.config.get("google_api_key") else "✗ Missing",
            "Kaggle Username": st.session_state.config.get("kaggle_username", "Not set"),
            "Kaggle Key": "✓ Configured" if st.session_state.config.get("kaggle_key") else "✗ Missing"
        })
    else:
        st.error("Environment not configured")
    
    st.subheader("Agent Status")
    if st.session_state.agent_initialized:
        st.success("✓ AI Agent initialized and ready")
    else:
        st.error("✗ AI Agent not initialized")
    
    if st.button("🔄 Reinitialize Agent"):
        st.session_state.agent_initialized = False
        st.rerun()


if __name__ == "__main__":
    main()

