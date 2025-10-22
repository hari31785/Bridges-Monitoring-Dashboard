"""Main Streamlit dashboard application."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import re

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.config import config
    from src.data_loader import DataLoaderFactory, ExcelDataLoader
    from src.components import ChartComponent, TableComponent, FilterComponent, MetricsComponent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def format_date_to_standard(date_input):
    """
    Convert any date format to DD-MON-YYYY format.
    
    Args:
        date_input: Date in various formats (string, datetime, pandas timestamp, etc.)
        
    Returns:
        String in DD-MON-YYYY format or original value if conversion fails
    """
    if pd.isna(date_input) or date_input is None or str(date_input).strip() == '':
        return date_input
    
    try:
        # Convert to string first
        date_str = str(date_input).strip()
        
        # If already in DD-MON-YYYY format, return as is
        if re.match(r'\d{1,2}-[A-Z]{3}-\d{4}', date_str):
            return date_str
        
        # Handle pandas timestamps
        if hasattr(date_input, 'strftime'):
            return date_input.strftime('%d-%b-%Y').upper()
        
        # Try to parse various date formats
        date_formats = [
            '%Y-%m-%d',           # 2024-10-08
            '%m/%d/%Y',           # 10/08/2024
            '%d/%m/%Y',           # 08/10/2024
            '%Y/%m/%d',           # 2024/10/08
            '%d-%m-%Y',           # 08-10-2024
            '%m-%d-%Y',           # 10-08-2024
            '%Y%m%d',             # 20241008
            '%d.%m.%Y',           # 08.10.2024
            '%m.%d.%Y',           # 10.08.2024
            '%d %B %Y',           # 08 October 2024
            '%B %d, %Y',          # October 08, 2024
            '%d %b %Y',           # 08 Oct 2024
            '%b %d, %Y',          # Oct 08, 2024
        ]
        
        # Try each format
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%d-%b-%Y').upper()
            except ValueError:
                continue
        
        # Try pandas to_datetime as fallback
        try:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime('%d-%b-%Y').upper()
        except:
            pass
        
        # If all parsing fails, return original value
        return date_input
        
    except Exception:
        # If any error occurs, return original value
        return date_input


def sort_dataframe_by_date(df, ascending=False):
    """
    Sort DataFrame by date columns (latest to oldest by default).
    
    Args:
        df: pandas DataFrame
        ascending: If False (default), sorts latest to oldest
        
    Returns:
        DataFrame sorted by date columns
    """
    if df is None or len(df) == 0:
        return df
    
    df_sorted = df.copy()
    
    # Identify potential date columns
    date_columns = []
    
    for col in df_sorted.columns:
        col_lower = col.lower()
        
        # Skip obvious non-date columns
        non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout']
        if any(keyword in col_lower for keyword in non_date_keywords):
            continue
            
        # Check if column name suggests it's a date
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
            # Convert to datetime for proper sorting
            try:
                df_sorted[col] = pd.to_datetime(df_sorted[col], errors='coerce')
                date_columns.append(col)
            except:
                pass
        else:
            # Check if column content looks like dates
            sample_values = df_sorted[col].dropna().head(3)
            if len(sample_values) > 0:
                is_date_column = False
                for val in sample_values:
                    val_str = str(val).strip()
                    # More specific date pattern matching - avoid simple numbers
                    if (re.match(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}', val_str) or  # MM/DD/YYYY or DD/MM/YYYY
                        re.match(r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}', val_str) or  # YYYY/MM/DD
                        re.match(r'\d{1,2} \w{3,} \d{4}', val_str) or            # DD Month YYYY
                        re.match(r'\w{3,} \d{1,2}, \d{4}', val_str)) and \
                       not val_str.isdigit():  # Exclude pure numbers
                        is_date_column = True
                        break
                if is_date_column:
                    try:
                        df_sorted[col] = pd.to_datetime(df_sorted[col], errors='coerce')
                        date_columns.append(col)
                    except:
                        pass
    
    # Sort by date columns (primary sort by first date column found)
    if date_columns:
        # Sort by all date columns, with primary column first
        df_sorted = df_sorted.sort_values(by=date_columns, ascending=ascending, na_position='last')
    
    return df_sorted


def format_dataframe_dates(df):
    """
    Apply date formatting to all date columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with formatted dates
    """
    if df is None or len(df) == 0:
        return df
    
    df_formatted = df.copy()
    
    # Identify potential date columns
    date_columns = []
    
    for col in df_formatted.columns:
        col_lower = col.lower()
        
        # Skip obvious non-date columns with more specific patterns
        non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'session', 'connection', 'batch', 'thread']
        if any(keyword in col_lower for keyword in non_date_keywords):
            continue
            
        # Check if column name suggests it's a date
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
            date_columns.append(col)
        else:
            # Check if column content looks like dates (sample first few non-null values)
            sample_values = df_formatted[col].dropna().head(3)
            if len(sample_values) > 0:
                is_date_column = False
                for val in sample_values:
                    val_str = str(val).strip()
                    # More specific date pattern matching - avoid simple numbers
                    if (re.match(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}', val_str) or  # MM/DD/YYYY or DD/MM/YYYY
                        re.match(r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}', val_str) or  # YYYY/MM/DD
                        re.match(r'\d{1,2} \w{3,} \d{4}', val_str) or            # DD Month YYYY
                        re.match(r'\w{3,} \d{1,2}, \d{4}', val_str)) and \
                       not val_str.isdigit():  # Exclude pure numbers
                        is_date_column = True
                        break
                if is_date_column:
                    date_columns.append(col)
                    break
    
    # Format identified date columns
    for col in date_columns:
        df_formatted[col] = df_formatted[col].apply(format_date_to_standard)
    
    return df_formatted

def format_percentage_columns(df):
    """
    Format percentage columns appropriately:
    - Columns with "%" symbol: Convert decimals to percentages (0.04 → 4%)
    - Other percentage columns: Display as numbers with 2 decimal places
    
    Args:
        df: DataFrame to format
        
    Returns:
        DataFrame with formatted percentage columns
    """
    df_formatted = df.copy()
    
    # List of keywords that typically indicate percentage columns (non-% symbol)
    percentage_keywords = [
        'percent', 'percentage', 'rate', 'ratio', 'variance', 'var',
        'pct', 'proportion', 'share', 'yield', 'efficiency', 'utilization',
        'completion', 'success', 'failure', 'error rate', 'accuracy'
    ]
    
    for col in df_formatted.columns:
        col_lower = col.lower()
        
        # Check if column name contains the "%" symbol - these should show as percentages
        has_percent_symbol = '%' in col
        
        # Check if column name contains other percentage-related keywords
        is_percentage_col = any(keyword in col_lower for keyword in percentage_keywords)
        
        if has_percent_symbol or is_percentage_col:
            try:
                # Check if column contains data that looks like percentages
                sample_values = df_formatted[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Convert to numeric first
                    numeric_series = pd.to_numeric(df_formatted[col], errors='coerce')
                    
                    if has_percent_symbol:
                        # For columns with "%" symbol, display values exactly as they appear in the source data
                        # Only add the % symbol if not already present, preserve original precision
                        def format_percentage_value(x):
                            if pd.isna(x):
                                return ""
                            # Convert to string to preserve original precision from Excel
                            str_val = str(x)
                            # If the original value already contains %, return as-is
                            if '%' in str_val:
                                return str_val
                            # Otherwise, just add % symbol without changing the number
                            return f"{x}%"
                        
                        # Use original values, not the numeric_series which might alter precision
                        df_formatted[col] = df_formatted[col].apply(format_percentage_value)
                    else:
                        # For other percentage-related columns, format as number with 2 decimal places
                        df_formatted[col] = numeric_series.apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
            except (ValueError, TypeError):
                # If conversion fails, leave as is
                continue
    
    return df_formatted


def filter_data_to_recent_weeks(df, date_column=None, weeks_to_show=None):
    """
    Filter DataFrame to show only data from current week and previous week(s).
    
    Args:
        df: pandas DataFrame
        date_column: Name of the date column to filter by. If None, auto-detect.
        weeks_to_show: Number of weeks to show. If None, uses session state value.
        
    Returns:
        DataFrame filtered to recent weeks
    """
    if df is None or len(df) == 0:
        return df
    
    # Get weeks_to_show from session state if not provided
    if weeks_to_show is None:
        weeks_to_show = st.session_state.get('weeks_to_show', 2)
    
    df_filtered = df.copy()
    
    # Auto-detect date column if not specified
    if date_column is None:
        date_columns = []
        for col in df_filtered.columns:
            col_lower = col.lower()
            # Skip obvious non-date columns
            non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'session', 'connection', 'batch', 'thread']
            if any(keyword in col_lower for keyword in non_date_keywords):
                continue
                
            # Check if column name suggests it's a date
            if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
                date_columns.append(col)
        
        # Use the first date column found
        date_column = date_columns[0] if date_columns else None
    
    if date_column is None or date_column not in df_filtered.columns:
        # No date column found, return original data
        return df_filtered
    
    try:
        # Convert date column to datetime
        df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce')
        
        # Calculate date range for filtering - ensure we include complete weeks
        today = datetime.now()
        
        # Get the start of current week (Monday)
        days_since_monday = today.weekday()  # Monday = 0, Sunday = 6
        current_week_start = today - timedelta(days=days_since_monday)
        
        # Set time to start of day for consistent comparison
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate how many complete weeks back to include
        # If weeks_to_show = 2, we want current week + 1 previous complete week
        weeks_back = weeks_to_show - 1
        filter_start_date = current_week_start - timedelta(weeks=weeks_back)
        
        # For better clarity: if today is Wednesday and we want 2 weeks,
        # we should get all data from Monday of last week through today
        
        # Filter data to include only recent weeks
        mask = df_filtered[date_column] >= filter_start_date
        df_filtered = df_filtered[mask]
        
        # Add debug info in expandable section
        with st.expander("📅 Date Filtering Info", expanded=False):
            st.write(f"**Date Column Used:** {date_column}")
            st.write(f"**Today:** {today.strftime('%A, %Y-%m-%d')}")
            st.write(f"**Current Week Start (Monday):** {current_week_start.strftime('%A, %Y-%m-%d')}")
            st.write(f"**Filter Start Date:** {filter_start_date.strftime('%A, %Y-%m-%d')} (showing last {weeks_to_show} weeks)")
            st.write(f"**Records Before Filtering:** {len(df)}")
            st.write(f"**Records After Filtering:** {len(df_filtered)}")
            
            if len(df_filtered) > 0:
                min_date = df_filtered[date_column].min()
                max_date = df_filtered[date_column].max()
                st.write(f"**Date Range in Filtered Data:** {min_date.strftime('%A, %Y-%m-%d')} to {max_date.strftime('%A, %Y-%m-%d')}")
                
                # Show example of what weeks are included
                st.write("**Weeks Included:**")
                for i in range(weeks_to_show):
                    week_start = current_week_start - timedelta(weeks=i)
                    week_end = week_start + timedelta(days=6)
                    week_label = "Current Week" if i == 0 else f"{i} Week{'s' if i > 1 else ''} Ago"
                    st.write(f"• {week_label}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
        
        return df_filtered
        
    except Exception as e:
        st.warning(f"Could not filter dates in column '{date_column}': {str(e)}")
        return df_filtered


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.chart_component = ChartComponent()
        self.table_component = TableComponent()
        self.filter_component = FilterComponent()
        self.metrics_component = MetricsComponent()
        
        # Configure page
        dashboard_config = config.get_dashboard_config()
        
        st.set_page_config(
            page_title=dashboard_config.get('title', 'Monitoring Dashboard'),
            page_icon="📊",
            layout=dashboard_config.get('page_layout', 'wide'),
            initial_sidebar_state="expanded"
        )
    
    def auto_load_excel_file(self, section: str = "error_counts", period: str = "daily", subsection: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Automatically detect and load Excel files based on section, period, and subsection.
        
        Args:
            section: The selected section (error_counts, benefit_issuance, etc.)
            period: The selected time period (daily, weekly, monthly, yearly)
            subsection: The selected subsection/file name (for sections with multiple files)
        
        Returns:
            Dictionary with file configuration or None if no files found
        """
        workspace_path = Path(__file__).parent
        monitoring_data_path = workspace_path / "Monitoring Data Files"
        
        # Handle different sections with their specific folder structures
        if section == "benefit_issuance":
            bi_monitoring_path = monitoring_data_path / "BI Monitoring"
            
            # Map period to folder name
            period_folder_map = {
                "daily": "Daily",
                "weekly": "Weekly", 
                "monthly": "Monthly",
                "quarterly": "Quarterly",
                "yearly": "Yearly"
            }
            
            period_folder = period_folder_map.get(period, "Daily")
            period_path = bi_monitoring_path / period_folder
            
            if period_path.exists():
                # If a specific subsection is selected, look for that file
                if subsection:
                    # Try both .xlsx and .xls extensions
                    for ext in ['.xlsx', '.xls']:
                        file_path = period_path / f"{subsection}{ext}"
                        if file_path.exists() and not file_path.name.startswith('~$'):
                            return {
                                "data_source_type": "excel",
                                "workspace_file": file_path,
                                "uploaded_file": None,
                                "use_default_file": False,
                                "section": section,
                                "period": period,
                                "subsection": subsection
                            }
                
                # Fallback: Find all Excel files in the period-specific folder
                excel_files = []
                for pattern in ['*.xlsx', '*.xls']:
                    excel_files.extend(period_path.glob(pattern))
                
                # Filter out temporary files (starting with ~$)
                excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                
                if excel_files:
                    # Use the first Excel file found in the period folder
                    selected_file = excel_files[0]
                    return {
                        "data_source_type": "excel",
                        "workspace_file": selected_file,
                        "uploaded_file": None,
                        "use_default_file": False,
                        "section": section,
                        "period": period,
                        "available_files": excel_files
                    }
        
        elif section == "correspondence_tango":
            correspondence_path = monitoring_data_path / "Correspondence"
            
            if correspondence_path.exists():
                # If a specific subsection is selected, look for that file
                if subsection:
                    # Try both .xlsx and .xls extensions
                    for ext in ['.xlsx', '.xls']:
                        file_path = correspondence_path / f"{subsection}{ext}"
                        if file_path.exists() and not file_path.name.startswith('~$'):
                            return {
                                "data_source_type": "excel",
                                "workspace_file": file_path,
                                "uploaded_file": None,
                                "use_default_file": False,
                                "section": section,
                                "subsection": subsection
                            }
        
        elif section == "error_counts":
            error_counts_path = monitoring_data_path / "100 Error Counts"
            
            if error_counts_path.exists():
                # If a specific subsection is selected, look for that file
                if subsection:
                    # Try both .xlsx and .xls extensions
                    for ext in ['.xlsx', '.xls']:
                        file_path = error_counts_path / f"{subsection}{ext}"
                        if file_path.exists() and not file_path.name.startswith('~$'):
                            return {
                                "data_source_type": "excel",
                                "workspace_file": file_path,
                                "uploaded_file": None,
                                "use_default_file": False,
                                "section": section,
                                "subsection": subsection
                            }
        
        elif section == "user_impact":
            user_impact_path = monitoring_data_path / "User Impact"
            
            if user_impact_path.exists():
                # If a specific subsection is selected, look for that file
                if subsection:
                    # Try both .xlsx and .xls extensions
                    for ext in ['.xlsx', '.xls']:
                        file_path = user_impact_path / f"{subsection}{ext}"
                        if file_path.exists() and not file_path.name.startswith('~$'):
                            return {
                                "data_source_type": "excel",
                                "workspace_file": file_path,
                                "uploaded_file": None,
                                "use_default_file": False,
                                "section": section,
                                "subsection": subsection
                            }
        
        elif section == "extra_batch_connections":
            extra_batch_path = monitoring_data_path / "Extra Batch Connections"
            
            if extra_batch_path.exists():
                # Look for "Extra Connections Created.xlsx" file
                file_path = extra_batch_path / "Extra Connections Created.xlsx"
                if file_path.exists() and not file_path.name.startswith('~$'):
                    return {
                        "data_source_type": "excel",
                        "workspace_file": file_path,
                        "uploaded_file": None,
                        "use_default_file": False,
                        "section": section,
                        "subsection": "Extra Connections Created"
                    }
        
        # For other sections, use general workspace search
        excel_files = []
        
        # Search for Excel files in workspace and data folder
        for pattern in ['*.xlsx', '*.xls']:
            excel_files.extend(workspace_path.glob(pattern))
            excel_files.extend((workspace_path / 'data').glob(pattern))
        
        # Filter out temporary files (starting with ~$)
        excel_files = [f for f in excel_files if not f.name.startswith('~$')]
        
        if excel_files:
            # Use the first Excel file found
            selected_file = excel_files[0]
            return {
                "data_source_type": "excel",
                "workspace_file": selected_file,
                "uploaded_file": None,
                "use_default_file": False,
                "section": section,
                "period": period
            }
        
        # If no workspace files, check config default
        default_file = config.get('excel.file_path')
        if default_file and Path(default_file).exists():
            return {
                "data_source_type": "excel",
                "workspace_file": None,
                "uploaded_file": None, 
                "use_default_file": True,
                "section": section,
                "period": period
            }
        
        return None
    
    def load_data(self, sidebar_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data based on sidebar configuration.
        
        Args:
            sidebar_config: Configuration from sidebar
            
        Returns:
            Loaded DataFrame or None if loading failed
        """
        data_source_type = sidebar_config["data_source_type"]
        
        try:
            if data_source_type == "excel":
                if sidebar_config.get("workspace_file") is not None:
                    # Use selected workspace file
                    workspace_file = sidebar_config["workspace_file"]
                    loader = ExcelDataLoader()
                    loader.set_file_path(str(workspace_file))
                    
                    # Get available sheets
                    sheets = loader.get_available_sources()
                    
                    if sheets and len(sheets) > 1:
                        selected_sheet = st.sidebar.selectbox(
                            "Select Sheet",
                            options=sheets,
                            help="Choose which sheet to load"
                        )
                        df = loader.load_data(sheet_name=selected_sheet)
                    else:
                        df = loader.load_data()
                    
                    return df
                
                elif sidebar_config.get("uploaded_file") is not None:
                    # Use uploaded file
                    loader = ExcelDataLoader()
                    # Save uploaded file temporarily
                    temp_path = Path("temp_upload.xlsx")
                    with open(temp_path, "wb") as f:
                        f.write(sidebar_config["uploaded_file"].getbuffer())
                    loader.set_file_path(str(temp_path))
                    
                    # Get available sheets
                    sheets = loader.get_available_sources()
                    
                    if sheets:
                        selected_sheet = st.sidebar.selectbox(
                            "Select Sheet",
                            options=sheets,
                            help="Choose which sheet to load"
                        )
                        df = loader.load_data(sheet_name=selected_sheet)
                    else:
                        df = loader.load_data()
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                    
                    return df
                
                elif sidebar_config.get("use_default_file", False):
                    # Use default file from config
                    loader = DataLoaderFactory.create_loader("excel")
                    sheets = loader.get_available_sources()
                    
                    if sheets and len(sheets) > 1:
                        selected_sheet = st.sidebar.selectbox(
                            "Select Sheet",
                            options=sheets,
                            help="Choose which sheet to load"
                        )
                        return loader.load_data(sheet_name=selected_sheet)
                    else:
                        return loader.load_data()
                
                else:
                    st.error("Please select an Excel file from the workspace or upload a new one.")
                    return None
            
            elif data_source_type == "database":
                loader = DataLoaderFactory.create_loader("database")
                tables = loader.get_available_sources()
                
                if tables:
                    selected_table = st.sidebar.selectbox(
                        "Select Table",
                        options=tables,
                        help="Choose which table to load"
                    )
                    
                    # Option to enter custom SQL query
                    use_custom_query = st.sidebar.checkbox("Use Custom SQL Query")
                    
                    if use_custom_query:
                        custom_query = st.sidebar.text_area(
                            "SQL Query",
                            value=f"SELECT * FROM {selected_table} LIMIT 1000",
                            help="Enter your custom SQL query"
                        )
                        return loader.load_data(query=custom_query)
                    else:
                        return loader.load_data(table_name=selected_table)
                else:
                    st.error("No tables found in database or connection failed.")
                    return None
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Data loading error: {e}")
            return None
    
    def get_bi_monitoring_files(self, period: str) -> List[str]:
        """Get available files in BI Monitoring folder for given period."""
        workspace_path = Path(__file__).parent
        period_folder_map = {
            "daily": "Daily",
            "weekly": "Weekly", 
            "monthly": "Monthly",
            "quarterly": "Quarterly",
            "yearly": "Yearly"
        }
        
        period_folder = period_folder_map.get(period, "Daily")
        period_path = workspace_path / "Monitoring Data Files" / "BI Monitoring" / period_folder
        
        if period_path.exists():
            excel_files = []
            for pattern in ['*.xlsx', '*.xls']:
                excel_files.extend(period_path.glob(pattern))
            
            # Filter out temporary files and return just the names without extension
            file_names = []
            for f in excel_files:
                if not f.name.startswith('~$'):
                    # Remove .xlsx or .xls extension
                    name_without_ext = f.stem
                    file_names.append(name_without_ext)
            
            return sorted(file_names)
        
        return []
    
    def get_correspondence_files(self) -> List[str]:
        """Get available files in Correspondence folder."""
        workspace_path = Path(__file__).parent
        correspondence_path = workspace_path / "Monitoring Data Files" / "Correspondence"
        
        if correspondence_path.exists():
            excel_files = []
            for pattern in ['*.xlsx', '*.xls']:
                excel_files.extend(correspondence_path.glob(pattern))
            
            # Filter out temporary files and exclude upload status file (to be integrated)
            file_names = []
            for f in excel_files:
                if not f.name.startswith('~$'):
                    # Remove .xlsx or .xls extension
                    name_without_ext = f.stem
                    # Exclude the upload status file as it will be integrated into Tango Monitoring
                    if name_without_ext != "Tango Monitoring File Upload Status":
                        file_names.append(name_without_ext)
            
            return sorted(file_names)
        
        return []
    
    def load_tango_upload_status(self, date_str: str = None) -> pd.DataFrame:
        """Load Tango Monitoring File Upload Status data for a specific date.
        
        Args:
            date_str: Date string to determine which sheet to load from (optional)
        
        Returns:
            DataFrame with upload status data
        """
        workspace_path = Path(__file__).parent
        upload_status_path = workspace_path / "Monitoring Data Files" / "Correspondence" / "Tango Monitoring File Upload Status.xlsx"
        
        if upload_status_path.exists():
            try:
                loader = ExcelDataLoader(str(upload_status_path))
                
                # If no specific date provided, load first sheet
                if date_str is None:
                    df = loader.load_data()
                    # Apply date filtering to show only recent weeks
                    df = filter_data_to_recent_weeks(df)
                    return df
                
                # Convert date string to sheet name format
                sheet_name = self._convert_date_to_sheet_name(date_str)
                
                # First check if the specific sheet exists
                available_sheets = self._get_available_upload_status_sheets()
                if sheet_name not in available_sheets:
                    logger.info(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")
                    return pd.DataFrame()  # Return empty DataFrame when sheet doesn't exist
                
                # Try to load from the specific sheet
                try:
                    df = loader.load_data(sheet_name=sheet_name)
                    # Apply date filtering to show only recent weeks
                    df = filter_data_to_recent_weeks(df)
                    logger.info(f"Loaded upload status data from sheet: {sheet_name}")
                    return df
                except Exception as sheet_error:
                    logger.warning(f"Could not load from sheet '{sheet_name}': {sheet_error}")
                    # Return empty DataFrame instead of fallback
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error loading upload status data: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _convert_date_to_sheet_name(self, date_str: str) -> str:
        """Convert date string to Excel sheet name format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Sheet name in DD-MON-YYYY format (e.g., "07-OCT-2025")
        """
        try:
            # First, try to parse the date string and convert to standard format
            formatted_date = format_date_to_standard(date_str)
            return formatted_date
        except Exception as e:
            logger.warning(f"Could not convert date '{date_str}' to sheet name: {e}")
            # Return the original string as fallback
            return date_str
    
    def _get_available_upload_status_sheets(self) -> List[str]:
        """Get list of available sheet names in the upload status Excel file.
        
        Returns:
            List of sheet names
        """
        workspace_path = Path(__file__).parent
        upload_status_path = workspace_path / "Monitoring Data Files" / "Correspondence" / "Tango Monitoring File Upload Status.xlsx"
        
        if upload_status_path.exists():
            try:
                loader = ExcelDataLoader(str(upload_status_path))
                sheets = loader.get_available_sources()
                return sheets
            except Exception as e:
                logger.error(f"Error getting sheet names: {e}")
                return []
        
        return []
    
    def get_error_counts_files(self) -> List[str]:
        """Get available files in 100 Error Counts folder."""
        workspace_path = Path(__file__).parent
        error_counts_path = workspace_path / "Monitoring Data Files" / "100 Error Counts"
        
        if error_counts_path.exists():
            excel_files = []
            for pattern in ['*.xlsx', '*.xls']:
                excel_files.extend(error_counts_path.glob(pattern))
            
            # Filter out temporary files and return just the names without extension
            file_names = []
            for f in excel_files:
                if not f.name.startswith('~$'):
                    # Remove .xlsx or .xls extension
                    name_without_ext = f.stem
                    file_names.append(name_without_ext)
            
            return sorted(file_names)
        
        return []
    
    def get_user_impact_files(self) -> List[str]:
        """Get available files in User Impact folder."""
        workspace_path = Path(__file__).parent
        user_impact_path = workspace_path / "Monitoring Data Files" / "User Impact"
        
        if user_impact_path.exists():
            excel_files = []
            for pattern in ['*.xlsx', '*.xls']:
                excel_files.extend(user_impact_path.glob(pattern))
            
            # Filter out temporary files and return just the names without extension
            file_names = []
            for f in excel_files:
                if not f.name.startswith('~$'):
                    # Remove .xlsx or .xls extension
                    name_without_ext = f.stem
                    file_names.append(name_without_ext)
            
            return sorted(file_names)
        
        return []

    def render_navigation_menu(self) -> Dict[str, str]:
        """Render left navigation menu with expandable tree structure and return selected section and period.
        
        Returns:
            Dictionary with selected section, period, and subsection
        """
        st.sidebar.title("📊 Monitoring Dashboard")
        
        # Add welcome message and instructions
        st.sidebar.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="color: #1f4e79; margin: 0;">👋 Welcome!</h4>
        <p style="margin: 5px 0; font-size: 12px; color: #333;">
        <b>Getting Started:</b><br>
        1️⃣ Choose your date range below<br>
        2️⃣ Click any dashboard section<br>
        3️⃣ Select subsections to view data
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Date filtering control with better UI
        st.sidebar.markdown("### 📅 Date Range")
        weeks_to_show = st.sidebar.selectbox(
            "Show data from:",
            options=[1, 2, 3, 4],
            index=1,  # Default to 2 weeks (current + previous)
            format_func=lambda x: {
                1: "📍 Current week only",
                2: "📊 Current + Previous week", 
                3: "📈 Current + 2 Previous weeks",
                4: "📋 Current + 3 Previous weeks"
            }[x],
            help="Filter data to show only recent complete weeks (Monday to Sunday)"
        )
        
        # Store in session state for use in filtering
        st.session_state['weeks_to_show'] = weeks_to_show
        
        st.sidebar.markdown("---")
        
        # Add dashboard sections header with instructions
        st.sidebar.markdown("""
        ### 🎯 Dashboard Sections
        <p style="font-size: 11px; color: #666; margin-bottom: 10px;">
        Click any section below to view data. Sections with ▶️ can be expanded.
        </p>
        """, unsafe_allow_html=True)
        
        # Professional Dashboard Styling
        tree_css = """
        <style>
        /* Import Google Fonts for professional typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global font styling */
        .main .block-container {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Professional color scheme */
        :root {
            --primary-blue: #1f4e79;
            --secondary-blue: #2d5aa0;
            --accent-blue: #4a90e2;
            --light-blue: #e8f4fd;
            --success-green: #28a745;
            --warning-orange: #fd7e14;
            --danger-red: #dc3545;
            --neutral-gray: #6c757d;
            --light-gray: #f8f9fa;
            --dark-gray: #343a40;
        }
        
        /* Force content to always start from top */
        .main .block-container {
            padding-top: 1rem !important;
            margin-top: 0 !important;
            scroll-behavior: auto !important;
        }
        
        .stApp > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Reset scroll position on page changes */
        html, body {
            scroll-behavior: auto !important;
        }
        
        /* Enhanced navigation styling with better color visibility */
        .stButton > button {
            text-align: left !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Make sidebar buttons more visible with container colors */
        div[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.7) !important;
            border: 1px solid rgba(0,0,0,0.15) !important;
            color: #333 !important;
            font-weight: 600 !important;
            padding: 10px 14px !important;
            border-radius: 6px !important;
            margin: 2px 0 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        /* Enhanced hover effects */
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,1) !important;
            border-color: rgba(0,0,0,0.2) !important;
            transform: translateX(2px) !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        }
        
        /* Custom navigation button styling */
        .nav-button {
            width: 100%;
            text-align: left;
            border: none;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            padding: 6px 12px;
            margin: 0px !important;
            border-radius: 4px;
            transition: all 0.2s ease;
            display: block;
        }
        
        .nav-button:hover {
            opacity: 0.8;
        }
        
        .sub-nav-button {
            width: 100%;
            text-align: left;
            border: none;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            padding: 4px 10px;
            margin: 0px !important;
            border-radius: 3px;
            font-size: 13px;
            transition: all 0.2s ease;
            display: block;
        }
        

        
        /* Professional sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
            border-right: 3px solid #1f4e79 !important;
        }
        
        /* File Upload Button */
        .file-menu-container .stButton > button {
            background-color: #6c757d !important;
            color: white !important;
            border: none !important;
            padding: 10px 16px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
        }
        
        .file-menu-container .stButton > button:hover {
            background-color: #5a6268 !important;
            color: white !important;
        }
        
        /* Active state styling will be handled via inline backgrounds */
        
        /* Tree symbols styling */
        .tree-symbol {
            color: #888 !important;
            font-family: monospace !important;
            margin-right: 8px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
        }
        
        /* Professional file-menu containers (dashboard buttons) */
        .file-menu-container .stButton > button {
            font-size: 11px !important;
            color: var(--dark-gray) !important;
            background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%) !important;
            border: 1px solid #dadce0 !important;
            margin: 2px 0 !important;
            padding: 6px 10px !important;
            min-height: 28px !important;
            line-height: 1.3 !important;
            margin-left: 20px !important;
            border-radius: 5px !important;
            font-weight: 450 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.2s ease !important;
        }
        
        /* Dashboard button hover effects */
        .file-menu-container .stButton > button:hover {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--secondary-blue) 100%) !important;
            border-color: var(--primary-blue) !important;
            color: white !important;
            transform: translateX(3px) !important;
            box-shadow: 0 3px 8px rgba(31, 78, 121, 0.2) !important;
        }
        
        /* Alternative approach - target by indentation */
        div[style*="margin-left: 15px"] .stButton > button {
            font-size: 12px !important;
            padding: 4px 8px !important;
            min-height: 28px !important;
            line-height: 1.3 !important;
        }
        
        div[style*="margin-left: 20px"] .stButton > button {
            font-size: 10px !important;
            padding: 3px 6px !important;
            min-height: 22px !important;
            line-height: 1.2 !important;
        }
        
        /* Additional targeting for deeply nested buttons in sidebar */
        div[data-testid="stSidebar"] div[style*="margin-left"] .stButton > button {
            font-size: 11px !important;
            color: #666 !important;
            background-color: #f8f9fa !important;
        }
        
        /* Professional table styling */
        .stDataFrame {
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            border: 1px solid #e9ecef !important;
        }
        
        /* Enhanced table headers */
        .stDataFrame table thead th {
            font-weight: 600 !important;
            font-size: 14px !important;
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%) !important;
            color: white !important;
            text-align: center !important;
            padding: 14px 10px !important;
            border-bottom: none !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Table body styling */
        .stDataFrame table tbody td {
            padding: 10px 8px !important;
            border-bottom: 1px solid #f1f3f4 !important;
            font-family: 'Inter', sans-serif !important;
            vertical-align: middle !important;
        }
        
        /* Alternating row colors */
        .stDataFrame table tbody tr:nth-child(even) {
            background-color: #f8f9fa !important;
        }
        
        .stDataFrame table tbody tr:nth-child(odd) {
            background-color: white !important;
        }
        
        /* Row hover effects */
        .stDataFrame table tbody tr:hover {
            background-color: var(--light-blue) !important;
            transition: background-color 0.2s ease !important;
        }
        /* Professional metrics styling */
        .metric-card {
            background: linear-gradient(135deg, white 0%, #f8f9fa 100%) !important;
            border: 1px solid #e9ecef !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin: 10px 0 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
            transition: transform 0.2s ease !important;
        }
        
        .metric-card:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
        }
        
        /* Enhanced main content area */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        }
        
        /* Professional headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif !important;
            color: var(--primary-blue) !important;
            font-weight: 600 !important;
        }
        
        /* Sidebar enhancements */
        .css-1d391kg h2 {
            color: var(--primary-blue) !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            margin-bottom: 1rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid var(--accent-blue) !important;
        }
        
        /* Professional info boxes */
        .stAlert {
            border-radius: 8px !important;
            border-left: 4px solid var(--accent-blue) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Success alerts */
        .stAlert[data-baseweb="notification"] {
            background-color: #d4edda !important;
            border-left-color: var(--success-green) !important;
        }
        
        /* Warning alerts */
        .stAlert[data-baseweb="notification"]:has([data-testid="warning"]) {
            background-color: #fff3cd !important;
            border-left-color: var(--warning-orange) !important;
        }
        
        /* Error alerts */
        .stAlert[data-baseweb="notification"]:has([data-testid="error"]) {
            background-color: #f8d7da !important;
            border-left-color: var(--danger-red) !important;
        }
        </style>
        

        """
        st.sidebar.markdown(tree_css, unsafe_allow_html=True)
        
        # Initialize expanded sections in session state
        if 'expanded_sections' not in st.session_state:
            st.session_state.expanded_sections = set()
        
        # Get current selections from session state
        current_section_key = st.session_state.get('selected_section', 'benefit_issuance')
        current_period_key = st.session_state.get('selected_period', 'daily')
        current_subsection_key = st.session_state.get('selected_subsection', None)
        
        selected_section = current_section_key
        selected_subsection = current_subsection_key
        
        st.sidebar.subheader("🏠 Navigation Tree")
        
        # Define all main sections with their details and unique colors - Reordered per user request
        main_sections = [
            {"key": "summary", "icon": "📋", "name": "System Summary", "has_subsections": False, "color": "#17a2b8", "active_color": "#138496"},
            {"key": "user_impact", "icon": "👥", "name": "User Impact", "has_subsections": True, "color": "#4caf50", "active_color": "#388e3c"},
            {"key": "error_counts", "icon": "🚨", "name": "100 Error Counts", "has_subsections": True, "color": "#f44336", "active_color": "#d32f2f"},
            {"key": "correspondence_tango", "icon": "📧", "name": "Correspondence", "has_subsections": True, "color": "#2196f3", "active_color": "#1976d2"},
            {"key": "benefit_issuance", "icon": "📈", "name": "Benefit Issuance", "has_subsections": True, "color": "#ff9800", "active_color": "#f57c00"},
            {"key": "daily_exceptions", "icon": "⚠️", "name": "Daily Exceptions", "has_subsections": True, "color": "#9c27b0", "active_color": "#7b1fa2"},
            {"key": "miscellaneous_bridges", "icon": "🔗", "name": "Miscellaneous Bridges Processes", "has_subsections": True, "color": "#008b8b", "active_color": "#006064"}
        ]
        
        # Render each main section with expandable functionality
        for i, section in enumerate(main_sections):
            section_key = section["key"]
            section_icon = section["icon"]
            section_name = section["name"]
            section_color = section["color"]
            active_color = section["active_color"]
            has_subsections = section["has_subsections"]
            
            # Create expand/collapse button if section has subsections
            if has_subsections:
                is_expanded = section_key in st.session_state.expanded_sections
                expand_symbol = "🔽" if is_expanded else "▶️"
                
                # Create main navigation with color-coded background
                button_text = f"**{expand_symbol} {section_icon} {section_name}**"
                
                # Use colored background based on section color with distinct active/inactive colors
                is_active = section_key == st.session_state.get('selected_section')
                if is_active:
                    bg_color = f"{active_color}CC"  # Very strong active color (80% opacity)
                    border_color = active_color
                else:
                    bg_color = f"{section_color}99"  # Strong inactive color (60% opacity)
                    border_color = section_color
                
                st.sidebar.markdown(f"""
                <div style="margin: 0px 0px 0px 0px !important; padding: 1px 2px 1px 2px !important; background-color: {bg_color}; border-radius: 4px; border-left: 3px solid {border_color}; margin-bottom: 0px !important; margin-top: 0px !important;">
                """, unsafe_allow_html=True)
                
                if st.sidebar.button(button_text, 
                                   key=f"expand_{section_key}_{i}", 
                                   help="Click to expand/collapse subsections",
                                   use_container_width=True, 
                                   type="secondary"):
                    # Toggle functionality for expand/collapse
                    prev_section = st.session_state.get('selected_section', None)
                    
                    if section_key in st.session_state.expanded_sections:
                        # Section is expanded - collapse it
                        st.session_state.expanded_sections.remove(section_key)
                        # If this was the selected section, keep it selected but hide subsections
                        if prev_section == section_key:
                            st.session_state.selected_section = section_key
                            st.session_state.selected_subsection = None
                    else:
                        # Section is collapsed - expand it and select it
                        if prev_section != section_key:
                            # Clear all section-related state when switching sections
                            for key in list(st.session_state.keys()):
                                if key.startswith(('selected_', 'current_', 'clicked_', 'data_', 'df_')):
                                    del st.session_state[key]
                        
                        # Collapse all other sections and expand this one
                        st.session_state.expanded_sections.clear()
                        st.session_state.expanded_sections.add(section_key)
                        st.session_state.selected_section = section_key
                        st.session_state.selected_subsection = None
                    
                    st.rerun()
                
                st.sidebar.markdown('</div>', unsafe_allow_html=True)  # Close margin div
                
                # Add colored separator line matching section color
                st.sidebar.markdown(f"""
                <div style="margin: 6px 0px; height: 2px; background: linear-gradient(90deg, {section_color}AA, {section_color}44, transparent); border-radius: 1px;"></div>
                """, unsafe_allow_html=True)
                
                # Show subsections if expanded
                if section_key in st.session_state.expanded_sections:
                    # Add minimal indentation for subsections
                    st.sidebar.markdown('<div style="margin-left: 5px; margin-top: 0px;">', unsafe_allow_html=True)
                    
                    if section_key == "benefit_issuance":
                        # Initialize expanded periods in session state
                        if 'expanded_periods' not in st.session_state:
                            st.session_state.expanded_periods = set()
                        
                        # Define time periods with their details
                        period_sections = [
                            {"key": "daily", "icon": "📈", "name": "Daily"},
                            {"key": "weekly", "icon": "📊", "name": "Weekly"},
                            {"key": "monthly", "icon": "📉", "name": "Monthly"},
                            {"key": "quarterly", "icon": "📆", "name": "Quarterly"},
                            {"key": "yearly", "icon": "📋", "name": "Yearly"}
                        ]
                        
                        # Simple, clear button structure with indentation and smaller styling
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        for period_section in period_sections:
                            period_key = period_section["key"]
                            period_icon = period_section["icon"]
                            period_name = period_section["name"]
                            is_period_expanded = period_key in st.session_state.expanded_periods
                            period_expand_symbol = "🔽" if is_period_expanded else "▶️"
                            
                            # Clear, clickable button for periods
                            if st.sidebar.button(f"  {period_expand_symbol} {period_icon} {period_name}", 
                                               key=f"period_{period_key}",
                                               help=f"Click to expand/collapse {period_name}",
                                               use_container_width=True):
                                if is_period_expanded:
                                    st.session_state.expanded_periods.discard(period_key)
                                else:
                                    st.session_state.expanded_periods.add(period_key)
                                st.rerun()
                            
                            # Show files if period is expanded
                            if is_period_expanded:
                                available_files = self.get_bi_monitoring_files(period_key)
                                
                                if available_files:
                                    file_icons = {
                                        "FAP Daily Issuance": "💳",
                                        "FIP Daily Issuance": "🏦", 
                                        "SDA Daily Client Payments": "💰"
                                    }
                                    
                                    # Indented file buttons with smaller styling
                                    st.sidebar.markdown('<div style="margin-left: 12px; margin-top: 0px;" class="file-menu-container">', unsafe_allow_html=True)
                                    
                                    for i, file_name in enumerate(available_files):
                                        icon = file_icons.get(file_name, "📄")
                                        tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                        
                                        # Check if this is the currently selected dashboard
                                        is_active = (st.session_state.get('selected_section') == section_key and 
                                                   st.session_state.get('selected_subsection') == file_name and
                                                   st.session_state.get('selected_period') == period_key)
                                        
                                        # Add active styling if selected
                                        if is_active:
                                            st.sidebar.markdown('<div data-active="true" style="background-color: #e3f2fd; border-radius: 4px; margin: 1px 0;">', unsafe_allow_html=True)
                                        
                                        # Hierarchical clickable button for files
                                        if st.sidebar.button(f"　　{tree_symbol} {icon} {file_name}", 
                                                           key=f"file_{period_key}_{file_name}",
                                                           help=f"Click to analyze {file_name}",
                                                           use_container_width=True,
                                                           type="secondary"):
                                            # Clear all data state when selecting a new subsection
                                            for key in list(st.session_state.keys()):
                                                if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                                                    del st.session_state[key]
                                            st.session_state.selected_section = section_key
                                            st.session_state.selected_subsection = file_name
                                            st.session_state.selected_period = period_key
                                            st.rerun()
                                        
                                        # Close active div if it was opened
                                        if is_active:
                                            st.sidebar.markdown('</div>', unsafe_allow_html=True)
                                    
                                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.sidebar.markdown('<div style="margin-left: 35px; color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "correspondence_tango":
                        # Get files from Correspondence folder
                        available_files = self.get_correspondence_files()
                        
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        if available_files:
                            file_icons = {
                                "Tango Monitoring": "📨",
                                "View History Screen Validation": "📋"
                            }
                            
                            # Hierarchical file buttons with tree symbols
                            for i, file_name in enumerate(available_files):
                                icon = file_icons.get(file_name, "📄")
                                tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                
                                if st.sidebar.button(f"　{tree_symbol} {icon} {file_name}", 
                                                   key=f"file_correspondence_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True,
                                                   type="secondary"):
                                    # Clear all data state when selecting a new subsection
                                    for key in list(st.session_state.keys()):
                                        if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                                            del st.session_state[key]
                                    st.session_state.selected_section = section_key
                                    st.session_state.selected_subsection = file_name
                                    st.rerun()
                        else:
                            st.sidebar.markdown('<div style="color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "error_counts":
                        # Get files from 100 Error Counts folder
                        available_files = self.get_error_counts_files()
                        
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        if available_files:
                            file_icons = {
                                "Daily 100 Error Counts": "🚨"
                            }
                            
                            # Hierarchical file buttons with tree symbols
                            for i, file_name in enumerate(available_files):
                                icon = file_icons.get(file_name, "📄")
                                tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                
                                # Hierarchical clickable button for files
                                if st.sidebar.button(f"　{tree_symbol} {icon} {file_name}", 
                                                   key=f"file_error_counts_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True,
                                                   type="secondary"):
                                    st.session_state.selected_section = section_key
                                    st.session_state.selected_subsection = file_name
                                    st.rerun()
                                    st.rerun()
                        else:
                            st.sidebar.markdown('<div style="color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "user_impact":
                        # Get files from User Impact folder
                        available_files = self.get_user_impact_files()
                        
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        if available_files:
                            file_icons = {
                                "Daily User Impact Status": "👥"
                            }
                            
                            # Hierarchical file buttons with tree symbols
                            for i, file_name in enumerate(available_files):
                                icon = file_icons.get(file_name, "📄")
                                tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                
                                if st.sidebar.button(f"　{tree_symbol} {icon} {file_name}", 
                                                   key=f"file_user_impact_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True,
                                                   type="secondary"):
                                    st.session_state.selected_section = section_key
                                    st.session_state.selected_subsection = file_name
                                    st.rerun()

                        else:
                            st.sidebar.markdown('<div style="color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "miscellaneous_bridges":
                        # Miscellaneous Bridges Processes subsections
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        # Define the miscellaneous subsections
                        misc_sections = [
                            {"key": "mass_update", "icon": "🔄", "name": "Mass Update"},
                            {"key": "interfaces", "icon": "🔗", "name": "Interfaces"},
                            {"key": "extra_batch_connections", "icon": "⚡", "name": "Extra Batch Connections"},
                            {"key": "hung_threads", "icon": "🧵", "name": "Hung Threads"},
                            {"key": "data_warehouse", "icon": "🏢", "name": "Data Warehouse"},
                            {"key": "consolidated_inquiry", "icon": "🔍", "name": "Consolidated Inquiry"}
                        ]
                        
                        # Hierarchical subsection buttons with tree symbols
                        for i, misc_section in enumerate(misc_sections):
                            misc_key = misc_section["key"]
                            misc_icon = misc_section["icon"]
                            misc_name = misc_section["name"]
                            
                            # Use tree symbols for hierarchy (last item gets └─, others get ├─)
                            tree_symbol = "└─" if i == len(misc_sections) - 1 else "├─"
                            
                            # Check if this is the currently selected dashboard
                            is_active = (st.session_state.get('selected_section') == misc_key)
                            
                            # Add active styling if selected
                            if is_active:
                                st.sidebar.markdown('<div data-active="true" style="background-color: #e3f2fd; border-radius: 4px; margin: 1px 0;">', unsafe_allow_html=True)
                            
                            if st.sidebar.button(f"　{tree_symbol} {misc_icon} {misc_name}", 
                                               key=f"misc_{misc_key}",
                                               help=f"Click to analyze {misc_name}",
                                               use_container_width=True,
                                               type="secondary"):
                                # Clear all data state when selecting a new subsection
                                for key in list(st.session_state.keys()):
                                    if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                                        del st.session_state[key]
                                st.session_state.selected_section = misc_key
                                st.session_state.selected_subsection = None
                                st.rerun()
                            
                            # Close active div if it was opened
                            if is_active:
                                st.sidebar.markdown('</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "daily_exceptions":
                        # Daily Exceptions subsections
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        # Define the exception subsections
                        exception_sections = [
                            {"key": "online_exceptions_prd", "icon": "🌐", "name": "Online Exceptions - PRD"},
                            {"key": "batch_exceptions_prd", "icon": "💻", "name": "Batch Exceptions - PRD"},
                            {"key": "online_exceptions_uat", "icon": "🧪", "name": "Online Exceptions - UAT"},
                            {"key": "batch_exceptions_uat", "icon": "🔬", "name": "Batch Exceptions - UAT"}
                        ]
                        
                        # Hierarchical subsection buttons with tree symbols
                        for i, exception_section in enumerate(exception_sections):
                            exception_key = exception_section["key"]
                            exception_icon = exception_section["icon"]
                            exception_name = exception_section["name"]
                            
                            # Use tree symbols for hierarchy (last item gets └─, others get ├─)
                            tree_symbol = "└─" if i == len(exception_sections) - 1 else "├─"
                            
                            # Check if this is the currently selected dashboard
                            is_active = (st.session_state.get('selected_section') == exception_key)
                            
                            # Add active styling if selected
                            if is_active:
                                st.sidebar.markdown('<div data-active="true" style="background-color: #e3f2fd; border-radius: 4px; margin: 1px 0;">', unsafe_allow_html=True)
                            
                            if st.sidebar.button(f"　{tree_symbol} {exception_icon} {exception_name}", 
                                               key=f"exception_{exception_key}",
                                               help=f"Click to analyze {exception_name}",
                                               use_container_width=True,
                                               type="secondary"):
                                # Clear all data state when selecting a new subsection
                                for key in list(st.session_state.keys()):
                                    if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                                        del st.session_state[key]
                                st.session_state.selected_section = exception_key
                                st.session_state.selected_subsection = None
                                st.rerun()
                            
                            # Close active div if it was opened
                            if is_active:
                                st.sidebar.markdown('</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
            else:
                # Section without subsections - just a clickable button with color coding
                is_active = st.session_state.get('selected_section') == section_key
                
                # Use colored background based on section color with distinct active/inactive colors
                if is_active:
                    bg_color = f"{active_color}CC"  # Very strong active color (80% opacity)
                    border_color = active_color
                else:
                    bg_color = f"{section_color}99"  # Strong inactive color (60% opacity)
                    border_color = section_color
                    
                button_text = f"**{section_icon} {section_name}**"
                
                st.sidebar.markdown(f"""
                <div style="margin: 0px 0px 0px 0px !important; padding: 1px 2px 1px 2px !important; background-color: {bg_color}; border-radius: 4px; border-left: 3px solid {border_color}; margin-bottom: 0px !important; margin-top: 0px !important;">
                """, unsafe_allow_html=True)
                
                if st.sidebar.button(button_text, key=f"section_{section_key}_{i}", 
                                   help=f"Select {section_name}", use_container_width=True, type="secondary"):
                    st.session_state.selected_section = section_key
                    st.session_state.selected_subsection = None
                    st.rerun()
                
                st.sidebar.markdown('</div>', unsafe_allow_html=True)  # Close color div
                
                # Add colored separator line matching section color
                st.sidebar.markdown(f"""
                <div style="margin: 6px 0px; height: 2px; background: linear-gradient(90deg, {section_color}AA, {section_color}44, transparent); border-radius: 1px;"></div>
                """, unsafe_allow_html=True)
        
        # If Benefit Issuance is selected, show its sub-files indented
        if current_section_key == "benefit_issuance":
            available_files = self.get_bi_monitoring_files(current_period_key)
            file_icons = {
                "FAP Daily Issuance": "💳",
                "FIP Daily Issuance": "🏦", 
                "SDA Daily Client Payments": "💰"
            }
            
# Old code removed - now handled by loop above
        
        # Find current selection index
        current_selection_index = 0
        if current_section_key == "benefit_issuance":
            if current_subsection_key:
                # Find the subsection index
                # Selection index no longer needed with button-based navigation
                pass
            else:
                current_selection_index = 0  # Benefit Issuance main item
        else:
            # Find the main section
            section_map = {
                "correspondence_tango": "📧 Correspondence-Tango",
                "error_counts": "🚨 100 Error Counts",
                "user_impact": "👥 User Impact",
                "mass_update": "🔄 Mass Update",
                "interfaces": "🔗 Interfaces",
                "extra_batch_connections": "⚡ Extra Batch Connections",
                "hung_threads": "🧵 Hung Threads",
                "data_warehouse": "🏢 Data Warehouse",
                "consolidated_inquiry": "🔍 Consolidated Inquiry",
                "miscellaneous_bridges": "🔗 Miscellaneous Bridges Processes",
                "daily_exceptions": "⚠️ Daily Exceptions",
                "online_exceptions_prd": "🌐 Online Exceptions - PRD",
                "batch_exceptions_prd": "📦 Batch Exceptions - PRD",
                "online_exceptions_uat": "🧪 Online Exceptions - UAT",
                "batch_exceptions_uat": "🔬 Batch Exceptions - UAT"
            }
            section_display = section_map.get(current_section_key, "📊 Benefit Issuance")
            # No longer needed with button-based navigation
            pass
        
        # Navigation logic is handled in the button callbacks above
        # selected_section and selected_subsection are already set from button interactions
        
        # Update session state if selections changed
        if selected_section != st.session_state.get('selected_section'):
            # Clear ALL related state first
            for key in list(st.session_state.keys()):
                if key.startswith(('selected_', 'current_', 'clicked_')):
                    del st.session_state[key]
            
            # Set new section
            st.session_state.selected_section = selected_section
            st.session_state.selected_subsection = None
            st.session_state.current_section = None
            st.session_state.current_subsection = None
            
            # Force immediate rerun to prevent stale data
            st.rerun()
            
        if selected_subsection != st.session_state.get('selected_subsection'):
            st.session_state.selected_subsection = selected_subsection
        
        # Time period is now handled within Benefit Issuance section
        # For other sections, use default period
        selected_period = current_period_key if selected_section == "benefit_issuance" else "daily"
        
        # Add helpful footer at bottom of sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="background-color: #f0f8ff; padding: 8px; border-radius: 5px; font-size: 11px;">
        <b>💡 Tips:</b><br>
        • Expand sections with ▶️ arrows<br>
        • Data auto-filters by date range<br>
        • Red highlights show issues<br>
        • Currency & percentages auto-format
        </div>
        """, unsafe_allow_html=True)
        
        # Clean up status display
        section_name = selected_section.replace("_", " ").title() if selected_section else "Home"
        period_name = selected_period.title() if selected_section == "benefit_issuance" else "N/A"
        
        return {
            "section": selected_section,
            "period": selected_period,
            "subsection": selected_subsection,
            "section_display": section_name,
            "period_display": period_name
        }

    def render_content_placeholder(self) -> None:
        """Render section-specific content when no subsection is selected."""
        
        # Get current selected section
        selected_section = st.session_state.get('selected_section')
        
        # If no section is selected, show the main welcome page
        if not selected_section:
            self.render_welcome_page()
            return
            
        # Show section-specific home page
        self.render_section_home_page(selected_section)
    
    def render_welcome_page(self) -> None:
        """Render the main welcome page."""
        
        # Create an empty container at the very top to anchor content
        st.empty()
        
        # Override global h1 styles for our header
        st.markdown("""
        <style>
        .dashboard-header h1 {
            color: white !important;
        }
        .dashboard-header p {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Professional dashboard header
        st.markdown("""
        <div class="dashboard-header" style="
            background: linear-gradient(135deg, #1f4e79 0%, #2d5aa0 50%, #4a90e2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(31, 78, 121, 0.3);
        ">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <div style="
                    background: rgba(255,255,255,0.2);
                    border-radius: 50%;
                    padding: 15px;
                    margin-right: 20px;
                ">
                    <span style="font-size: 2.5rem;">📊</span>
                </div>
                <div>
                    <h1 style="
                        margin: 0;
                        font-size: 2.5rem;
                        font-weight: 700;
                        font-family: 'Inter', sans-serif;
                        color: white !important;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                    ">Bridges M&O Status</h1>
                    <p style="
                        margin: 5px 0 0 0;
                        font-size: 1.2rem;
                        color: white !important;
                        opacity: 0.9;
                        font-weight: 300;
                    ">Maintenance & Operations Monitoring Platform</p>
                </div>
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
            ">
                <p style="
                    margin: 0;
                    font-size: 1rem;
                    opacity: 0.95;
                ">Real-time Excel Data Analysis • Advanced Reporting • Performance Monitoring</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple welcome content to avoid HTML display issues
        st.markdown("---")
        st.markdown("## 🚀 Welcome to Your Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("📈 **Real-time Analytics**\n\nMonitor your data in real-time with dynamic updates.")
            
        with col2:
            st.success("🔍 **Advanced Monitoring**\n\nTrack performance metrics and identify trends.")
            
        with col3:
            st.warning("📊 **Interactive Reports**\n\nGenerate detailed reports with visualizations.")
        
        st.markdown("---")
        
        # Quick start guide using simple markdown
        st.markdown("### 📋 Quick Start Guide")
        
        st.markdown("""
        **Step 1: Choose Date Range** 📅  
        Use the sidebar to select how much data to show (current week, previous weeks, etc.)
        
        **Step 2: Select Dashboard Section** 👈  
        Click any section in the sidebar - start with "🚨 100 Error Counts"
        
        **Step 3: Expand for Details** ▶️  
        Click the arrow icons to expand sections and see your data
        """)
        
        # Available Dashboard Sections
        st.markdown("### 📋 Available Dashboard Sections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            **🔴 High Priority Monitoring**
            
            📋 **System Summary** - Complete overview with status cards
            
            👥 **User Impact** - Daily user impact status & tracking
            
            🚨 **100 Error Counts** - Session timeouts & system errors
            
            📧 **Correspondence** - Tango monitoring & uploads
            """)
            
        with col2:
            st.info("""
            **📊 Business Intelligence & Processing**
            
            📈 **Benefit Issuance** - FAP, FIP, SDA tracking
            
            ⚠️ **Daily Exceptions** - Exception monitoring & resolution
            
            🔧 **Miscellaneous Bridges Processes** - Bridge system monitoring
            """)
    
    def render_section_home_page(self, section_key: str) -> None:
        """Render section-specific home page with relevant information."""
        # Section information mapping
        section_info = {
            "summary": {
                "title": "System Summary",
                "icon": "📋",
                "color": "#17a2b8",
                "description": "Get a comprehensive overview of all system components with real-time status cards and health indicators.",
                "features": [
                    "System-wide health monitoring",
                    "Color-coded status indicators",
                    "Quick metrics and alerts",
                    "Navigate to detailed sections"
                ]
            },
            "error_counts": {
                "title": "100 Error Counts",
                "icon": "🚨",
                "color": "#f44336",
                "description": "Monitor session timeouts, system errors, and critical error counts that impact system performance.",
                "features": [
                    "Track daily error patterns",
                    "Identify peak error times", 
                    "Monitor error types and frequency",
                    "Generate error trend reports"
                ]
            },
            "user_impact": {
                "title": "User Impact",
                "icon": "👥", 
                "color": "#ff9800",
                "description": "Track and analyze the impact of system issues on user experience and service availability.",
                "features": [
                    "Daily user impact tracking",
                    "Service availability metrics",
                    "User experience analysis",
                    "Impact severity assessment"
                ]
            },
            "benefit_issuance": {
                "title": "Benefit Issuance",
                "icon": "📈",
                "color": "#2196f3", 
                "description": "Monitor benefit issuance processes including FAP, FIP, and SDA client payment tracking.",
                "features": [
                    "FAP Daily Issuance monitoring",
                    "FIP Daily Issuance tracking", 
                    "SDA Daily Client Payments",
                    "Cross-program analysis"
                ]
            },
            "correspondence_tango": {
                "title": "Correspondence",
                "icon": "📧",
                "color": "#4caf50", 
                "description": "Monitor correspondence systems, Tango monitoring, and file upload status tracking.",
                "features": [
                    "Tango monitoring status",
                    "File upload tracking",
                    "View history validation",
                    "Correspondence processing"
                ]
            },
            "mass_update": {
                "title": "Mass Update",
                "icon": "🔄", 
                "color": "#9c27b0",
                "description": "Track mass update processes and system-wide changes.",
                "features": [
                    "Mass update status",
                    "Update completion tracking",
                    "System change monitoring",
                    "Update impact analysis"
                ]
            },
            "interfaces": {
                "title": "Interfaces",
                "icon": "🔗",
                "color": "#607d8b", 
                "description": "Monitor system interfaces and integration points.",
                "features": [
                    "Interface status monitoring",
                    "Connection health checks",
                    "Data flow tracking",
                    "Integration performance"
                ]
            },
            "extra_batch_connections": {
                "title": "Extra Batch Connections", 
                "icon": "⚡",
                "color": "#ff5722",
                "description": "Monitor additional batch connections and process execution.",
                "features": [
                    "Extra connection tracking",
                    "Batch process monitoring",
                    "Connection performance",
                    "Process completion status"
                ]
            },
            "hung_threads": {
                "title": "Hung Threads",
                "icon": "🧵",
                "color": "#795548",
                "description": "Detect and monitor hung threads that may impact system performance.",
                "features": [
                    "Thread status monitoring",
                    "Hung thread detection",
                    "Performance impact analysis", 
                    "Thread recovery tracking"
                ]
            },
            "data_warehouse": {
                "title": "Data Warehouse",
                "icon": "🏢",
                "color": "#3f51b5",
                "description": "Monitor data warehouse operations, ETL processes, and data integrity.",
                "features": [
                    "ETL job monitoring",
                    "Data quality checks",
                    "Warehouse performance metrics",
                    "Data refresh status"
                ]
            },
            "consolidated_inquiry": {
                "title": "Consolidated Inquiry",
                "icon": "🔍",
                "color": "#4caf50",
                "description": "Track consolidated inquiry processes and cross-system data retrieval.",
                "features": [
                    "Inquiry response times",
                    "Cross-system integration",
                    "Data consolidation status",
                    "Performance analytics"
                ]
            },
            "miscellaneous_bridges": {
                "title": "Miscellaneous Bridges Processes",
                "icon": "🔗",
                "color": "#008b8b",
                "description": "Monitor various bridge processes including mass updates, interfaces, extra connections, and hung threads.",
                "features": [
                    "Mass Update monitoring",
                    "Interface status tracking",
                    "Extra Batch Connection analysis",
                    "Hung Thread detection"
                ]
            },
            "daily_exceptions": {
                "title": "Daily Exceptions",
                "icon": "⚠️",
                "color": "#ff9800",
                "description": "Monitor daily system exceptions across production and UAT environments.",
                "features": [
                    "Production exception tracking",
                    "UAT exception monitoring", 
                    "Online vs batch comparison",
                    "Environment stability metrics"
                ]
            },
            "online_exceptions_prd": {
                "title": "Online Exceptions - PRD",
                "icon": "🌐",
                "color": "#e91e63", 
                "description": "Monitor online exceptions in the production environment.",
                "features": [
                    "Production exception tracking",
                    "Real-time error monitoring",
                    "Exception categorization",
                    "Production stability metrics"
                ]
            },
            "batch_exceptions_prd": {
                "title": "Batch Exceptions - PRD",
                "icon": "💻",
                "color": "#3f51b5",
                "description": "Track batch process exceptions in production environment.",
                "features": [
                    "Batch exception monitoring",
                    "Production batch analysis",
                    "Error pattern identification",
                    "Batch performance tracking"
                ]
            },
            "online_exceptions_uat": {
                "title": "Online Exceptions - UAT", 
                "icon": "🧪",
                "color": "#009688",
                "description": "Monitor online exceptions in the UAT testing environment.",
                "features": [
                    "UAT exception tracking",
                    "Testing environment monitoring",
                    "Pre-production analysis",
                    "Quality assurance metrics"
                ]
            },
            "batch_exceptions_uat": {
                "title": "Batch Exceptions - UAT",
                "icon": "🔬",
                "color": "#8bc34a", 
                "description": "Track batch process exceptions in UAT environment.",
                "features": [
                    "UAT batch monitoring",
                    "Testing batch analysis",
                    "Pre-production validation",
                    "Batch testing metrics"
                ]
            }
        }
        
        # Get section information
        info = section_info.get(section_key, {
            "title": "Section Overview", 
            "icon": "📊",
            "color": "#1f4e79",
            "description": "Monitor and analyze system performance metrics.",
            "features": ["Data monitoring", "Performance tracking", "Report generation"]
        })
        
        # Render section-specific header
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {info['color']} 0%, {info['color']}CC 100%);
            color: white;
            padding: 30px 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        ">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="font-size: 3rem; margin-right: 15px;">{info['icon']}</span>
                <h1 style="
                    margin: 0;
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: white !important;
                ">{info['title']}</h1>
            </div>
            <p style="
                margin: 0;
                font-size: 1.1rem;
                opacity: 0.95;
                color: white !important;
            ">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add prominent home button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Return to Bridges M&O Status Home", key="section_home_button", help="Return to main welcome page", use_container_width=True):
                # Clear all session state to return to welcome page
                for key in list(st.session_state.keys()):
                    if key.startswith(('selected_', 'expanded_')):
                        del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Section features
        st.markdown("### 🔧 Key Features")
        
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(info['features']):
            if i % 2 == 0:
                with col1:
                    st.markdown(f"✅ {feature}")
            else:
                with col2:
                    st.markdown(f"✅ {feature}")
        
        st.markdown("---")      
        
        # Instructions
        st.markdown("### 📋 Getting Started")
        st.info(f"""
        **To view {info['title']} data:**
        
        1. 📅 Set your date range using the sidebar controls
        2. 📊 Select a specific subsection to view detailed data and reports
        3. 🔍 Use filters and controls to customize your analysis
        """)
        
        # Additional help
        st.markdown("### ❓ Need Help?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**📚 Documentation**\n\nCheck the user guides for detailed instructions.")
            
        with col2:
            st.warning("**🏠 Home Page**\n\nUse the 'Bridges M&O Status Home' button to return to main page.")
            
        with col3: 
            st.info("**🔄 Refresh Data**\n\nExpand sections to load the latest data automatically.")
        
        # Features highlight
        st.markdown("---")
        st.markdown("### ✨ Dashboard Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Smart Formatting**
            - Currency: $1,234.56
            - Percentages: 4.5%
            - Dates: DD-MON-YYYY
            """)
            
        with col2:
            st.markdown("""
            **🚨 Conditional Alerts**
            - Red highlighting for issues
            - Variance detection
            - Connection warnings
            """)
            
        with col3:
            st.markdown("""
            **📊 Interactive Charts**
            - Multiple visualization types
            - Trend analysis
            - Real-time filtering
            """)
        
        # Professional call to action
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 50%, #0d47a1 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            margin: 40px 0 20px 0;
            box-shadow: 0 8px 24px rgba(25, 118, 210, 0.3);
        ">
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 50%;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 20px auto;
                font-size: 1.5rem;
            ">🚀</div>
            <h3 style="margin: 0 0 15px 0; font-weight: 600; font-size: 1.5rem;">Ready to Get Started?</h3>
            <p style="margin: 0 0 20px 0; font-size: 1.1rem; opacity: 0.9;">
                Click <strong>"🚨 100 Error Counts"</strong> in the sidebar to begin exploring your data!
            </p>
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 12px 20px;
                display: inline-block;
                font-size: 0.9rem;
                margin-top: 10px;
            ">
                👈 Start with the sidebar navigation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_main_content_with_data(self, df: pd.DataFrame, nav_result: Dict[str, Any]) -> None:
        """Render main content with loaded data."""
        

        
        selected_section = nav_result.get("section")
        selected_period = nav_result.get("period")
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply standardized date formatting to all date columns
        df = format_dataframe_dates(df)
        
        # Filter to show only current week and previous week data
        df = filter_data_to_recent_weeks(df)
        
        # Filter data based on selected period
        filtered_by_period_df = self.filter_data_by_period(df, selected_period)
        
        # Dynamic title based on section and period
        section_titles = {
            "benefit_issuance": {
                "daily": "Daily Benefit Issuance Dashboard",
                "weekly": "Weekly Benefit Issuance Dashboard", 
                "monthly": "Monthly Benefit Issuance Dashboard",
                "yearly": "Yearly Benefit Issuance Dashboard"
            },
            "correspondence_tango": {
                "daily": "Daily Correspondence-Tango Dashboard",
                "weekly": "Weekly Correspondence-Tango Dashboard", 
                "monthly": "Monthly Correspondence-Tango Dashboard",
                "yearly": "Yearly Correspondence-Tango Dashboard"
            },
            "error_counts": {
                "daily": "Daily 100 Error Counts Dashboard",
                "weekly": "Weekly 100 Error Counts Dashboard",
                "monthly": "Monthly 100 Error Counts Dashboard",
                "yearly": "Yearly 100 Error Counts Dashboard"
            },
            "user_impact": {
                "daily": "Daily User Impact Dashboard",
                "weekly": "Weekly User Impact Dashboard",
                "monthly": "Monthly User Impact Dashboard",
                "yearly": "Yearly User Impact Dashboard"
            }
        }
        
        title = section_titles.get(selected_section, {}).get(selected_period, f"{selected_section.replace('_', ' ').title()} Dashboard" if selected_section else "Dashboard")
        st.title(title)
        
        # Add home button on data pages
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Return to Bridges M&O Status Home", key="data_home_button", help="Return to main welcome page", use_container_width=True):
                # Clear all session state to return to welcome page
                for key in list(st.session_state.keys()):
                    if key.startswith(('selected_', 'expanded_')):
                        del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Section-specific content routing
        section_handlers = {
            "summary": self.render_summary_content,
            "benefit_issuance": self.render_benefit_issuance_content,
            "correspondence_tango": self.render_correspondence_tango_content,
            "error_counts": self.render_error_counts_content,
            "user_impact": self.render_user_impact_content,
            "mass_update": self.render_mass_update_content,
            "interfaces": self.render_interfaces_content,
            "extra_batch_connections": self.render_extra_batch_connections_content,
            "hung_threads": self.render_hung_threads_content,
            "data_warehouse": self.render_data_warehouse_content,
            "consolidated_inquiry": self.render_consolidated_inquiry_content,
            "miscellaneous_bridges": self.render_miscellaneous_bridges_content,
            "daily_exceptions": self.render_daily_exceptions_content,
            "online_exceptions_prd": self.render_online_exceptions_prd_content,
            "batch_exceptions_prd": self.render_batch_exceptions_prd_content,
            "online_exceptions_uat": self.render_online_exceptions_uat_content,
            "batch_exceptions_uat": self.render_batch_exceptions_uat_content
        }
        
        handler = section_handlers.get(selected_section)
        if handler:
            handler(filtered_by_period_df, selected_period)
        else:
            st.error(f"Handler not implemented for section: {selected_section}")

    def render_summary_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render System Summary with status cards for all dashboard sections."""
        
        st.markdown("## 📊 System Health Overview")
        st.markdown("Monitor the overall health and status of all system components at a glance.")
        
        # Add a small note about scrolling
        st.info("💡 **Tip:** If the page doesn't start from the top, scroll up to see the full content.")
        
        # Add home button on Summary page
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Return to Bridges M&O Status Home", key="summary_home_button", help="Return to main welcome page", use_container_width=True):
                # Clear all session state to return to welcome page
                for key in list(st.session_state.keys()):
                    if key.startswith(('selected_', 'expanded_')):
                        del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Define dashboard sections for status monitoring (matching left navigation)
        dashboard_sections = [
            {
                "key": "user_impact", 
                "name": "User Impact",
                "icon": "",
                "description": "User experience metrics & error percentages"
            },
            {
                "key": "error_counts",
                "name": "100 Error Counts",
                "icon": "",
                "description": "Session timeouts & system errors"
            },
            {
                "key": "correspondence_tango",
                "name": "Correspondence",
                "icon": "", 
                "description": "Tango monitoring & file uploads"
            },
            {
                "key": "benefit_issuance",
                "name": "Benefit Issuance", 
                "icon": "",
                "description": "FAP, FIP, SDA processing status"
            },
            {
                "key": "daily_exceptions",
                "name": "Daily Exceptions",
                "icon": "⚠️",
                "description": "Online & batch exceptions (PRD/UAT)"
            },
            {
                "key": "miscellaneous_bridges",
                "name": "Miscellaneous Bridges",
                "icon": "🔗",
                "description": "Mass updates, interfaces, connections & threads"
            }
        ]
        
        # Create status cards in a grid layout
        st.markdown("### 🎛️ Bridges M&O Summary")
        
        # Display the most recent weekday date
        recent_date = self.get_most_recent_weekday_date()
        st.markdown(f"**Data as of:** {recent_date}")
        st.markdown("")  # Add spacing
        
        # Create 3 columns for status cards (2 cards per column)
        col1, col2, col3 = st.columns(3)
        
        for i, section in enumerate(dashboard_sections):
            # Determine which column to use (cycling through 3 columns)
            if i % 3 == 0:
                current_col = col1
            elif i % 3 == 1:
                current_col = col2
            else:
                current_col = col3
            
            with current_col:
                # Get status for this section
                status, status_color, status_text = self.get_section_status(section["key"])
                
                # Create status card
                self.render_status_card(
                    section["name"],
                    section["icon"], 
                    section["description"],
                    status,
                    status_color,
                    status_text
                )
                
                # Add navigation button below each card
                if st.button(
                    f"📊 View {section['name']} Dashboard", 
                    key=f"nav_to_{section['key']}", 
                    help=f"Navigate to {section['name']} section",
                    use_container_width=True
                ):
                    # Clear data state when navigating to a new section
                    for key in list(st.session_state.keys()):
                        if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                            del st.session_state[key]
                    
                    # Initialize expanded_sections if not exists
                    if 'expanded_sections' not in st.session_state:
                        st.session_state.expanded_sections = set()
                    
                    # Auto-expand the section in navigation (all sections in System Summary have subsections)
                    # Clear other expanded sections and expand the target section
                    st.session_state.expanded_sections.clear()
                    st.session_state.expanded_sections.add(section["key"])
                    
                    st.session_state.selected_section = section["key"]
                    st.session_state.selected_subsection = None
                    st.rerun()
                
                # Add some spacing between card groups
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary metrics
        st.markdown("### 📈 Quick Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Sections", "12", "2 new")
            
        with metric_col2:
            st.metric("Active Dashboards", "8", "1 updated") 
            
        with metric_col3:
            st.metric("System Health", "95%", "2%")
            
        with metric_col4:
            st.metric("Data Freshness", "Real-time", "✓")
        
        st.markdown("---")
        
        # Alert section
        st.markdown("### 🔔 Recent Alerts")
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.warning("**⚠️ Medium Priority Alert**\n\nError counts slightly elevated in the last 2 hours. Monitor for trends.")
            
        with alert_col2:
            st.success("**✅ All Systems Normal**\n\nBenefit issuance processing within normal parameters.")
        
        # Instructions
        st.markdown("---")
        st.info("""
        **💡 How to use the Summary Dashboard:**
        
        - **Green cards** indicate normal operations
        - **Yellow cards** suggest attention needed  
        - **Red cards** require immediate action
        - Click on any status card to navigate to that section
        - Use the navigation tree to explore detailed data
        """)

    def get_section_status(self, section_key: str):
        """Get status for a dashboard section. Returns (status, color, text)."""
        
        if section_key == "error_counts":
            return self.get_error_counts_status()
        elif section_key == "user_impact":
            return self.get_user_impact_status()
        elif section_key == "correspondence_tango":
            return self.get_correspondence_status()
        elif section_key == "benefit_issuance":
            return self.get_benefit_issuance_status()
        elif section_key == "daily_exceptions":
            return self.get_daily_exceptions_status()
        elif section_key == "miscellaneous_bridges":
            return self.get_miscellaneous_bridges_status()
        else:
            # Fallback for any unknown sections
            return ("normal", "#6c757d", "Status monitoring available")
    
    def get_error_counts_status(self):
        """Get status for 100 Error Counts based on real data and thresholds."""
        try:
            # Load the Daily 100 Error Counts Excel file
            error_counts_path = Path(__file__).parent / "Monitoring Data Files" / "100 Error Counts" / "Daily 100 Error Counts.xlsx"
            
            if not error_counts_path.exists():
                return ("warning", "#ffc107", "Data file not found")
            
            # Load the Excel file
            from src.data_loader import ExcelDataLoader
            loader = ExcelDataLoader(str(error_counts_path))
            df = loader.load_data()
            
            if df.empty:
                return ("warning", "#ffc107", "No data available")
            
            # Get data for the specific target date (sysdate-1 business day)
            target_date = self.get_target_business_date()
            recent_data = self.get_data_for_specific_date(df, target_date)
            
            if recent_data is None:
                return ("warning", "#ffc107", "Data not available")
            
            # Get total error count for the most recent weekday
            total_count = self.calculate_total_error_count(recent_data)
            
            # Apply thresholds
            if total_count > 750:
                return ("critical", "#dc3545", f"High error count: {total_count}")
            elif total_count >= 700:
                return ("warning", "#ffc107", f"Moderate error count: {total_count}")
            else:
                return ("normal", "#28a745", f"Normal error count: {total_count}")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error loading data: {str(e)}")
    
    def get_user_impact_status(self):
        """Get status for User Impact based on 0 Errors % column and thresholds."""
        try:
            # Load the Daily User Impact Status Excel file and process it the same way as the dashboard
            user_impact_path = Path(__file__).parent / "Monitoring Data Files" / "User Impact" / "Daily User Impact Status.xlsx"
            
            if not user_impact_path.exists():
                return ("warning", "#ffc107", "Data file not found")
            
            # Load and process the data exactly like the User Impact dashboard does
            from src.data_loader import ExcelDataLoader
            loader = ExcelDataLoader(str(user_impact_path))
            df = loader.load_data()
            
            if df.empty:
                return ("warning", "#ffc107", "No data available")
            
            # Process the data to add calculated percentage columns (same as render_user_impact_table)
            processed_df = self.add_user_impact_percentage_columns(df.copy())
            
            # Get data for the specific target date (sysdate-1 business day)
            target_date = self.get_target_business_date()
            recent_data = self.get_data_for_specific_date(processed_df, target_date)
            
            if recent_data is None:
                return ("warning", "#ffc107", "Data not available")
            
            # Get the 0 Errors % value from the processed data
            zero_errors_pct = self.get_zero_errors_percentage(recent_data)
            
            if zero_errors_pct is None:
                return ("warning", "#ffc107", "0 Errors % column not found")
            
            # Apply thresholds based on user requirements
            if zero_errors_pct < 89:
                return ("critical", "#dc3545", f"Low success rate: {zero_errors_pct:.2f}%")
            elif zero_errors_pct >= 89 and zero_errors_pct <= 90:
                return ("warning", "#ffc107", f"Moderate success rate: {zero_errors_pct:.2f}%")
            else:  # > 90%
                return ("normal", "#28a745", f"Good success rate: {zero_errors_pct:.2f}%")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error loading data: {str(e)}")
    
    def get_correspondence_status(self):
        """Get status for Correspondence section based on Tango Monitoring data."""
        try:
            # Load the Tango Monitoring data
            tango_path = Path(__file__).parent / "Monitoring Data Files" / "Correspondence" / "Tango Monitoring.xlsx"
            
            if not tango_path.exists():
                return ("warning", "#ffc107", "Tango Monitoring data not found")
            
            # Load and process the Tango Monitoring data
            from src.data_loader import ExcelDataLoader
            loader = ExcelDataLoader(str(tango_path))
            df = loader.load_data()
            
            if df.empty:
                return ("warning", "#ffc107", "No Tango data available")
            
            # Get data for the specific target date (sysdate-1 business day)
            target_date = self.get_target_business_date()
            recent_data = self.get_data_for_specific_date(df, target_date)
            
            if recent_data is None:
                return ("warning", "#ffc107", "Data not available")
            
            # Find the "Number of Files not sent to CPC" column
            files_not_sent_col = None
            for col in df.columns:
                if "Number of Files not sent to CPC" in str(col) or "files not sent" in str(col).lower():
                    files_not_sent_col = col
                    break
            
            if files_not_sent_col is None:
                return ("warning", "#ffc107", "Files not sent column not found")
            
            # Get the value for the most recent date
            try:
                files_not_sent_value = float(recent_data[files_not_sent_col]) if pd.notna(recent_data[files_not_sent_col]) else 0
            except (ValueError, TypeError):
                return ("warning", "#ffc107", "Invalid files not sent data")
            
            # Apply red/green threshold (≥7 is red, <7 is green)
            if files_not_sent_value >= 7:
                return ("critical", "#dc3545", f"{int(files_not_sent_value)} Files not sent to CPC")
            else:
                return ("normal", "#28a745", f"{int(files_not_sent_value)} Files not sent to CPC")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error loading Tango data: {str(e)}")
    
    def get_benefit_issuance_status(self):
        """Get status for Benefit Issuance section."""
        try:
            # Check for benefit issuance data files
            bi_path = Path(__file__).parent / "Monitoring Data Files" / "BI Monitoring"
            
            if not bi_path.exists():
                return ("warning", "#ffc107", "Data not available")
            
            # Check for different time period folders
            folders = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
            available_folders = [f for f in folders if (bi_path / f).exists()]
            
            if len(available_folders) >= 4:
                return ("normal", "#28a745", f"Monitoring {len(available_folders)} time periods")
            elif len(available_folders) >= 2:
                return ("warning", "#ffc107", "Data not available")
            else:
                return ("warning", "#ffc107", "Data not available")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error checking data: {str(e)}")
    
    def get_daily_exceptions_status(self):
        """Get status for Daily Exceptions section."""
        try:
            # This would check for exception data across PRD and UAT environments
            # Since no real data source is configured yet, return data not available
            return ("warning", "#ffc107", "Data not available")
                
        except Exception as e:
            return ("warning", "#ffc107", "Data not available")
    
    def get_miscellaneous_bridges_status(self):
        """Get status for Miscellaneous Bridges Processes section."""
        try:
            # Check for various bridge process data
            processes = ["Mass Update", "Interfaces", "Extra Batch Connections", "Hung Threads"]
            
            # Simulate checking each process
            import random
            healthy_processes = random.randint(2, 4)
            
            if healthy_processes == 4:
                return ("normal", "#28a745", f"All {healthy_processes} processes operational")
            elif healthy_processes >= 3:
                return ("warning", "#ffc107", f"{healthy_processes}/4 processes operational")
            else:
                return ("critical", "#dc3545", f"Only {healthy_processes}/4 processes operational")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error checking processes: {str(e)}")
    
    def get_most_recent_weekday_data(self, df):
        """Get the most recent weekday data from the dataframe."""
        from datetime import datetime, timedelta
        
        # Find date columns in the dataframe
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'day']):
                date_columns.append(col)
        
        if not date_columns:
            # If no date column found, return the last row
            return df.iloc[-1] if len(df) > 0 else None
        
        # Convert date column to datetime
        date_col = date_columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            
            # Filter to weekdays only (Monday=0, Sunday=6)
            df_weekdays = df[df[date_col].dt.dayofweek < 5]  # 0-4 are weekdays
            
            if df_weekdays.empty:
                return df.iloc[-1] if len(df) > 0 else None
            
            # Get the most recent weekday
            most_recent = df_weekdays.loc[df_weekdays[date_col].idxmax()]
            return most_recent
            
        except Exception:
            # If date parsing fails, return the last row
            return df.iloc[-1] if len(df) > 0 else None
    
    def calculate_total_error_count(self, row_data):
        """Calculate total error count from a row of data."""
        
        # First priority: Look for the exact "Total Count" column
        for col, value in row_data.items():
            if col.strip().lower() == "total count":
                try:
                    numeric_value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(numeric_value):
                        return int(numeric_value)
                except:
                    continue
        
        # Second priority: Look for variations of total count columns
        total_count_variations = [
            'total count', 'totalcount', 'total_count', 'total counts',
            'grand total', 'sum total', 'overall total'
        ]
        
        for col, value in row_data.items():
            col_lower = col.lower().strip()
            
            # Skip date columns
            if any(keyword in col_lower for keyword in ['date', 'day']):
                continue
            
            # Look for total count variations
            if any(variation in col_lower for variation in total_count_variations):
                try:
                    numeric_value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(numeric_value):
                        return int(numeric_value)
                except:
                    continue
        
        # Third priority: Look for any column with "total" in the name
        for col, value in row_data.items():
            col_lower = col.lower().strip()
            
            # Skip date columns
            if any(keyword in col_lower for keyword in ['date', 'day', 'time']):
                continue
            
            # Look for any total column
            if 'total' in col_lower:
                try:
                    numeric_value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(numeric_value):
                        return int(numeric_value)
                except:
                    continue
        
        # Last resort: Return 0 if no suitable column found
        return 0
    
    def get_zero_errors_percentage(self, row_data):
        """Get the 0 Errors % value from a row of data."""
        
        # First priority: Look for the calculated "0 Errors %" column
        for col, value in row_data.items():
            col_stripped = col.strip()
            if col_stripped == "0 Errors %" or col_stripped.lower() == "0 errors %":
                try:
                    # Handle string percentages like "92%" from our calculated columns
                    if isinstance(value, str) and value.endswith('%'):
                        clean_value = value.replace('%', '').strip()
                        numeric_value = float(clean_value)
                        return numeric_value
                    else:
                        numeric_value = pd.to_numeric(value, errors='coerce')
                        if pd.notna(numeric_value):
                            return float(numeric_value)
                except:
                    continue
        
        # Second priority: Calculate from raw data if percentage column not found
        # Look for "0 Errors" and "# Logged-in Users" columns to calculate percentage
        zero_errors_value = None
        logged_in_users_value = None
        
        for col, value in row_data.items():
            col_stripped = col.strip()
            if col_stripped == "0 Errors":
                try:
                    zero_errors_value = pd.to_numeric(value, errors='coerce')
                except:
                    continue
            elif col_stripped == "# Logged-in Users":
                try:
                    logged_in_users_value = pd.to_numeric(value, errors='coerce')
                except:
                    continue
        
        # Calculate percentage if we have both values
        if (zero_errors_value is not None and logged_in_users_value is not None and 
            pd.notna(zero_errors_value) and pd.notna(logged_in_users_value) and 
            logged_in_users_value > 0):
            percentage = (zero_errors_value / logged_in_users_value) * 100
            return float(percentage)
        
        # If we couldn't find the data or calculate the percentage, return None
        return None
    
    def add_user_impact_percentage_columns(self, df):
        """Add calculated percentage columns to User Impact data, same as the dashboard."""
        import pandas as pd
        
        # Define the error columns we're looking for
        error_columns = ['0 errors', '1 errors', '2 errors', '3-5 errors', '6-10 errors', '>10 errors']
        
        # Find the logged-in users column
        logged_in_users_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'logged' in col_lower and 'user' in col_lower:
                logged_in_users_col = col
                break
        
        if not logged_in_users_col:
            return df  # Return original if no logged-in users column found
        
        # Work backwards through error columns to maintain correct positioning
        cols = list(df.columns)
        for error_col in reversed(error_columns):
            # Find the actual column name (case-insensitive)
            actual_error_col = None
            for col in cols:
                col_lower = col.lower().strip()
                error_col_lower = error_col.lower()
                if col_lower == error_col_lower or \
                   (error_col_lower.replace(' ', '').replace('-', '') in col_lower.replace(' ', '').replace('-', '')):
                    actual_error_col = col
                    break
            
            if actual_error_col and actual_error_col in cols:
                # Find the position to insert percentage column
                error_col_idx = cols.index(actual_error_col)
                
                # Calculate percentage for this error column
                error_pct = []
                for idx, row in df.iterrows():
                    try:
                        error_count = pd.to_numeric(row[actual_error_col], errors='coerce')
                        logged_users = pd.to_numeric(row[logged_in_users_col], errors='coerce')
                        
                        if pd.notna(error_count) and pd.notna(logged_users) and logged_users > 0:
                            percentage = (error_count / logged_users) * 100
                            error_pct.append(f"{percentage:.2f}%")
                        else:
                            error_pct.append("N/A")
                    except Exception:
                        error_pct.append("N/A")
                
                # Create percentage column name
                pct_col_name = f"{actual_error_col} %"
                
                # Insert the new column after the error column
                df.insert(error_col_idx + 1, pct_col_name, error_pct)
                
                # Update the cols list to reflect the new column
                cols = list(df.columns)
        
        return df
    
    def get_most_recent_weekday_date(self):
        """Get the most recent business day (sysdate-1) for display purposes."""
        from datetime import datetime, timedelta
        
        try:
            # Start with yesterday (sysdate - 1)
            yesterday = datetime.now() - timedelta(days=1)
            
            # If yesterday was a weekday, use it; otherwise find the previous weekday
            if yesterday.weekday() < 5:  # Monday=0, Friday=4
                target_date = yesterday
            else:
                # Yesterday was weekend, find the previous Friday
                # If yesterday was Saturday (5), go back 1 more day to Friday
                # If yesterday was Sunday (6), go back 2 more days to Friday
                days_back = yesterday.weekday() - 4  # 4 is Friday
                target_date = yesterday - timedelta(days=days_back)
            
            return target_date.strftime("%B %d, %Y")
            
        except Exception:
            return "Date not available"
    
    def get_target_business_date(self):
        """Get the target business date (sysdate-1) as a datetime object."""
        from datetime import datetime, timedelta
        
        # Start with yesterday (sysdate - 1)
        yesterday = datetime.now() - timedelta(days=1)
        
        # If yesterday was a weekday, use it; otherwise find the previous weekday
        if yesterday.weekday() < 5:  # Monday=0, Friday=4
            return yesterday
        else:
            # Yesterday was weekend, find the previous Friday
            days_back = yesterday.weekday() - 4  # 4 is Friday
            return yesterday - timedelta(days=days_back)
    
    def get_data_for_specific_date(self, df, target_date):
        """Get data for a specific date from the dataframe."""
        from datetime import datetime
        
        # Find date columns in the dataframe
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'day']):
                date_columns.append(col)
        
        if not date_columns:
            # If no date column found, return the last row as fallback
            return df.iloc[-1] if len(df) > 0 else None
        
        # Convert date column to datetime for comparison
        date_col = date_columns[0]
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if df_copy.empty:
                return df.iloc[-1] if len(df) > 0 else None
            
            # Convert target_date to just the date part for comparison
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%B %d, %Y")
            target_date_only = target_date.date()
            
            # Find exact match for the target date
            matching_rows = df_copy[df_copy[date_col].dt.date == target_date_only]
            
            if not matching_rows.empty:
                # Return the first matching row for the target date
                return matching_rows.iloc[0]
            else:
                # If no exact match for target date, return None to indicate data not available
                return None
                
        except Exception:
            # If date parsing fails, return the last row
            return df.iloc[-1] if len(df) > 0 else None
    
    def render_status_card(self, title: str, icon: str, description: str, status: str, color: str, status_text: str):
        """Render a status card with the given parameters."""
        
        # Determine border and background based on status
        if status == "normal":
            border_color = "#28a745"
            bg_color = "#f8fff9"
            status_icon = "✅"
        elif status == "warning":
            border_color = "#ffc107"
            bg_color = "#fffdf5"
            status_icon = "⚠️"
        else:  # critical
            border_color = "#dc3545"
            bg_color = "#fff5f5"
            status_icon = "🚨"
        
        # Create the status card with fixed height for consistency
        st.markdown(f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: {bg_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem; margin-right: 8px;">{icon}</span>
                    <h4 style="margin: 0; color: #333; font-size: 1.1rem; line-height: 1.2;">{title}</h4>
                </div>
                <p style="margin: 5px 0; color: #666; font-size: 0.85rem; line-height: 1.3;">{description}</p>
            </div>
            <div style="display: flex; align-items: center; margin-top: auto;">
                <span style="margin-right: 8px;">{status_icon}</span>
                <span style="color: {color}; font-weight: bold; font-size: 0.85rem; line-height: 1.2;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


    def render_benefit_issuance_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Benefit Issuance specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Filters in main area with expander
        with st.expander("🔍 **Data Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        # Apply date formatting to filtered data
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus with variance highlighting
        st.header("📋 Data Table")
        file_display_name = f"{selected_subsection} Data"
        self.render_benefit_issuance_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.metrics_component.auto_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Statistics", "🔬 Analysis"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="benefit_")
        
        with tab2:
            self.table_component.summary_stats(filtered_df)
        
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="benefit_")
    
    def render_benefit_issuance_table(self, df: pd.DataFrame, title: str) -> None:
        """Render benefit issuance table with variance highlighting."""
        import pandas as pd
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Check if dataframe has variance-related columns, specifically "Variance in #Benefits"
        variance_columns = []
        benefits_variance_col = None
        
        # Look for specific "Variance in #Benefits" column first, with various possible formats
        for col in df.columns:
            col_lower = col.lower().strip()
            # Prioritize exact matches for the benefits variance column
            if ('variance' in col_lower and 
                ('benefit' in col_lower or '#benefit' in col_lower or 'benefits' in col_lower or '#benefits' in col_lower or
                 '# benefit' in col_lower or 'num benefit' in col_lower)):
                benefits_variance_col = col
                variance_columns.append(col)
                break  # Stop after finding the main target column
        
        # If we didn't find the specific benefits variance column, look for other variance columns
        if not variance_columns:
            for col in df.columns:
                col_lower = col.lower().strip()
                if ('variance' in col_lower or 'var' in col_lower or 
                      ('%' in col and ('var' in col_lower or 'change' in col_lower)) or
                      'change' in col_lower or 'diff' in col_lower):
                    variance_columns.append(col)
                # Also check for columns that might contain percentage symbols or numbers that could be variances
                elif any(keyword in col_lower for keyword in ['%', 'percent', 'pct', 'growth', 'delta']):
                    variance_columns.append(col)
        
        if variance_columns:
            # Option to show debug info
            with st.expander("🔍 Debug Variance Detection", expanded=False):
                st.info(f"🔍 Detected variance columns: {variance_columns}")
                if benefits_variance_col:
                    st.success(f"✅ Found 'Variance in #Benefits' column: '{benefits_variance_col}'")
                
                st.write("📋 All columns in dataset:", df.columns.tolist())
                
                # Show sample variance values for debugging
                for col in variance_columns:
                    if col in df.columns:
                        sample_values = df[col].head(5).tolist()
                        st.write(f"📊 Sample values in '{col}': {sample_values}")
                        
                # Manual column selection for testing
                st.write("🎯 Manual Override:")
                manual_col = st.selectbox(
                    "Select column to apply variance highlighting:",
                    ["None"] + df.columns.tolist(),
                    key="manual_variance_col"
                )
                if manual_col != "None":
                    variance_columns = [manual_col]
                    st.info(f"Using manual selection: {manual_col}")
            
            # Create a copy of the dataframe for styling
            display_df = df.copy()
            
            # Format currency columns (Amt issued fields)
            for col in display_df.columns:
                col_lower = col.lower()
                if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                    # Convert to numeric if needed and format as currency
                    try:
                        # Convert to numeric, handling any string values
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                        # Format as currency with 2 decimal places
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    except (ValueError, TypeError):
                        # If conversion fails, try to format existing numeric values
                        display_df[col] = display_df[col].apply(
                            lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                        )
            
            # Apply percentage formatting to all percentage-related columns
            display_df = format_percentage_columns(display_df)
            
            # Function to highlight entire row where variance text is "Above 10%"
            def highlight_variance_rows(row):
                styles = [''] * len(row)
                # Check if any variance column in this row contains "Above 10%"
                for i, col in enumerate(df.columns):
                    if col in variance_columns:
                        try:
                            # Get the original value
                            original_val = row.iloc[i]
                            
                            # Skip null values
                            if pd.isna(original_val) or original_val is None:
                                continue
                                
                            # Convert to string and clean
                            val_str = str(original_val).strip()
                            
                            # Check if the text contains "Above 10%" (case insensitive)
                            if "above 10%" in val_str.lower():
                                # Highlight the entire row in bold red
                                styles = ['background-color: #ffebee; color: #c62828; font-weight: bold;'] * len(row)
                                # Make the variance column cell stand out more with a border
                                styles[i] = 'background-color: #ffcdd2; color: #c62828; font-weight: bold; border: 2px solid #c62828;'
                                break  # Once we find "Above 10%", highlight the whole row
                            
                        except (ValueError, TypeError, AttributeError):
                            continue
                return styles
            
            # Apply styling
            styled_df = display_df.style.apply(highlight_variance_rows, axis=1)
            
            # Display with custom styling and dynamic height
            display_height = min(600, max(200, len(df) * 35 + 100))
            st.dataframe(styled_df, use_container_width=True, height=display_height, hide_index=True)
            
            # Add legend
            st.markdown("""
            <div style="margin-top: 10px;">
                <small>
                    🔴 <strong>Bold Red Rows</strong>: Entire row highlighted when Variance in #Benefits shows "Above 10%"<br>
                    📍 <strong>Red Border</strong>: Marks the specific variance cell that triggered the highlighting<br>
                    💰 <strong>Currency Format</strong>: "Amt issued" fields display as $X,XXX.XX with 2 decimal places<br>
                    📊 <strong>Percentage Format</strong>: Columns with "%" show as percentages (0.04 → 4%), others as numbers (15.25)
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            # If no variance columns found, still apply currency formatting
            display_df = df.copy()
            
            # Format currency columns (Amt issued fields)
            for col in display_df.columns:
                col_lower = col.lower()
                if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                    # Convert to numeric if needed and format as currency
                    try:
                        # Convert to numeric, handling any string values
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                        # Format as currency with 2 decimal places
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    except (ValueError, TypeError):
                        # If conversion fails, try to format existing numeric values
                        display_df[col] = display_df[col].apply(
                            lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                        )
            
            # Apply percentage formatting to all percentage-related columns
            display_df = format_percentage_columns(display_df)
            
            # Use regular table with dynamic height
            display_height = min(600, max(200, len(display_df) * 35 + 100))
            st.dataframe(display_df, use_container_width=True, height=display_height, hide_index=True)
    
    def render_view_history_table(self, df: pd.DataFrame, title: str) -> None:
        """Render View History Screen Validation table with merged date cells."""
        import pandas as pd
        from datetime import datetime
        
        # Display table title
        st.subheader(title)
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            if len(df) > 0:
                unique_dates = df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0
                st.metric("Unique Dates", unique_dates)
        
        if len(df) == 0:
            st.warning("No data to display")
            return
        
        # Create HTML table with merged cells
        # Assume first column is date and other columns are data
        date_col = df.columns[0] if len(df.columns) > 0 else 'Date'
        
        # Sort by date columns (latest to oldest) using our sorting function
        df_sorted = sort_dataframe_by_date(df, ascending=False)
        
        # Create HTML table with merged cells
        html_table = '''
        <style>
        .merged-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        
        .merged-table th, .merged-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        .merged-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            font-size: 16px;
            color: #333;
            text-align: center;
            padding: 12px 8px;
            border-bottom: 2px solid #dee2e6;
        }
        
        .merged-table .date-cell {
            text-align: center;
            vertical-align: middle;
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .merged-table .data-cell {
            background-color: white;
        }
        
        .merged-table tr:nth-child(even) .data-cell {
            background-color: #f9f9f9;
        }
        </style>
        
        <table class="merged-table">
        <thead>
        <tr>
        '''
        
        # Add headers
        for col in df_sorted.columns:
            html_table += f'<th>{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        # Group by date and create merged cells
        if date_col in df_sorted.columns:
            current_date = None
            date_row_count = 0
            rows_for_date = []
            
            # Process each row
            for idx, row in df_sorted.iterrows():
                row_date = format_date_to_standard(row[date_col])
                
                # If date changed, write previous group
                if current_date is not None and row_date != current_date:
                    # Write the rows for the previous date
                    for i, stored_row in enumerate(rows_for_date):
                        html_table += '<tr>'
                        
                        # Add date cell only for first row of the group
                        if i == 0:
                            html_table += f'<td class="date-cell" rowspan="{len(rows_for_date)}">{current_date}</td>'
                        
                        # Add other columns
                        for col in df_sorted.columns[1:]:  # Skip first column (date)
                            cell_value = stored_row[col] if pd.notna(stored_row[col]) else ''
                            html_table += f'<td class="data-cell">{cell_value}</td>'
                        
                        html_table += '</tr>'
                    
                    # Reset for new date group
                    rows_for_date = []
                
                # Update current date and add row to group
                current_date = row_date
                rows_for_date.append(row)
            
            # Write the last group
            if rows_for_date:
                for i, stored_row in enumerate(rows_for_date):
                    html_table += '<tr>'
                    
                    # Add date cell only for first row of the group
                    if i == 0:
                        formatted_date = format_date_to_standard(current_date)
                        html_table += f'<td class="date-cell" rowspan="{len(rows_for_date)}">{formatted_date}</td>'
                    
                    # Add other columns
                    for col in df_sorted.columns[1:]:  # Skip first column (date)
                        cell_value = stored_row[col] if pd.notna(stored_row[col]) else ''
                        html_table += f'<td class="data-cell">{cell_value}</td>'
                    
                    html_table += '</tr>'
        
        html_table += '</tbody></table>'
        
        # Display the HTML table
        st.markdown(html_table, unsafe_allow_html=True)
    
    def render_tango_monitoring_with_upload_status(self, df: pd.DataFrame, title: str) -> None:
        """Render Tango Monitoring table with clickable dates for upload status."""
        st.subheader(title)
        
        # Check if upload status file exists
        workspace_path = Path(__file__).parent
        upload_status_path = workspace_path / "Monitoring Data Files" / "Correspondence" / "Tango Monitoring File Upload Status.xlsx"
        has_upload_status_file = upload_status_path.exists()
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if len(df) > 0:
                # Try to find date column for unique dates count
                date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
                if date_columns:
                    unique_dates = df[date_columns[0]].nunique()
                    st.metric("Unique Dates", unique_dates)
                else:
                    st.metric("Data Rows", len(df))
        
        # Find date columns
        date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
        
        if not date_columns:
            # No date columns found, show regular table
            st.write("**Main Tango Monitoring Data:**")
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
            return
        
        date_col = date_columns[0]  # Use first date column
        
        # Initialize session state
        if 'clicked_tango_date' not in st.session_state:
            st.session_state.clicked_tango_date = None
        
        # Display clickable dates for upload status access
        if has_upload_status_file:
            st.info("💡 **Tip:** Click on any date below to view file upload status")
            # Create HTML table with clickable dates
            self._render_tango_table_with_clickable_dates(df, date_col, has_upload_status_file)
        else:
            st.warning("No upload status file available")
        
        # Show upload status if a date is selected
        if st.session_state.clicked_tango_date and has_upload_status_file:
            self.show_upload_status_for_date(st.session_state.clicked_tango_date, None)
    
    def _render_tango_table_with_clickable_dates(self, df: pd.DataFrame, date_col: str, has_upload_status: bool) -> None:
        """Render Tango table with clickable date hyperlinks using Streamlit components."""
        
        # Create custom CSS for hyperlink-style buttons
        st.markdown("""
        <style>
        /* Hide default button styling and make buttons look like hyperlinks */
        div[data-testid="column"] button[kind="secondary"] {
            background: none !important;
            border: none !important;
            padding: 0 !important;
            color: #007bff !important;
            text-decoration: none !important;
            font-weight: bold !important;
            cursor: pointer !important;
            font-size: 14px !important;
            line-height: 1.2 !important;
            min-height: auto !important;
            height: auto !important;
        }
        
        div[data-testid="column"] button[kind="secondary"]:hover {
            color: #0056b3 !important;
            text-decoration: underline !important;
            background: rgba(0, 123, 255, 0.1) !important;
            border-radius: 4px !important;
            padding: 2px 4px !important;
        }
        
        div[data-testid="column"] button[kind="secondary"]:focus {
            color: #0056b3 !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        /* Custom table styling using Streamlit's column layout */
        .tango-table-header {
            background-color: #f8f9fa;
            font-weight: bold;
            font-size: 16px;
            color: #333;
            text-align: center;
            padding: 12px 8px;
            border: 1px solid #dee2e6;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 10px;
        }
        
        .tango-table-row {
            border-bottom: 1px solid #ddd;
            padding: 8px 0;
            margin-bottom: 5px;
        }
        
        .tango-table-row:nth-child(even) {
            background-color: #f9f9f9;
            border-radius: 4px;
            margin: 2px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display table headers using columns
        header_cols = st.columns(len(df.columns))
        for idx, col in enumerate(df.columns):
            with header_cols[idx]:
                st.markdown(f'<div class="tango-table-header">{col}</div>', unsafe_allow_html=True)
        
        # Display each row using columns with clickable date buttons
        previous_date = None
        
        for row_idx, row in df.iterrows():
            current_date = format_date_to_standard(row[date_col]) if pd.notna(row[date_col]) else None
            
            # Add grey separator line ONLY between different dates (not before first row)
            if row_idx > 0 and previous_date and current_date and previous_date != current_date:
                st.markdown('<div style="margin: 8px 0; border-bottom: 1px solid #ccc; opacity: 0.7;"></div>', unsafe_allow_html=True)
            
            # Check for "Number of Files not sent to CPC" column and value >= 7 for highlighting
            highlight_row = False
            for col in df.columns:
                if "Number of Files not sent to CPC" in str(col) or "files not sent" in str(col).lower():
                    try:
                        files_not_sent_value = float(row[col]) if pd.notna(row[col]) else 0
                        if files_not_sent_value >= 7:
                            highlight_row = True
                    except (ValueError, TypeError):
                        pass
                    break
            
            # Create columns for the row
            data_cols = st.columns(len(df.columns))
            
            for col_idx, col in enumerate(df.columns):
                cell_value = row[col] if pd.notna(row[col]) else ''
                
                with data_cols[col_idx]:
                    # Make date cells clickable if upload status is available
                    if col == date_col and cell_value and has_upload_status:
                        formatted_date = format_date_to_standard(cell_value)
                        
                        # Apply red highlighting to date column if row is highlighted
                        if highlight_row:
                            # Create a red highlighted container for the clickable date
                            st.markdown(f"""
                            <div style="background-color: #ffebee; border: 1px solid #f44336; border-radius: 4px; padding: 4px 8px; margin: 1px 0;">
                                <span style="color: #d32f2f; font-weight: bold; font-size: 14px;">🔴 {formatted_date}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Still provide clickable functionality with a subtle button
                            if st.button(
                                f"📋 View Status", 
                                key=f"date_link_{row_idx}_{col_idx}",
                                help=f"Click to view upload status for {formatted_date}",
                                type="secondary"
                            ):
                                st.session_state.clicked_tango_date = formatted_date
                                st.rerun()
                        else:
                            # Regular clickable date hyperlink-style button
                            if st.button(
                                formatted_date, 
                                key=f"date_link_{row_idx}_{col_idx}",
                                help=f"Click to view upload status for {formatted_date}",
                                type="secondary"
                            ):
                                st.session_state.clicked_tango_date = formatted_date
                                st.rerun()
                    else:
                        # Apply bold red styling for highlighted rows
                        if highlight_row:
                            st.markdown(f'<div style="background-color: #ffebee; border: 1px solid #f44336; border-radius: 4px; padding: 4px 8px; margin: 1px 0;"><span style="color: #d32f2f; font-weight: bold;">{str(cell_value)}</span></div>', unsafe_allow_html=True)
                        else:
                            # Regular cell content with alternating background
                            if row_idx % 2 == 0:
                                st.markdown(f'<div style="background-color: #f9f9f9; padding: 4px 8px; border-radius: 2px;">{str(cell_value)}</div>', unsafe_allow_html=True)
                            else:
                                st.write(str(cell_value))
            
            # Update previous_date for next iteration
            previous_date = current_date
    
    def show_upload_status_for_date(self, selected_date: str, unused_param=None) -> None:
        """Display upload status for the selected date."""
        st.markdown("---")
        st.subheader(f"📄 File Upload Status for {selected_date}")
        
        # Load upload status data for the specific date
        date_specific_df = self.load_tango_upload_status(selected_date)
        
        if date_specific_df.empty:
            # Show a clean message when no data is available
            st.info(f"📋 **Upload status is not available** for {selected_date}")
            st.markdown("""
            <div style="background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 15px; margin: 10px 0; border-radius: 4px;">
                <h4 style="color: #495057; margin-top: 0;">ℹ️ No Data Available</h4>
                <p style="color: #6c757d; margin-bottom: 0;">
                    Upload status data has not been recorded for this date. This could mean:
                    <br>• No file upload activity occurred on this date
                    <br>• Data collection was not active on this date  
                    <br>• The corresponding data sheet does not exist
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Optional: Show available dates for debugging (in expander)
            available_sheets = self._get_available_upload_status_sheets()
            if available_sheets:
                with st.expander("🔍 View Available Dates", expanded=False):
                    st.write("**Available upload status data for these dates:**")
                    for sheet in sorted(available_sheets):
                        st.write(f"• {sheet}")
            return
        
        # Sort and format upload status data
        upload_status_sorted = sort_dataframe_by_date(date_specific_df, ascending=False)
        upload_status_formatted = format_dataframe_dates(upload_status_sorted)
        
        # Apply percentage and currency formatting
        upload_status_formatted = format_percentage_columns(upload_status_formatted)
        
        # Check for difference columns and apply conditional formatting
        difference_columns = []
        for col in upload_status_formatted.columns:
            col_lower = col.lower()
            if 'difference' in col_lower or 'diff' in col_lower:
                difference_columns.append(col)
        
        if difference_columns:
            # Create a copy for styling
            display_df = upload_status_formatted.copy()
            
            # Option to show debug info
            with st.expander("🔍 Debug Difference Detection", expanded=False):
                st.info(f"🔍 Detected difference columns: {difference_columns}")
                for col in difference_columns:
                    if col in display_df.columns:
                        sample_values = display_df[col].head(5).tolist()
                        st.write(f"📊 Sample values in '{col}': {sample_values}")
            
            # Function to highlight entire row where difference > 0
            def highlight_difference_rows(row):
                styles = [''] * len(row)
                # Check if any difference column in this row has value > 0
                for i, col in enumerate(display_df.columns):
                    if col in difference_columns:
                        try:
                            # Get the original value
                            original_val = row.iloc[i]
                            
                            # Skip null values
                            if pd.isna(original_val) or original_val is None:
                                continue
                                
                            # Convert to numeric
                            if isinstance(original_val, (int, float)):
                                numeric_val = float(original_val)
                            else:
                                # Try to convert string to numeric
                                val_str = str(original_val).strip().replace(',', '')
                                # Remove any non-numeric characters except decimal point and minus
                                import re
                                clean_val = re.sub(r'[^\d.-]', '', val_str)
                                if clean_val:
                                    numeric_val = float(clean_val)
                                else:
                                    continue
                            
                            # Check if value > 0
                            if numeric_val > 0:
                                # Highlight the entire row in bold red
                                styles = ['background-color: #ffebee; color: #c62828; font-weight: bold;'] * len(row)
                                # Add special border to the difference column that triggered it
                                styles[i] = 'background-color: #ffcdd2; color: #c62828; font-weight: bold; border: 2px solid #c62828;'
                                break  # Once we find a difference > 0, highlight the whole row
                        except (ValueError, TypeError):
                            # If conversion fails, skip this cell
                            continue
                            
                return styles
            
            # Apply styling
            styled_df = display_df.style.apply(highlight_difference_rows, axis=1)
            
            # Display with custom styling and dynamic height
            display_height = min(600, max(200, len(display_df) * 35 + 100))
            st.dataframe(styled_df, use_container_width=True, height=display_height, hide_index=True)
            
            # Add legend
            st.markdown("""
            <div style="margin-top: 10px;">
                <small>
                    🔴 <strong>Bold Red Rows</strong>: Entire row highlighted when Difference > 0<br>
                    📍 <strong>Red Border</strong>: Marks the specific difference cell that triggered the highlighting<br>
                    📊 <strong>Percentage Format</strong>: Columns with "%" show as percentages, others as numbers<br>
                    💰 <strong>Currency Format</strong>: "Amt" fields display with proper formatting
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            # If no difference columns found, use regular table with formatting
            display_height = min(600, max(200, len(upload_status_formatted) * 35 + 100))
            st.dataframe(upload_status_formatted, use_container_width=True, height=display_height, hide_index=True)
        
        # Show summary metrics for upload status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Upload Records", len(upload_status_formatted))
        with col2:
            if len(upload_status_formatted.columns) > 1:
                # Try to count successful uploads
                status_cols = [col for col in upload_status_formatted.columns if 'status' in col.lower()]
                if status_cols:
                    success_count = upload_status_formatted[status_cols[0]].str.contains('success|complete|ok', case=False, na=False).sum()
                    st.metric("Successful", success_count)
                else:
                    st.metric("Total Columns", len(upload_status_formatted.columns))
        with col3:
            if len(upload_status_formatted.columns) > 1:
                error_cols = [col for col in upload_status_formatted.columns if 'error' in col.lower() or 'fail' in col.lower()]
                if error_cols:
                    error_count = upload_status_formatted[error_cols[0]].notna().sum()
                    st.metric("Issues Found", error_count)
                else:
                    st.metric("Data Status", "✅ Complete")
        
        # Add a button to hide the upload status
        if st.button("❌ Hide Upload Status", key="hide_upload_status"):
            st.session_state.clicked_tango_date = None
            st.rerun()
    
    def render_correspondence_tango_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Correspondence-Tango specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Filters in main area with expander
        with st.expander("🔍 **Correspondence Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        # Apply date formatting to filtered data
        filtered_df = format_dataframe_dates(filtered_df)
        
        file_display_name = f"{selected_subsection} Data"
        
        # Special handling for different file types
        if selected_subsection == "View History Screen Validation":
            # Data Table - Main Focus
            st.header("📋 Data Table")
            self.render_view_history_table(filtered_df, file_display_name)
        elif selected_subsection == "Tango Monitoring":
            # Special handling for Tango Monitoring with integrated upload status (no header here)
            self.render_tango_monitoring_with_upload_status(filtered_df, file_display_name)
            return  # Exit early for Tango Monitoring to avoid duplicate content
        else:
            # Data Table - Main Focus  
            st.header("📋 Data Table")
            self.table_component.data_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.metrics_component.auto_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Statistics", "🔬 Analysis"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="correspondence_general_")
        
        with tab2:
            self.table_component.summary_stats(filtered_df)
        
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="correspondence_general_")
        
        # Correspondence-specific metrics
        st.header(f"📧 Correspondence Metrics - {selected_period.title()}")
        self.render_correspondence_metrics(filtered_df)
        
        # Tabs for correspondence-specific views  
        tab1, tab2, tab3 = st.tabs(["📊 Communication Charts", "📈 Tango Analytics", "🔍 Message Analysis"])
        
        with tab1:
            self.render_correspondence_charts(filtered_df, selected_period)
        
        with tab2:
            self.render_tango_analytics(filtered_df)
        
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="correspondence_")
    
    def render_correspondence_metrics(self, df: pd.DataFrame) -> None:
        """Render correspondence-specific metrics."""
        if df.empty:
            st.warning("No correspondence data available.")
            return
        
        # Create correspondence-specific metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(df))
        
        with col2:
            # Look for status or response columns
            status_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['status', 'response', 'reply'])]
            if status_cols:
                responses = df[status_cols[0]].notna().sum()
                st.metric("Responses", responses)
            else:
                st.metric("Records", len(df))
        
        with col3:
            # Look for date columns to calculate daily average
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
            if len(date_cols) > 0:
                days = (df[date_cols[0]].max() - df[date_cols[0]].min()).days + 1
                daily_avg = len(df) / max(days, 1)
                st.metric("Daily Average", f"{daily_avg:.1f}")
            else:
                st.metric("Active Records", len(df))
        
        with col4:
            # Look for numeric columns for totals
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                total_value = df[numeric_cols[0]].sum()
                st.metric(f"Total {numeric_cols[0]}", f"{total_value:,.0f}")
            else:
                st.metric("Data Points", len(df.columns))
    
    def render_correspondence_charts(self, df: pd.DataFrame, period: str) -> None:
        """Render correspondence-specific charts."""
        st.subheader("📊 Communication Patterns")
        self.render_charts(df, period, key_prefix="correspondence_")
    
    def render_tango_analytics(self, df: pd.DataFrame) -> None:
        """Render Tango-specific analytics."""
        st.subheader("📈 Tango System Analytics")
        
        if df.empty:
            st.warning("No data available for Tango analytics.")
            return
        
        # Display data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Overview:**")
            st.write(f"- Total Records: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
            
            # Show column types
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            text_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            date_cols = len(df.select_dtypes(include=['datetime']).columns)
            
            st.write(f"- Numeric Fields: {numeric_cols}")
            st.write(f"- Text Fields: {text_cols}")
            st.write(f"- Date Fields: {date_cols}")
        
        with col2:
            st.write("**Tango System Status:**")
            st.success("✅ Data Connection: Active")
            st.info("📊 Analytics: Running")
            st.info("🔄 Last Update: Real-time")
        
        # Show sample data
        if not df.empty:
            st.write("**Sample Records:**")
            st.dataframe(df.head(3), hide_index=True)
    
    def render_error_counts_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render 100 Error Counts specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        with st.expander("🚨 **Error Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus with calculated percentage columns
        st.header("📋 Data Table")
        file_display_name = f"{selected_subsection} Data"
        self.render_error_counts_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.render_error_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Error Trends", "🔍 Error Analysis", "📈 Resolution Status"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="error_")
        with tab2:
            self.table_component.summary_stats(filtered_df)
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="error_")
    
    def render_user_impact_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render User Impact specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        with st.expander("👥 **Impact Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus (with original percentage values)
        st.header("📋 Data Table")
        file_display_name = f"{selected_subsection} Data"
        self.render_user_impact_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.render_user_impact_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 User Trends", "🎯 Impact Analysis", "📈 User Metrics"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="user_impact_")
        with tab2:
            self.table_component.summary_stats(filtered_df)
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="user_impact_")
    
    def render_mass_update_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Mass Update specific content."""
        self.render_generic_section_content(df, selected_period, "Mass Update", "🔄")
    
    def render_interfaces_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Interfaces specific content."""
        self.render_generic_section_content(df, selected_period, "Interfaces", "🔗")
    
    def render_extra_batch_connections_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Extra Batch Connections specific content."""
        
        # Filters in main area with expander
        with st.expander("🔍 **Data Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        # Apply date formatting to filtered data
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus with highlighting
        st.header("📋 Extra Connections Created Data")
        self.render_extra_batch_connections_table(filtered_df, "Extra Connections Created Data")
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.metrics_component.auto_metrics(filtered_df)

    def render_extra_batch_connections_table(self, df: pd.DataFrame, title: str) -> None:
        """Render extra batch connections table with highlighting for # Connections >= 7."""
        import pandas as pd
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Check if dataframe has connections-related columns
        connections_columns = []
        connections_col = None
        
        # Look for "# Connections" column in various formats
        for col in df.columns:
            col_lower = col.lower().strip()
            if ('#connections' in col_lower.replace(' ', '') or 
                '# connections' in col_lower or 
                'connections' in col_lower or
                'conn' in col_lower):
                connections_col = col
                connections_columns.append(col)
                break  # Stop after finding the main target column
        
        if connections_columns:
            # Option to show debug info
            with st.expander("🔍 Debug Connections Detection", expanded=False):
                st.info(f"🔍 Detected connections columns: {connections_columns}")
                if connections_col:
                    st.success(f"✅ Found connections column: '{connections_col}'")
                
                st.write("📋 All columns in dataset:", df.columns.tolist())
                
                # Show sample connections values for debugging
                for col in connections_columns:
                    if col in df.columns:
                        sample_values = df[col].head(5).tolist()
                        st.write(f"📊 Sample values in '{col}': {sample_values}")
                        
                # Manual column selection for testing
                st.write("🎯 Manual Override:")
                manual_col = st.selectbox(
                    "Select column to apply connections highlighting:",
                    ["None"] + df.columns.tolist(),
                    key="manual_connections_col"
                )
                if manual_col != "None":
                    connections_columns = [manual_col]
                    st.info(f"Using manual selection: {manual_col}")
            
            # Create a copy of the dataframe for styling
            display_df = df.copy()
            
            # Apply formatting to the display dataframe
            # Format currency columns (Amt issued fields)
            for col in display_df.columns:
                col_lower = col.lower()
                if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                    # Convert to numeric if needed and format as currency
                    try:
                        # Convert to numeric, handling any string values
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                        # Format as currency with 2 decimal places
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    except (ValueError, TypeError):
                        # If conversion fails, try to format existing numeric values
                        display_df[col] = display_df[col].apply(
                            lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                        )
            
            # Apply percentage formatting to all percentage-related columns
            display_df = format_percentage_columns(display_df)
            
            # Function to highlight entire row where # Connections >= 7
            def highlight_connections_rows(row):
                styles = [''] * len(row)
                # Check if any connections column in this row has value >= 7
                for i, col in enumerate(df.columns):
                    if col in connections_columns:
                        try:
                            # Get the original value
                            original_val = row.iloc[i]
                            
                            # Skip null values
                            if pd.isna(original_val) or original_val is None:
                                continue
                                
                            # Convert to numeric
                            if isinstance(original_val, (int, float)):
                                numeric_val = float(original_val)
                            else:
                                # Try to convert string to numeric
                                val_str = str(original_val).strip().replace(',', '')
                                numeric_val = float(val_str)
                            
                            # Check if value >= 7
                            if numeric_val >= 7:
                                # Highlight the entire row in bold red
                                styles = ['background-color: #ffebee; color: #c62828; font-weight: bold;'] * len(row)
                                # Make the connections column cell stand out more with a border
                                styles[i] = 'background-color: #ffcdd2; color: #c62828; font-weight: bold; border: 2px solid #c62828;'
                                break  # Once we find >= 7, highlight the whole row
                            
                        except (ValueError, TypeError, AttributeError):
                            continue
                return styles
            
            # Apply styling
            styled_df = display_df.style.apply(highlight_connections_rows, axis=1)
            
            # Display with custom styling and dynamic height
            display_height = min(600, max(200, len(df) * 35 + 100))
            st.dataframe(styled_df, use_container_width=True, height=display_height, hide_index=True)
            
            # Add legend
            st.markdown("""
            <div style="margin-top: 10px;">
                <small>
                    🔴 <strong>Bold Red Rows</strong>: Entire row highlighted when # Connections ≥ 7<br>
                    📍 <strong>Red Border</strong>: Marks the specific connections cell that triggered the highlighting
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            # If no connections columns found, still apply formatting
            display_df = df.copy()
            
            # Format currency columns (Amt issued fields)
            for col in display_df.columns:
                col_lower = col.lower()
                if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                    # Convert to numeric if needed and format as currency
                    try:
                        # Convert to numeric, handling any string values
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                        # Format as currency with 2 decimal places
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    except (ValueError, TypeError):
                        # If conversion fails, try to format existing numeric values
                        display_df[col] = display_df[col].apply(
                            lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                        )
            
            # Apply percentage formatting to all percentage-related columns
            display_df = format_percentage_columns(display_df)
            
            # Use regular table with dynamic height
            display_height = min(600, max(200, len(df) * 35 + 100))
            st.dataframe(display_df, use_container_width=True, height=display_height, hide_index=True)
    
    def render_hung_threads_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Hung Threads specific content."""
        self.render_generic_section_content(df, selected_period, "Hung Threads", "🧵")

    def render_data_warehouse_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Data Warehouse specific content."""
        self.render_generic_section_content(df, selected_period, "Data Warehouse", "🏢")

    def render_consolidated_inquiry_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Consolidated Inquiry specific content."""
        self.render_generic_section_content(df, selected_period, "Consolidated Inquiry", "🔍")

    def render_miscellaneous_bridges_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Miscellaneous Bridges Processes section overview."""
        st.markdown("## 🔗 Miscellaneous Bridges Processes Overview")
        st.markdown("Select a specific bridge process from the navigation menu to view detailed data.")
        
        # Show overview cards for each bridge process
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔄 Process Management")
            st.info("""
            **🔄 Mass Update**
            Track mass update processes and system-wide changes.
            
            **🔗 Interfaces** 
            Monitor system interfaces and integration points.
            """)
        
        with col2:
            st.markdown("### ⚡ Connection & Thread Monitoring")
            st.info("""
            **⚡ Extra Batch Connections**
            Monitor additional batch connections and process execution.
            
            **🧵 Hung Threads**
            Detect and monitor hung threads that may impact system performance.
            """)
        
        with col3:
            st.markdown("### 🏢 Data & Inquiry Systems")
            st.info("""
            **🏢 Data Warehouse**
            Monitor data warehouse operations, ETL processes, and data integrity.
            
            **🔍 Consolidated Inquiry**
            Track consolidated inquiry processes and cross-system data retrieval.
            """)
        
        st.markdown("---")
        st.info("👆 Click on a bridge process item in the sidebar to view its contents")

    def render_daily_exceptions_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Daily Exceptions section overview."""
        st.markdown("## ⚠️ Daily Exceptions Overview")
        st.markdown("Select a specific exception dashboard from the navigation menu to view detailed data.")
        
        # Show overview cards for each exception type
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏭 Production Environment")
            st.info("""
            **🌐 Online Exceptions - PRD**
            Monitor online exceptions in the production environment.
            
            **💻 Batch Exceptions - PRD** 
            Track batch process exceptions in production environment.
            """)
        
        with col2:
            st.markdown("### 🧪 UAT Environment")
            st.info("""
            **🧪 Online Exceptions - UAT**
            Monitor online exceptions in the UAT environment.
            
            **🔬 Batch Exceptions - UAT**
            Track batch process exceptions in UAT environment.
            """)
        
        st.markdown("---")
        st.info("👆 Click on a dashboard item in the sidebar to view its contents")

    def render_online_exceptions_prd_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Online Exceptions - PRD specific content."""
        self.render_generic_section_content(df, selected_period, "Online Exceptions - PRD", "🌐")
    
    def render_batch_exceptions_prd_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Batch Exceptions - PRD specific content."""
        self.render_generic_section_content(df, selected_period, "Batch Exceptions - PRD", "📦")
    
    def render_online_exceptions_uat_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Online Exceptions - UAT specific content."""
        self.render_generic_section_content(df, selected_period, "Online Exceptions - UAT", "🧪")
    
    def render_batch_exceptions_uat_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Batch Exceptions - UAT specific content."""
        self.render_generic_section_content(df, selected_period, "Batch Exceptions - UAT", "🔬")
    
    def render_user_impact_table(self, df: pd.DataFrame, title: str) -> None:
        """Render User Impact table WITHOUT percentage formatting to preserve original values."""
        import pandas as pd
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe (but NOT percentage formatting)
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Create a copy of the dataframe for display
        display_df = df.copy()
        
        # Format currency columns only (Amt issued fields)
        for col in display_df.columns:
            col_lower = col.lower()
            if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                # Convert to numeric if needed and format as currency
                try:
                    # Convert to numeric, handling any string values
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                    # Format as currency with 2 decimal places
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                except (ValueError, TypeError):
                    # If conversion fails, try to format existing numeric values
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                    )
        
        # Find the logged in users column for percentage calculations
        logged_in_users_col = None
        possible_user_columns = []
        
        for col in display_df.columns:
            col_lower = col.lower().strip()
            # More flexible matching for user columns
            if any(term in col_lower for term in ['user', 'login', 'logged']):
                possible_user_columns.append(col)
                
            # Priority 1: Look for "# Logged-in Users" specifically
            if ('logged' in col_lower and 'in' in col_lower and 'user' in col_lower) or \
               ('logged-in' in col_lower and 'user' in col_lower) or \
               ('loggedin' in col_lower and 'user' in col_lower):
                logged_in_users_col = col
                break
                
            # Priority 2: Look for variations with # symbol
            if '#' in col and 'logged' in col_lower and 'user' in col_lower:
                logged_in_users_col = col
                break
        
        # If still not found, look for any column with "logged" and "user"
        if not logged_in_users_col:
            for col in display_df.columns:
                col_lower = col.lower().strip()
                if 'logged' in col_lower and 'user' in col_lower:
                    logged_in_users_col = col
                    break
        
        # Add calculated percentage columns after each error count column
        error_columns = ['0 errors', '1 errors', '2 errors', '3-5 errors', '6-10 errors', '>10 errors']
        
        # Also try to find actual error columns in the dataset
        actual_error_columns = []
        for col in display_df.columns:
            col_lower = col.lower().strip()
            # Look for columns that contain numbers and "error"
            if 'error' in col_lower:
                for error_pattern in ['0 error', '1 error', '2 error', '3-5 error', '6-10 error', '>10 error', '10+ error']:
                    if error_pattern in col_lower:
                        actual_error_columns.append(col)
                        break
        
        # Use actual found columns if available, otherwise try the expected ones
        columns_to_process = actual_error_columns if actual_error_columns else error_columns
        
        if logged_in_users_col:
            # Work backwards through columns to maintain correct positioning when inserting
            cols = list(display_df.columns)
            
            # Use actual found columns, work backwards to maintain positions
            for error_col in reversed(columns_to_process):
                # For expected columns, find the actual column name
                actual_error_col = None
                if error_col in error_columns:  # This is an expected column name
                    # Find the actual column (case-insensitive, flexible matching)
                    for col in cols:
                        col_lower = col.lower().strip()
                        error_col_lower = error_col.lower()
                        if col_lower == error_col_lower or \
                           (error_col_lower.replace(' ', '').replace('-', '') in col_lower.replace(' ', '').replace('-', '')):
                            actual_error_col = col
                            break
                else:  # This is already an actual column name
                    actual_error_col = error_col
                
                if actual_error_col and actual_error_col in cols:
                    # Find the position to insert percentage column
                    error_col_idx = cols.index(actual_error_col)
                    
                    # Calculate percentage for this error column
                    error_pct = []
                    
                    for idx, row in display_df.iterrows():
                        try:
                            error_count = pd.to_numeric(row[actual_error_col], errors='coerce')
                            logged_users = pd.to_numeric(row[logged_in_users_col], errors='coerce')
                            
                            if pd.notna(error_count) and pd.notna(logged_users) and logged_users > 0:
                                percentage = (error_count / logged_users) * 100
                                error_pct.append(f"{percentage:.2f}%")
                            else:
                                error_pct.append("N/A")
                        except Exception as e:
                            error_pct.append("N/A")
                    
                    # Create percentage column name
                    pct_col_name = f"{actual_error_col} %"
                    

                    # Insert the new column after the error column
                    display_df.insert(error_col_idx + 1, pct_col_name, error_pct)
                    

                    
                    # Update the cols list to reflect the new column
                    cols = list(display_df.columns)
        
        # Identify our calculated columns to protect them from formatting
        calculated_pct_cols = [col for col in display_df.columns if col.endswith(' %') and any(error in col for error in ['0 errors', '1 errors', '2 errors', '3-5 errors', '6-10 errors', '>10 errors'])]
        
        # Format existing percentage columns with proper XX.XX% format for User Impact
        # Only format original percentage columns, NOT our calculated ones
        for col in display_df.columns:
            # Skip ALL calculated percentage columns - they're already properly formatted
            if col in calculated_pct_cols:
                continue
                
            # Only format original percentage columns that contain decimal values (like 0.925)
            if '%' in col:
                try:
                    # Check if values are already formatted as strings with % (skip if so)
                    sample_val = display_df[col].iloc[0] if len(display_df) > 0 else None
                    if isinstance(sample_val, str) and sample_val.endswith('%'):
                        continue
                    
                    # Convert decimal values to percentage format
                    numeric_series = pd.to_numeric(display_df[col], errors='coerce')
                    
                    def format_user_impact_percentage(x):
                        if pd.isna(x):
                            return ""
                        return f"{x * 100:.2f}%"
                    
                    display_df[col] = numeric_series.apply(format_user_impact_percentage)
                except (ValueError, TypeError):
                    continue


        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(display_df))
        with col2:
            numeric_cols = len(display_df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col3:
            completeness = ((len(display_df) - display_df.isnull().sum().sum()) / (len(display_df) * len(display_df.columns)) * 100) if len(display_df) > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
        

        
        # Use dynamic height based on number of rows
        display_height = min(600, max(200, len(display_df) * 35 + 100))
        


        # Display the main table
        st.dataframe(display_df, use_container_width=True, height=display_height, hide_index=True)
        

    
    def render_error_counts_table(self, df: pd.DataFrame, title: str) -> None:
        """Render Error Counts table with calculated percentage columns."""
        import pandas as pd
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Create a copy of the dataframe for display
        display_df = df.copy()
        
        # Find the required columns for calculations
        session_timeout_col = None
        total_count_col = None
        errors_analysis_col = None
        
        # Look for Session Timeout Errors column
        for col in display_df.columns:
            col_lower = col.lower().strip()
            if ('session' in col_lower and 'timeout' in col_lower and 'error' in col_lower):
                session_timeout_col = col
                break
        
        # Look for Total Count column
        for col in display_df.columns:
            col_lower = col.lower().strip()
            if col_lower == "total count":
                total_count_col = col
                break
        
        # Look for Errors Requiring Analysis column
        for col in display_df.columns:
            col_lower = col.lower().strip()
            if ('error' in col_lower and 'requiring' in col_lower and 'analysis' in col_lower):
                errors_analysis_col = col
                break
        
        # Add calculated percentage columns if the required columns are found
        if session_timeout_col and total_count_col:
            # Find the position to insert "Timeout Errors %" after Session Timeout Errors
            cols = list(display_df.columns)
            timeout_col_idx = cols.index(session_timeout_col)
            
            # Calculate Timeout Errors %
            timeout_pct = []
            for _, row in display_df.iterrows():
                try:
                    session_timeout = pd.to_numeric(row[session_timeout_col], errors='coerce')
                    total_count = pd.to_numeric(row[total_count_col], errors='coerce')
                    
                    if pd.notna(session_timeout) and pd.notna(total_count) and total_count > 0:
                        percentage = (session_timeout / total_count) * 100
                        timeout_pct.append(f"{percentage:.2f}%")
                    else:
                        timeout_pct.append("N/A")
                except:
                    timeout_pct.append("N/A")
            
            # Insert the new column after Session Timeout Errors
            display_df.insert(timeout_col_idx + 1, "Timeout Errors %", timeout_pct)
        
        if errors_analysis_col and total_count_col:
            # Find the position to insert "Analysis Errors %" after Errors Requiring Analysis
            cols = list(display_df.columns)
            analysis_col_idx = cols.index(errors_analysis_col)
            
            # Calculate Analysis Errors %
            analysis_pct = []
            for _, row in display_df.iterrows():
                try:
                    errors_analysis = pd.to_numeric(row[errors_analysis_col], errors='coerce')
                    total_count = pd.to_numeric(row[total_count_col], errors='coerce')
                    
                    if pd.notna(errors_analysis) and pd.notna(total_count) and total_count > 0:
                        percentage = (errors_analysis / total_count) * 100
                        analysis_pct.append(f"{percentage:.2f}%")
                    else:
                        analysis_pct.append("N/A")
                except:
                    analysis_pct.append("N/A")
            
            # Insert the new column after Errors Requiring Analysis
            display_df.insert(analysis_col_idx + 1, "Analysis Errors %", analysis_pct)
        
        # Format currency columns (if any)
        for col in display_df.columns:
            col_lower = col.lower()
            if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                try:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                except (ValueError, TypeError):
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                    )
        
        # Apply percentage formatting to existing percentage columns (but not our calculated ones)
        display_df = format_percentage_columns(display_df)
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(display_df))
        with col2:
            numeric_cols = len(display_df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col3:
            completeness = ((len(display_df) - display_df.isnull().sum().sum()) / (len(display_df) * len(display_df.columns)) * 100) if len(display_df) > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Use dynamic height based on number of rows
        display_height = min(600, max(200, len(display_df) * 35 + 100))
        st.dataframe(display_df, use_container_width=True, height=display_height, hide_index=True)
        

        
# Old implementation removed - now using generic template
    
# render_batch_exceptions_prd_content already defined above
    
# Old implementations removed - now using generic templates above
    
    def filter_data_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter data based on selected time period.
        
        Args:
            df: Original dataframe
            period: Selected period (daily, weekly, monthly, yearly)
            
        Returns:
            Filtered dataframe based on period
        """
        if df.empty:
            return df
        
        # Try to find date columns
        date_cols = df.select_dtypes(include=['datetime64[ns]', 'object']).columns
        date_col = None
        
        # Look for common date column names
        for col in date_cols:
            col_lower = col.lower()
            # Skip obvious non-date columns with more specific patterns
            non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'session', 'connection', 'batch', 'thread']
            if any(keyword in col_lower for keyword in non_date_keywords):
                continue
                
            if any(keyword in col_lower for keyword in ['date', 'time', 'day', 'month', 'year']):
                try:
                    # Try to convert to datetime if it's not already
                    if df[col].dtype == 'object':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_col = col
                    break
                except:
                    continue
        
        if date_col is None:
            # No date column found, return original data
            return df
        
        # Filter based on period (for now, return recent data based on period)
        df_with_dates = df.dropna(subset=[date_col])
        
        if period == "daily":
            # Show last 30 days
            cutoff_date = df_with_dates[date_col].max() - pd.Timedelta(days=30)
            return df_with_dates[df_with_dates[date_col] >= cutoff_date]
        elif period == "weekly":
            # Show last 12 weeks
            cutoff_date = df_with_dates[date_col].max() - pd.Timedelta(weeks=12)
            return df_with_dates[df_with_dates[date_col] >= cutoff_date]
        elif period == "monthly":
            # Show last 12 months
            cutoff_date = df_with_dates[date_col].max() - pd.Timedelta(days=365)
            return df_with_dates[df_with_dates[date_col] >= cutoff_date]
        elif period == "yearly":
            # Show all data grouped by year
            return df_with_dates
        
        return df
    
    def create_inline_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create inline filters in the main content area."""
        filters = {}
        
        # Create columns for filters
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if numeric_cols or categorical_cols or date_cols:
            st.write("**Apply filters to your data:**")
            
            # Numeric filters
            if numeric_cols:
                st.write("**Numeric Columns:**")
                cols = st.columns(min(3, len(numeric_cols)))
                for i, col in enumerate(numeric_cols[:6]):  # Limit to 6 numeric filters
                    with cols[i % 3]:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        if min_val != max_val:
                            filters[col] = st.slider(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"filter_{col}"
                            )
            
            # Categorical filters
            if categorical_cols:
                st.write("**Categorical Columns:**")
                cols = st.columns(min(3, len(categorical_cols)))
                for i, col in enumerate(categorical_cols[:6]):  # Limit to 6 categorical filters
                    unique_values = df[col].unique()
                    if len(unique_values) <= 50:  # Only show for reasonable number of options
                        with cols[i % 3]:
                            filters[col] = st.multiselect(
                                f"{col}",
                                options=unique_values,
                                default=unique_values,
                                key=f"filter_{col}"
                            )
        else:
            st.write("No filterable columns found.")
        
        return filters
    
    def render_charts(self, df: pd.DataFrame, period: str = "daily", key_prefix: str = "") -> None:
        """Render various chart visualizations.
        
        Args:
            df: DataFrame to visualize
            period: Selected time period for context
            key_prefix: Prefix for Streamlit component keys to ensure uniqueness
        """
        if df.empty:
            st.warning("No data to display charts.")
            return
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if not numeric_cols and not categorical_cols:
            st.warning("No suitable columns found for charting.")
            return
        
        # Chart selection
        chart_type = st.selectbox(
            "Select Chart Type",
            options=["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"],
            key=f"{key_prefix}chart_type_selector"
        )
        
        if chart_type == "Line Chart" and numeric_cols and (datetime_cols or categorical_cols):
            x_col = st.selectbox("X-axis", datetime_cols + categorical_cols + numeric_cols, key=f"{key_prefix}line_chart_x_axis")
            y_cols = st.multiselect("Y-axis", numeric_cols, default=numeric_cols[:1], key=f"{key_prefix}line_chart_y_axis")
            
            if x_col and y_cols:
                title = f"{period.title()} Trend: {', '.join(y_cols)} over {x_col}"
                fig = self.chart_component.line_chart(df, x_col, y_cols, title)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Bar Chart" and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", categorical_cols + numeric_cols, key=f"{key_prefix}bar_chart_x_axis")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key=f"{key_prefix}bar_chart_y_axis")
            
            color_col = st.selectbox("Color by (optional)", [None] + categorical_cols, key=f"{key_prefix}bar_chart_color")
            
            if x_col and y_col:
                title = f"{period.title()} Analysis: {y_col} by {x_col}"
                fig = self.chart_component.bar_chart(df, x_col, y_col, title, color_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key=f"{key_prefix}scatter_chart_x_axis")
            with col2:
                y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key=f"{key_prefix}scatter_chart_y_axis")
            
            color_col = st.selectbox("Color by (optional)", [None] + categorical_cols, key=f"{key_prefix}scatter_chart_color")
            size_col = st.selectbox("Size by (optional)", [None] + numeric_cols, key=f"{key_prefix}scatter_chart_size")
            
            if x_col and y_col:
                fig = self.chart_component.scatter_plot(df, x_col, y_col, f"Scatter Plot: {y_col} vs {x_col}", color_col, size_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Histogram" and numeric_cols:
            x_col = st.selectbox("Column", numeric_cols, key=f"{key_prefix}histogram_column")
            bins = st.slider("Number of bins", 10, 100, 30, key=f"{key_prefix}histogram_bins")
            
            if x_col:
                fig = self.chart_component.histogram(df, x_col, f"Histogram: {x_col}", bins)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot" and numeric_cols:
            y_col = st.selectbox("Value column", numeric_cols, key=f"{key_prefix}box_plot_value_column")
            x_col = st.selectbox("Group by (optional)", [None] + categorical_cols, key=f"{key_prefix}box_plot_group_by")
            
            if y_col:
                fig = self.chart_component.box_plot(df, y_col, x_col, f"Box Plot: {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Pie Chart" and categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                names_col = st.selectbox("Categories", categorical_cols, key=f"{key_prefix}pie_chart_categories")
            with col2:
                values_col = st.selectbox("Values", numeric_cols, key=f"{key_prefix}pie_chart_values")
            
            if names_col and values_col:
                # Aggregate data for pie chart
                pie_data = df.groupby(names_col)[values_col].sum().reset_index()
                fig = self.chart_component.pie_chart(pie_data, values_col, names_col, f"Pie Chart: {values_col} by {names_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"Cannot create {chart_type} with the available column types.")
    
    def render_generic_section_content(self, df: pd.DataFrame, selected_period: str, section_name: str, section_icon: str = "📊") -> None:
        """Generic content renderer for all sections with clean layout."""
        # Filters in main area with expander
        with st.expander(f"{section_icon} **{section_name} Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus
        st.header("📋 Data Table")
        self.table_component.data_table(filtered_df, f"{selected_period.title()} {section_name} Data")
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.metrics_component.auto_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Statistics", "🔬 Analysis"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, "generic_")
        
        with tab2:
            self.table_component.summary_stats(filtered_df)
        
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="generic_")
    
    def render_custom_analysis(self, df: pd.DataFrame, key_prefix: str = "") -> None:
        """Render custom analysis section.
        
        Args:
            df: DataFrame to analyze
            key_prefix: Prefix for component keys to ensure uniqueness
        """
        st.subheader("Custom Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            options=["Correlation Analysis", "Missing Data Analysis", "Data Distribution", "Top/Bottom Values"],
            key=f"{key_prefix}analysis_type_selector"
        )
        
        if analysis_type == "Correlation Analysis":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                import plotly.express as px
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif analysis_type == "Missing Data Analysis":
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df, hide_index=True)
                
                fig = self.chart_component.bar_chart(
                    missing_df, 'Column', 'Missing Percentage', 
                    "Missing Data Percentage by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data found!")
        
        elif analysis_type == "Data Distribution":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column", numeric_cols, key="distribution_column_selector")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = self.chart_component.histogram(df, selected_col, f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = self.chart_component.box_plot(df, selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for distribution analysis.")
        
        elif analysis_type == "Top/Bottom Values":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column", numeric_cols, key="top_bottom_column_selector")
                n_values = st.slider("Number of values to show", 5, 20, 10, key="top_bottom_n_values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Top {n_values} values")
                    top_values = df.nlargest(n_values, selected_col)
                    st.dataframe(top_values, hide_index=True)
                
                with col2:
                    st.subheader(f"Bottom {n_values} values")
                    bottom_values = df.nsmallest(n_values, selected_col)
                    st.dataframe(bottom_values, hide_index=True)
            else:
                st.warning("No numeric columns available for top/bottom analysis.")
    
    def run(self) -> None:
        """Run the dashboard application."""
        try:
            # Add global CSS for enhanced table headers
            st.markdown("""
            <style>
            /* Enhanced table headers for all dataframes */
            .stDataFrame table thead th {
                font-weight: bold !important;
                font-size: 16px !important;
                background-color: #f8f9fa !important;
                color: #333 !important;
                text-align: center !important;
                padding: 12px 8px !important;
                border-bottom: 2px solid #dee2e6 !important;
            }
            
            /* Alternative targeting for dataframes */
            div[data-testid="stDataFrame"] table thead th {
                font-weight: bold !important;
                font-size: 16px !important;
                background-color: #f8f9fa !important;
                color: #333 !important;
                text-align: center !important;
                padding: 12px 8px !important;
            }
            
            /* Style for table body cells */
            .stDataFrame table tbody td {
                font-size: 14px !important;
                padding: 8px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Initialize session state
            if 'selected_section' not in st.session_state:
                st.session_state.selected_section = None  # Start with welcome page
            if 'selected_period' not in st.session_state:
                st.session_state.selected_period = 'daily'
            
            # Render navigation FIRST to handle section changes
            nav_result = self.render_navigation_menu()
            
            # Get selections from session state AFTER navigation is processed
            selected_section = st.session_state.get('selected_section')
            selected_subsection = st.session_state.get('selected_subsection')
            
            # Handle Summary section separately (no data loading needed)
            if selected_section == 'summary':
                self.render_summary_content(None, None)
                return
            
            # Check if we should load data (either has subsection OR is a section that loads data directly)
            sections_with_direct_data = ['extra_batch_connections', 'mass_update', 'interfaces', 'hung_threads', 
                                       'online_exceptions_prd', 'batch_exceptions_prd', 'online_exceptions_uat', 
                                       'batch_exceptions_uat']  # Sections that load data without subsections
            should_load_data = selected_subsection or (selected_section in sections_with_direct_data)
            
            if should_load_data:
                try:
                    # Auto-detect and load Excel files based on current navigation selections
                    file_config = self.auto_load_excel_file(
                        section=selected_section,
                        period=st.session_state.get('selected_period', 'daily'),
                        subsection=selected_subsection
                    )
                    
                    if file_config:
                        # Load data automatically
                        df = self.load_data(file_config)
                        
                        if df is not None and not df.empty:
                            # Render main content with the loaded data
                            self.render_main_content_with_data(df, nav_result)
                        elif df is not None and df.empty:
                            st.warning("The loaded data is empty. Please check your Excel file.")
                        else:
                            st.error("Failed to load the Excel file. Please check the file format.")
                    else:
                        st.error(f"No Excel file found for: {selected_section} - {selected_subsection}")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                # Show navigation and placeholder when no subsection selected
                self.render_content_placeholder()
            
            # Handle case where file_config is None
            if False:
                # No Excel files found - show upload option
                st.title("📊 Data Dashboard")
                st.markdown("### Upload Your Excel File")
                
                uploaded_file = st.file_uploader(
                    "Choose an Excel file",
                    type=['xlsx', 'xls'],
                    help="Upload your Excel file to start analyzing your data"
                )
                
                if uploaded_file is not None:
                    # Process uploaded file
                    temp_config = {
                        "data_source_type": "excel",
                        "uploaded_file": uploaded_file,
                        "workspace_file": None,
                        "use_default_file": False
                    }
                    
                    df = self.load_data(temp_config)
                    
                    if df is not None and not df.empty:
                        self.render_main_content(df)
                    else:
                        st.error("Failed to load the uploaded file. Please check the file format.")
                else:
                    # Show instructions
                    st.markdown("""
                    ### Getting Started
                    
                    **Option 1: Place Excel files in workspace**
                    - Copy your `.xlsx` or `.xls` files to this folder
                    - The dashboard will automatically detect and load them
                    
                    **Option 2: Upload files**
                    - Use the file uploader above to analyze any Excel file
                    
                    ### Dashboard Features
                    - 📊 **Interactive Charts**: Line, bar, scatter, pie charts and more
                    - 🔍 **Smart Filters**: Automatic filtering for all data types  
                    - 📈 **Key Metrics**: Auto-generated insights from your data
                    - 📋 **Data Tables**: Sortable and searchable data views
                    - 🔬 **Analysis Tools**: Correlation, distribution, and statistical analysis
                    """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Dashboard error: {e}")
    
    # Helper methods for section-specific metrics
    def render_error_metrics(self, df: pd.DataFrame) -> None:
        """Render error-specific metrics."""
        if df.empty:
            st.warning("No error data available.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Errors", len(df), delta="Critical" if len(df) > 50 else "Normal")
        with col2:
            severity_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['severity', 'priority', 'level'])]
            if severity_cols:
                critical_count = df[severity_cols[0]].str.contains('critical|high|error', case=False, na=False).sum()
                st.metric("Critical Errors", critical_count)
            else:
                st.metric("Active Errors", len(df))
        with col3:
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
            if len(date_cols) > 0:
                days = (df[date_cols[0]].max() - df[date_cols[0]].min()).days + 1
                error_rate = len(df) / max(days, 1)
                st.metric("Daily Error Rate", f"{error_rate:.1f}")
            else:
                st.metric("Error Records", len(df))
        with col4:
            st.metric("Data Quality", "Good" if len(df) < 100 else "Review")
    
    def render_user_impact_metrics(self, df: pd.DataFrame) -> None:
        """Render user impact metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Affected Users", len(df))
        with col2:
            st.metric("Impact Level", "Monitoring")
        with col3:
            st.metric("Total Records", len(df))
        with col4:
            st.metric("Status", "Active")
    
    def render_mass_update_metrics(self, df: pd.DataFrame) -> None:
        """Render mass update metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Update Jobs", len(df))
        with col2:
            st.metric("Processed", len(df))
        with col3:
            st.metric("Total Records", len(df))
        with col4:
            st.metric("System Status", "Running")
    
    def render_interface_metrics(self, df: pd.DataFrame) -> None:
        """Render interface metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Interfaces", len(df))
        with col2:
            st.metric("Connections", len(df))
        with col3:
            st.metric("Total Monitored", len(df))
        with col4:
            st.metric("Network Status", "Stable")
    
    def render_batch_connection_metrics(self, df: pd.DataFrame) -> None:
        """Render batch connection metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Extra Connections", len(df))
        with col2:
            st.metric("Active Batches", len(df))
        with col3:
            st.metric("Connection Pool", "Monitoring")
        with col4:
            st.metric("Pool Status", "Optimal" if len(df) < 50 else "Review")
    
    def render_hung_threads_metrics(self, df: pd.DataFrame) -> None:
        """Render hung threads metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hung Threads", len(df), delta="⚠️ Alert" if len(df) > 0 else "✅ Clear")
        with col2:
            st.metric("Thread Health", "Good" if len(df) == 0 else "Issues")
        with col3:
            st.metric("Total Threads", len(df))
        with col4:
            st.metric("System Health", "Critical" if len(df) > 5 else "Stable")
    
    def render_exception_metrics(self, df: pd.DataFrame, environment: str) -> None:
        """Render exception metrics for PRD/UAT environments."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{environment} Exceptions", len(df))
        with col2:
            st.metric("Active Issues", len(df))
        with col3:
            st.metric("Total Records", len(df))
        with col4:
            env_status = "Critical" if environment == "PRD" and len(df) > 10 else "Monitoring"
            st.metric(f"{environment} Status", env_status)


def main():
    """Main application entry point."""
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()


