"""Main Streamlit dashboard application."""

import streamlit as st
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
                        # For columns with "%" symbol, convert decimals to percentages
                        # (0.04 → 4%, 0.95 → 95%)
                        df_formatted[col] = numeric_series.apply(
                            lambda x: f"{x * 100:.0f}%" if pd.notna(x) else ""
                        )
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
                
                # Try to load from the specific sheet
                try:
                    df = loader.load_data(sheet_name=sheet_name)
                    # Apply date filtering to show only recent weeks
                    df = filter_data_to_recent_weeks(df)
                    logger.info(f"Loaded upload status data from sheet: {sheet_name}")
                    return df
                except Exception as sheet_error:
                    logger.warning(f"Could not load from sheet '{sheet_name}': {sheet_error}")
                    # Fallback to first sheet
                    df = loader.load_data()
                    # Apply date filtering to show only recent weeks
                    df = filter_data_to_recent_weeks(df)
                    logger.info("Loaded upload status data from default sheet")
                    return df
                    
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
        3️⃣ Expand folders to see data
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
        
        /* Main navigation buttons with professional styling */
        .stButton > button {
            text-align: left !important;
            padding: 10px 16px !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            margin: 3px 0 !important;
            border: 1px solid #e9ecef !important;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Enhanced hover effects */
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--light-blue) 0%, #cce7ff 100%) !important;
            border-color: var(--accent-blue) !important;
            color: var(--primary-blue) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Professional sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
            border-right: 3px solid var(--primary-blue) !important;
        }
        
        /* Sub-menu containers with professional styling */
        .sub-menu-container .stButton > button {
            font-size: 12px !important;
            color: var(--neutral-gray) !important;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            border: 1px solid #dee2e6 !important;
            margin: 2px 0 !important;
            padding: 8px 12px !important;
            min-height: 32px !important;
            line-height: 1.4 !important;
            margin-left: 12px !important;
            font-weight: 450 !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.2s ease !important;
        }
        
        /* Enhanced sub-menu hover effects */
        .sub-menu-container .stButton > button:hover {
            background: linear-gradient(135deg, var(--light-blue) 0%, #d1ecf1 100%) !important;
            border-color: var(--accent-blue) !important;
            color: var(--primary-blue) !important;
            transform: translateX(2px) !important;
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
        
        # Add Home button to return to welcome page
        if st.sidebar.button("🏠 Bridges M&O Status Home", key="home_button", help="Return to welcome page"):
            # Clear all session state to return to welcome page
            for key in list(st.session_state.keys()):
                if key.startswith(('selected_', 'expanded_')):
                    del st.session_state[key]
            st.rerun()
        
        # Define all main sections with their details - Reordered per user request
        main_sections = [
            {"key": "error_counts", "icon": "�", "name": "100 Error Counts", "has_subsections": True},
            {"key": "user_impact", "icon": "�", "name": "User Impact", "has_subsections": True},
            {"key": "benefit_issuance", "icon": "�", "name": "Benefit Issuance", "has_subsections": True},
            {"key": "correspondence_tango", "icon": "�", "name": "Correspondence", "has_subsections": True},
            {"key": "mass_update", "icon": "🔄", "name": "Mass Update", "has_subsections": False},
            {"key": "interfaces", "icon": "🔗", "name": "Interfaces", "has_subsections": False},
            {"key": "extra_batch_connections", "icon": "⚡", "name": "Extra Batch Connections", "has_subsections": False},
            {"key": "hung_threads", "icon": "🧵", "name": "Hung Threads", "has_subsections": False},
            {"key": "online_exceptions_prd", "icon": "🌐", "name": "Online Exceptions - PRD", "has_subsections": False},
            {"key": "batch_exceptions_prd", "icon": "�", "name": "Batch Exceptions - PRD", "has_subsections": False},
            {"key": "online_exceptions_uat", "icon": "🧪", "name": "Online Exceptions - UAT", "has_subsections": False},
            {"key": "batch_exceptions_uat", "icon": "🔬", "name": "Batch Exceptions - UAT", "has_subsections": False}
        ]
        
        # Render each main section with expandable functionality
        for section in main_sections:
            section_key = section["key"]
            section_icon = section["icon"]
            section_name = section["name"]
            has_subsections = section["has_subsections"]
            
            # Create expand/collapse button if section has subsections
            if has_subsections:
                is_expanded = section_key in st.session_state.expanded_sections
                expand_symbol = "🔽" if is_expanded else "▶️"
                
                # Create tree-like structure with proper alignment
                button_text = f"{expand_symbol} {section_icon} {section_name}"
                if st.sidebar.button(button_text, 
                                   key=f"expand_{section_key}", 
                                   help="Expand/Collapse section",
                                   use_container_width=True):
                    if is_expanded:
                        st.session_state.expanded_sections.discard(section_key)
                    else:
                        st.session_state.expanded_sections.add(section_key)
                        # Clear previous selections when switching to a different section
                        prev_section = st.session_state.get('selected_section', None)
                        if prev_section != section_key:
                            # Clear all section-related state when switching sections
                            for key in list(st.session_state.keys()):
                                if key.startswith(('selected_', 'current_', 'clicked_', 'data_', 'df_')):
                                    del st.session_state[key]
                            st.session_state.selected_section = section_key
                    st.rerun()
                
                # Show subsections if expanded
                if section_key in st.session_state.expanded_sections:
                    # Add indentation for subsections
                    st.sidebar.markdown('<div style="margin-left: 20px;">', unsafe_allow_html=True)
                    
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
                                    st.sidebar.markdown('<div style="margin-left: 20px;" class="file-menu-container">', unsafe_allow_html=True)
                                    
                                    for file_name in available_files:
                                        icon = file_icons.get(file_name, "📄")
                                        
                                        # Check if this is the currently selected dashboard
                                        is_active = (st.session_state.get('selected_section') == section_key and 
                                                   st.session_state.get('selected_subsection') == file_name and
                                                   st.session_state.get('selected_period') == period_key)
                                        
                                        # Add active styling if selected
                                        if is_active:
                                            st.sidebar.markdown('<div data-active="true" style="background-color: #e3f2fd; border-radius: 4px; margin: 1px 0;">', unsafe_allow_html=True)
                                        
                                        # Clear, clickable button for files
                                        if st.sidebar.button(f"    📄 {icon} {file_name}", 
                                                           key=f"file_{period_key}_{file_name}",
                                                           help=f"Click to analyze {file_name}",
                                                           use_container_width=True):
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
                            
                            # Clear, clickable buttons for files
                            for file_name in available_files:
                                icon = file_icons.get(file_name, "📄")
                                
                                if st.sidebar.button(f"  📄 {icon} {file_name}", 
                                                   key=f"file_correspondence_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True):
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
                            
                            # Clear, clickable buttons for files
                            for file_name in available_files:
                                icon = file_icons.get(file_name, "📄")
                                
                                # Clear, clickable button for files
                                if st.sidebar.button(f"  📄 {icon} {file_name}", 
                                                   key=f"file_error_counts_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True):
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
                            
                            # Clear, clickable buttons for files
                            for file_name in available_files:
                                icon = file_icons.get(file_name, "📄")
                                
                                if st.sidebar.button(f"  📄 {icon} {file_name}", 
                                                   key=f"file_user_impact_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True):
                                    st.session_state.selected_section = section_key
                                    st.session_state.selected_subsection = file_name
                                    st.rerun()

                        else:
                            st.sidebar.markdown('<div style="color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
            else:
                # Section without subsections - just a clickable button
                # Check if this is the currently selected section
                is_active = st.session_state.get('selected_section') == section_key
                
                # Section without subsections - just a clickable button
                if st.sidebar.button(f"{section_icon} {section_name}", key=f"section_{section_key}", 
                                   help=f"Select {section_name}", use_container_width=True):
                    st.session_state.selected_section = section_key
                    st.session_state.selected_subsection = None
                    st.rerun()
            
            # Add small visual separator  
            st.sidebar.markdown("")
        
        # If Benefit Issuance is selected, show its sub-files indented
        if current_section_key == "benefit_issuance":
            available_files = self.get_bi_monitoring_files(current_period_key)
            file_icons = {
                "FAP Daily Issuance": "💳",
                "FIP Daily Issuance": "🏦", 
                "SDA Daily Client Payments": "�"
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
                "interfaces": "� Interfaces",
                "extra_batch_connections": "⚡ Extra Batch Connections",
                "hung_threads": "🧵 Hung Threads",
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
        section_name = selected_section.replace("_", " ").title()
        period_name = selected_period.title() if selected_section == "benefit_issuance" else "N/A"
        
        return {
            "section": selected_section,
            "period": selected_period,
            "subsection": selected_subsection,
            "section_display": section_name,
            "period_display": period_name
        }

    def render_content_placeholder(self) -> None:
        """Render improved placeholder content when no subsection is selected."""
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
            
            🚨 **100 Error Counts** - Session timeouts & errors
            
            👥 **User Impact** - Daily user impact status
            
            ⚡ **Extra Batch Connections** - Connection monitoring
            """)
            
        with col2:
            st.info("""
            **📊 Business Intelligence**
            
            📈 **Benefit Issuance** - FAP, FIP, SDA tracking
            
            📧 **Correspondence** - Tango monitoring & uploads
            
            🔄 **Mass Update** - System update tracking
            """)
        
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
        
        title = section_titles.get(selected_section, {}).get(selected_period, f"{selected_section.replace('_', ' ').title()} Dashboard")
        st.title(title)
        
        # Section-specific content routing
        section_handlers = {
            "benefit_issuance": self.render_benefit_issuance_content,
            "correspondence_tango": self.render_correspondence_tango_content,
            "error_counts": self.render_error_counts_content,
            "user_impact": self.render_user_impact_content,
            "mass_update": self.render_mass_update_content,
            "interfaces": self.render_interfaces_content,
            "extra_batch_connections": self.render_extra_batch_connections_content,
            "hung_threads": self.render_hung_threads_content,
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


    def render_benefit_issuance_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Benefit Issuance specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("� Click on a dashboard item in the sidebar to view its contents")
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
        for row_idx, row in df.iterrows():
            # Add row styling
            if row_idx % 2 == 0:
                st.markdown('<div class="tango-table-row">', unsafe_allow_html=True)
            
            data_cols = st.columns(len(df.columns))
            
            for col_idx, col in enumerate(df.columns):
                cell_value = row[col] if pd.notna(row[col]) else ''
                
                with data_cols[col_idx]:
                    # Make date cells clickable if upload status is available
                    if col == date_col and cell_value and has_upload_status:
                        formatted_date = format_date_to_standard(cell_value)
                        # Create clickable date hyperlink-style button
                        if st.button(
                            formatted_date, 
                            key=f"date_link_{row_idx}_{col_idx}",
                            help=f"Click to view upload status for {formatted_date}",
                            type="secondary"
                        ):
                            st.session_state.clicked_tango_date = formatted_date
                            st.rerun()
                    else:
                        # Regular cell content
                        st.write(str(cell_value))
            
            if row_idx % 2 == 0:
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add spacing between rows
            st.markdown('<div style="margin: 5px 0;"></div>', unsafe_allow_html=True)
    
    def show_upload_status_for_date(self, selected_date: str, unused_param=None) -> None:
        """Display upload status for the selected date."""
        st.markdown("---")
        st.subheader(f"📄 File Upload Status for {selected_date}")
        
        # Show available sheets for debugging
        available_sheets = self._get_available_upload_status_sheets()
        expected_sheet = self._convert_date_to_sheet_name(selected_date)
        
        with st.expander("🔍 Debug Info: Sheet Loading", expanded=False):
            st.write(f"**Selected Date:** {selected_date}")
            st.write(f"**Expected Sheet Name:** {expected_sheet}")
            st.write(f"**Available Sheets:** {available_sheets}")
            if expected_sheet in available_sheets:
                st.success(f"✅ Sheet '{expected_sheet}' found!")
            else:
                st.warning(f"⚠️ Sheet '{expected_sheet}' not found. Will use closest match or first sheet.")
        
        # Load upload status data for the specific date
        date_specific_df = self.load_tango_upload_status(selected_date)
        
        if date_specific_df.empty:
            st.warning(f"Upload status data is not available for {selected_date}.")
            st.info("💡 Try selecting a different date or check if the corresponding sheet exists in the Excel file.")
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
            st.info("� Click on a dashboard item in the sidebar to view its contents")
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
        tab1, tab2, tab3 = st.tabs(["� Charts", "📊 Statistics", "🔬 Analysis"])
        
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
            st.info("� Click on a dashboard item in the sidebar to view its contents")
            return
        
        with st.expander("🚨 **Error Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus
        st.header("� Data Table")
        file_display_name = f"{selected_subsection} Data"
        self.table_component.data_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"� Key Metrics")
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
            st.info("� Click on a dashboard item in the sidebar to view its contents")
            return
        
        with st.expander("👥 **Impact Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus
        st.header("� Data Table")
        file_display_name = f"{selected_subsection} Data"
        self.table_component.data_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"� Key Metrics")
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
                st.session_state.selected_section = 'benefit_issuance'
            if 'selected_period' not in st.session_state:
                st.session_state.selected_period = 'daily'
            
            # Render navigation FIRST to handle section changes
            nav_result = self.render_navigation_menu()
            
            # Get selections from session state AFTER navigation is processed
            selected_section = st.session_state.get('selected_section')
            selected_subsection = st.session_state.get('selected_subsection')
            
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
                    - � **Data Tables**: Sortable and searchable data views
                    - � **Analysis Tools**: Correlation, distribution, and statistical analysis
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
