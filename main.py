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


def format_date_to_month_year(date_input):
    """
    Convert any date format to Mon-YYYY format for Month & Year columns.
    
    Args:
        date_input: Date in various formats (string, datetime, pandas timestamp, etc.)
        
    Returns:
        String in Mon-YYYY format or original value if conversion fails
    """
    if pd.isna(date_input) or date_input is None or str(date_input).strip() == '':
        return date_input
    
    try:
        # Convert to string first
        date_str = str(date_input).strip()
        
        # If already in Mon-YYYY format, return as is
        if re.match(r'[A-Z][a-z]{2}-\d{4}', date_str):
            return date_str
        
        # Handle pandas timestamps
        if hasattr(date_input, 'strftime'):
            return date_input.strftime('%b-%Y')
        
        # Try to parse various date formats
        date_formats = [
            '%Y-%m-%d',           # 2025-11-25
            '%m/%d/%Y',           # 11/25/2025
            '%d/%m/%Y',           # 25/11/2025
            '%Y/%m/%d',           # 2025/11/25
            '%d-%m-%Y',           # 25-11-2025
            '%m-%d-%Y',           # 11-25-2025
            '%Y%m%d',             # 20251125
            '%d.%m.%Y',           # 25.11.2025
            '%m.%d.%Y',           # 11.25.2025
            '%d %B %Y',           # 25 November 2025
            '%B %d, %Y',          # November 25, 2025
            '%d %b %Y',           # 25 Nov 2025
            '%b %d, %Y',          # Nov 25, 2025
        ]
        
        # Try each format
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%b-%Y')
            except ValueError:
                continue
        
        # Try pandas to_datetime as fallback
        try:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime('%b-%Y')
        except:
            pass
        
        # If all parsing fails, return original value
        return date_input
        
    except Exception:
        return date_input

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
            '%Y-%m-%d %H:%M:%S',  # 2024-10-08 14:30:00 (datetime format)
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
        non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'runtime']
        if any(keyword in col_lower for keyword in non_date_keywords):
            continue
            
        # Check if column name suggests it's a date
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp', 'week of']) or \
           ('month' in col_lower and 'year' in col_lower):
            # Exclude columns that are clearly not dates (like processing times)
            if 'processing' in col_lower or 'load' in col_lower or 'staging' in col_lower:
                continue
            # Handle special case of "Week Of" columns with date ranges
            if 'week of' in col_lower:
                try:
                    # Extract the end date from date ranges like "29-SEP To 03-OCT"
                    def extract_end_date(date_range_str):
                        if pd.isna(date_range_str):
                            return pd.NaT
                        str_val = str(date_range_str).strip()
                        # Look for patterns like "DD-MMM To DD-MMM" or "DD-MMM-YYYY To DD-MMM-YYYY"
                        if ' to ' in str_val.lower():
                            parts = str_val.split(' To ')  # Keep original case
                            if len(parts) == 2:
                                end_date_str = parts[1].strip()
                                # Add current year if not present
                                if len(end_date_str.split('-')) == 2:  # Format: DD-MMM
                                    from datetime import datetime
                                    current_year = datetime.now().year
                                    end_date_str = f"{end_date_str}-{current_year}"
                                # Try to parse the end date
                                try:
                                    return pd.to_datetime(end_date_str, format='%d-%b-%Y', errors='coerce')
                                except:
                                    # Fallback to general parsing
                                    try:
                                        return pd.to_datetime(end_date_str, errors='coerce')
                                    except:
                                        return pd.NaT
                        return pd.NaT
                    
                    # Create a sortable date column based on end dates
                    df_sorted[col + '_sortable'] = df_sorted[col].apply(extract_end_date)
                    date_columns.append(col + '_sortable')
                except:
                    pass
            else:
                # Regular date column processing
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
    
    # Remove temporary sortable columns that were created for sorting
    columns_to_remove = [col for col in df_sorted.columns if col.endswith('_sortable')]
    if columns_to_remove:
        df_sorted = df_sorted.drop(columns=columns_to_remove)
    
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
        non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'session', 'connection', 'batch', 'thread', 'runtime']
        if any(keyword in col_lower for keyword in non_date_keywords):
            continue
            
        # Check if column name suggests it's a date
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
            # Exclude columns that are clearly not dates (like processing times)
            if 'processing' in col_lower or 'load' in col_lower or 'staging' in col_lower:
                continue
            date_columns.append(col)
        # Handle Month & Year columns separately with custom formatting
        elif 'month' in col_lower and 'year' in col_lower:
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
                        re.match(r'\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}', val_str) or  # YYYY-MM-DD HH:MM:SS
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
        # Special formatting for Month & Year columns to show Mon-YYYY format
        if 'month' in col.lower() and 'year' in col.lower():
            df_formatted[col] = df_formatted[col].apply(format_date_to_month_year)
        else:
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
                    # Skip variance columns that contain text values like "Above 10%" or "Below 10%"
                    if 'variance' in col_lower:
                        # Check if variance column contains text values
                        first_non_null = sample_values.iloc[0] if len(sample_values) > 0 else None
                        if first_non_null is not None and isinstance(first_non_null, str):
                            if any(text in str(first_non_null).lower() for text in ['above', 'below', '%']):
                                continue  # Skip formatting this variance column
                    
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


def remove_empty_rows(df):
    """
    Remove completely empty rows and rows where all meaningful columns are empty/NaN.
    This is an enhanced version that aggressively removes empty rows from Excel data.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with empty rows removed
    """
    if df is None or len(df) == 0:
        return df
    
    df_cleaned = df.copy()
    
    # Method 1: Remove rows where all values are NaN
    df_cleaned = df_cleaned.dropna(how='all')
    
    if len(df_cleaned) == 0:
        return df_cleaned
    
    # Method 2: Remove rows where all values are effectively empty
    # This includes NaN, empty strings, whitespace-only strings, and "None" strings
    def is_empty_value(val):
        """Check if a value is considered empty"""
        if pd.isna(val):
            return True
        if val is None:
            return True
        
        # Convert to string and check
        str_val = str(val).strip().lower()
        
        # Empty string or whitespace
        if str_val == '':
            return True
        
        # Common "empty" representations
        empty_representations = ['none', 'null', 'nan', 'n/a', 'na', '#n/a', '#na', 'undefined']
        if str_val in empty_representations:
            return True
        
        # Just whitespace characters
        if str_val.isspace():
            return True
            
        return False
    
    # Create a mask for rows where all values are empty
    empty_mask = df_cleaned.apply(lambda row: all(is_empty_value(val) for val in row), axis=1)
    df_cleaned = df_cleaned[~empty_mask]
    
    if len(df_cleaned) == 0:
        return df_cleaned
    
    # Method 3: More sophisticated approach - remove rows where all meaningful columns are empty
    # Identify columns that typically contain data vs. those that might be legitimately empty
    meaningful_columns = []
    for col in df_cleaned.columns:
        col_lower = col.lower().strip()
        
        # Skip columns that are commonly empty or metadata
        skip_keywords = ['unnamed', 'index', 'row', 'id', 'key', 'guid', 'uuid']
        if any(keyword in col_lower for keyword in skip_keywords):
            continue
            
        # Include columns that typically have data
        meaningful_columns.append(col)
    
    if meaningful_columns:
        # Remove rows where all meaningful columns are empty
        meaningful_empty_mask = df_cleaned[meaningful_columns].apply(
            lambda row: all(is_empty_value(val) for val in row), axis=1
        )
        df_cleaned = df_cleaned[~meaningful_empty_mask]
    
    if len(df_cleaned) == 0:
        return df_cleaned
    
    # Method 4: Remove trailing empty rows more aggressively
    # Work backwards from the end to find the last row with ANY meaningful data
    last_meaningful_row = -1
    
    for idx in range(len(df_cleaned) - 1, -1, -1):
        row = df_cleaned.iloc[idx]
        has_meaningful_data = False
        
        for val in row:
            if not is_empty_value(val):
                # Additional check for meaningful content (not just formatting)
                str_val = str(val).strip()
                if len(str_val) > 0 and str_val not in ['0', '0.0', '0.00']:
                    has_meaningful_data = True
                    break
        
        if has_meaningful_data:
            last_meaningful_row = idx
            break
    
    # Keep only rows up to and including the last meaningful row
    if last_meaningful_row >= 0 and last_meaningful_row < len(df_cleaned) - 1:
        df_cleaned = df_cleaned.iloc[:last_meaningful_row + 1]
    elif last_meaningful_row == -1:
        # No meaningful data found, return empty dataframe
        return df_cleaned.iloc[0:0]  # Empty dataframe with same structure
    
    # Method 5: Final cleanup - remove any rows that are all zeros or default values
    # This catches cases where Excel has rows with default values
    if len(df_cleaned) > 1:  # Only do this if we have multiple rows
        zero_like_mask = df_cleaned.apply(
            lambda row: all(
                is_empty_value(val) or str(val).strip() in ['0', '0.0', '0.00', 'false', 'False'] 
                for val in row
            ), axis=1
        )
        
        # Only remove zero-like rows if they're at the end
        # Find consecutive zero-like rows at the end
        zero_indices = df_cleaned.index[zero_like_mask].tolist()
        if zero_indices:
            # Check if zero-like rows are at the end
            last_index = df_cleaned.index[-1]
            consecutive_zeros_at_end = []
            
            for idx in reversed(zero_indices):
                if idx == last_index or (consecutive_zeros_at_end and idx == consecutive_zeros_at_end[0] - 1):
                    consecutive_zeros_at_end.insert(0, idx)
                    last_index = idx
                else:
                    break
            
            # Remove consecutive zero-like rows at the end
            if consecutive_zeros_at_end:
                df_cleaned = df_cleaned.drop(consecutive_zeros_at_end)
    
    return df_cleaned


def display_clean_dataframe(df, **kwargs):
    """
    Wrapper function to display dataframes with ULTRA-AGGRESSIVE empty row removal.
    This ensures absolutely no empty rows are displayed.
    
    Args:
        df: DataFrame to display
        **kwargs: Arguments to pass to st.dataframe or st.table
    """
    if df is None or len(df) == 0:
        st.info("No data available for display.")
        return
    
    # DEBUG: Show original dataframe info
    original_rows = len(df)
    
    # ULTRA-AGGRESSIVE CLEANING - Multiple passes to catch everything
    clean_df = df.copy()
    
    # Pass 1: Remove completely NaN rows
    clean_df = clean_df.dropna(how='all')
    
    if len(clean_df) == 0:
        st.info("No meaningful data available for display after cleaning.")
        return
    
    # Pass 2: Remove rows where ALL values are effectively empty
    def is_truly_empty(val):
        """Most aggressive empty value detection"""
        if pd.isna(val) or val is None:
            return True
        
        # Convert to string and check
        str_val = str(val).strip().lower()
        
        # Check for various empty representations
        empty_values = {
            '', 'none', 'null', 'nan', 'n/a', 'na', '#n/a', '#na', 
            'undefined', '0', '0.0', '0.00', '0.000', 'false', 'f',
            ' ', '\t', '\n', '\r', '  ', '   '
        }
        
        return str_val in empty_values or str_val.isspace() or len(str_val) == 0
    
    # Remove rows where ALL columns are truly empty
    truly_empty_mask = clean_df.apply(
        lambda row: not any(not is_truly_empty(val) for val in row), axis=1
    )
    clean_df = clean_df[~truly_empty_mask]
    
    # Pass 3: Remove rows that are all zeros or similar default values
    if len(clean_df) > 1:  # Only if we have multiple rows
        # Check for rows that are all zeros/defaults
        zero_like_mask = clean_df.apply(
            lambda row: all(
                is_truly_empty(val) or str(val).strip() in ['0', '0.0', '0.00', 'false', 'False', '0.000'] 
                for val in row
            ), axis=1
        )
        
        # Only remove these if they're at the end (trailing zeros)
        if zero_like_mask.any():
            last_meaningful_idx = -1
            for idx in range(len(clean_df) - 1, -1, -1):
                if not zero_like_mask.iloc[idx]:
                    last_meaningful_idx = idx
                    break
            
            if last_meaningful_idx >= 0:
                clean_df = clean_df.iloc[:last_meaningful_idx + 1]
    
    # Pass 4: Final check - remove any rows that somehow still look empty
    if len(clean_df) > 0:
        final_mask = clean_df.apply(
            lambda row: any(
                pd.notna(val) and 
                str(val).strip() != '' and 
                str(val).strip().lower() not in ['none', 'null', 'nan', 'n/a', '0', '0.0', '0.00']
                for val in row
            ), axis=1
        )
        clean_df = clean_df[final_mask]
    
    # Pass 5: Reset index to remove any gaps
    if len(clean_df) > 0:
        clean_df = clean_df.reset_index(drop=True)
    
    # DEBUG: Show cleaning results
    cleaned_rows = len(clean_df)
    if original_rows != cleaned_rows:
        st.caption(f"🧹 Cleaned data: {original_rows} → {cleaned_rows} rows (removed {original_rows - cleaned_rows} empty rows)")
    
    # Final display - Use st.table() for small datasets to eliminate empty rows
    if len(clean_df) > 0:
        # For small datasets (<=10 rows), use st.table() which doesn't show empty rows
        if len(clean_df) <= 10:
            # Create a styled version without index for display
            styled_df = clean_df.reset_index(drop=True)
            
            # Use st.dataframe with hide_index=True for full control and full width
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            # For larger datasets, use st.dataframe with the provided kwargs
            st.dataframe(clean_df, **kwargs)
    else:
        st.info("No meaningful data available for display after aggressive cleaning.")


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
            if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']) or \
               ('month' in col_lower and 'year' in col_lower):
                # Exclude columns that are clearly not dates (like processing times)
                if 'processing' in col_lower or 'load' in col_lower or 'staging' in col_lower:
                    continue
                date_columns.append(col)
        
        # Use the first date column found
        date_column = date_columns[0] if date_columns else None
        
    if date_column is None or date_column not in df_filtered.columns:
        # No date column found, return original data
        return df_filtered
    
    try:
        # Convert date column to datetime
        df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce')
        
        # Calculate date range for filtering
        today = datetime.now()
        
        # Check if this is monthly data by looking at the date column name
        date_col_lower = date_column.lower()
        is_monthly_data = ('month' in date_col_lower and 'year' in date_col_lower)
        
        if is_monthly_data:
            # For monthly data, limit to 4 most recent records instead of date filtering
            # Sort by date descending and take top 4 records
            df_filtered = df_filtered.sort_values(by=date_column, ascending=False)
            df_filtered = df_filtered.head(4)
            return df_filtered
        else:
            # For weekly/daily data, use week-based filtering
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
        
        return df_filtered
        
    except Exception as e:
        st.warning(f"Could not filter dates in column '{date_column}': {str(e)}")
        return df_filtered


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_daily_exceptions_summary(section: str) -> pd.DataFrame:
    """
    Get summary data for Daily Exceptions dashboards showing all available dates and row counts.
    
    Args:
        section: The section key (e.g., 'prd_online_exceptions')
        
    Returns:
        DataFrame with Date and count columns
    """
    from pathlib import Path
    import pandas as pd
    
    # Map section keys to Excel sheet names
    section_to_sheet_map = {
        "prd_online_exceptions": "PRD Online",
        "prd_batch_exceptions": "PRD Batch", 
        "prd_batch_runtime": "PRD Batch Runtime",
        "uat_online_exceptions": "UAT Online",
        "uat_batch_exceptions": "UAT Batch",
        "uat_batch_runtime": "UAT Batch Runtime"
    }
    
    # Determine column names based on section type
    runtime_sections = ["prd_batch_runtime", "uat_batch_runtime"]
    if section in runtime_sections:
        count_column = "Batch jobs that exceeded Average Runtime"
    else:
        count_column = "Number of New Exceptions"
    
    monitoring_data_path = Path("Monitoring Data Files")
    daily_exceptions_path = monitoring_data_path / "Daily Exceptions"
    
    summary_data = []
    
    if daily_exceptions_path.exists():
        sheet_name = section_to_sheet_map.get(section)
        
        if sheet_name:
            # Find all Excel files in the Daily Exceptions folder
            excel_files = []
            for pattern in ['*.xlsx', '*.xls']:
                excel_files.extend(daily_exceptions_path.glob(pattern))
            
            # Filter out temporary files
            excel_files = [f for f in excel_files if not f.name.startswith('~$')]
            
            # Sort by modification time (newest first)
            excel_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            for excel_file in excel_files:
                try:
                    # Extract date from filename (remove extension)
                    file_date = excel_file.stem
                    
                    # Read the specific sheet and count rows
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    row_count = len(df)
                    
                    summary_data.append({
                        "Date": file_date,
                        count_column: row_count
                    })
                    
                except Exception as e:
                    # If there's an error reading the file/sheet, skip it
                    continue
    
    # Create DataFrame from summary data
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort by date (newest first) - try to parse dates if possible
        try:
            summary_df['Date_parsed'] = pd.to_datetime(summary_df['Date'], errors='coerce')
            summary_df = summary_df.sort_values('Date_parsed', ascending=False, na_last=True)
            summary_df = summary_df.drop('Date_parsed', axis=1)
        except:
            # If date parsing fails, sort alphabetically (descending)
            summary_df = summary_df.sort_values('Date', ascending=False)
        
        # Reset index to ensure clean DataFrame
        summary_df = summary_df.reset_index(drop=True)
        
        return summary_df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["Date", count_column])


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
            initial_sidebar_state="collapsed"
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
                "yearly": "Yearly"
            }
            
            period_folder = period_folder_map.get(period, "Daily")
            period_path = bi_monitoring_path / period_folder
            
            if period_path.exists():
                # If a specific subsection is selected, look for that file
                if subsection:
                    # Map display names to actual file names for benefit issuance
                    display_to_file_map = {
                        # Daily files
                        "FAP Payments": "FAP Daily Issuance",
                        "FIP Payments (EBT & Warrants)": "FIP Daily Issuance",
                        "SDA Client Payments (EBT & Warrants)": "SDA Daily Client Payments",
                        # Weekly files
                        "CDC Warrants": "CDC Warrants",
                        "SER Warrants": "SER Warrants", 
                        "SDA Provider Payments": "SDA Provider Payments",
                        "RAP/RCA Client Payments": "RAP-RCA Client Payments",
                        "SSP Client Warrants": "SSP Client Warrants",
                        "Vendoring Payments": "Vendoring Payments",
                        # Monthly files
                        "FAP Payroll": "FAP Payroll",
                        "Cash Payroll": "Cash Payroll"
                    }
                    
                    # Use the mapped file name or the original subsection name
                    actual_file_name = display_to_file_map.get(subsection, subsection)
                    
                    # Try both .xlsx and .xls extensions
                    for ext in ['.xlsx', '.xls']:
                        file_path = period_path / f"{actual_file_name}{ext}"
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
        
        elif section == "online_ora_errors":
            ora_errors_path = monitoring_data_path / "ORA Errors"
            
            if ora_errors_path.exists():
                # Look for Online.xlsx file
                for ext in ['.xlsx', '.xls']:
                    file_path = ora_errors_path / f"Online{ext}"
                    if file_path.exists() and not file_path.name.startswith('~$'):
                        return {
                            "data_source_type": "excel",
                            "workspace_file": file_path,
                            "uploaded_file": None,
                            "use_default_file": False,
                            "section": section,
                            "subsection": "Online"
                        }
        
        elif section == "batch_ora_errors":
            ora_errors_path = monitoring_data_path / "ORA Errors"
            
            if ora_errors_path.exists():
                # Look for Batch.xlsx file
                for ext in ['.xlsx', '.xls']:
                    file_path = ora_errors_path / f"Batch{ext}"
                    if file_path.exists() and not file_path.name.startswith('~$'):
                        return {
                            "data_source_type": "excel",
                            "workspace_file": file_path,
                            "uploaded_file": None,
                            "use_default_file": False,
                            "section": section,
                            "subsection": "Batch"
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
        
        elif section == "mass_update":
            # Handle Mass Update section - look in a Mass Update folder if it exists
            mass_update_path = monitoring_data_path / "Mass Update"
            if mass_update_path.exists():
                # Find Excel files in Mass Update folder
                excel_files = []
                for pattern in ['*.xlsx', '*.xls']:
                    excel_files.extend(mass_update_path.glob(pattern))
                
                # Filter out temporary files
                excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                
                if excel_files:
                    # Use the first/most recent Excel file
                    selected_file = excel_files[0]
                    return {
                        "data_source_type": "excel",
                        "workspace_file": selected_file,
                        "uploaded_file": None,
                        "use_default_file": False,
                        "section": section,
                        "subsection": "Mass Update Data"
                    }
            # Return None if no folder or files found - this will trigger our graceful handling
            return None
        
        elif section == "interfaces":
            # Handle Interfaces section - look in an Interfaces folder if it exists
            interfaces_path = monitoring_data_path / "Interfaces"
            if interfaces_path.exists():
                # Find Excel files in Interfaces folder
                excel_files = []
                for pattern in ['*.xlsx', '*.xls']:
                    excel_files.extend(interfaces_path.glob(pattern))
                
                # Filter out temporary files
                excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                
                if excel_files:
                    # Use the first/most recent Excel file
                    selected_file = excel_files[0]
                    return {
                        "data_source_type": "excel",
                        "workspace_file": selected_file,
                        "uploaded_file": None,
                        "use_default_file": False,
                        "section": section,
                        "subsection": "Interfaces Data"
                    }
            # Return None if no folder or files found - this will trigger our graceful handling
            return None
        
        elif section == "hung_threads":
            # Handle Hung Threads section - look in a Hung Threads folder if it exists
            hung_threads_path = monitoring_data_path / "Hung Threads"
            if hung_threads_path.exists():
                # Find Excel files in Hung Threads folder
                excel_files = []
                for pattern in ['*.xlsx', '*.xls']:
                    excel_files.extend(hung_threads_path.glob(pattern))
                
                # Filter out temporary files
                excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                
                if excel_files:
                    # Use the first/most recent Excel file
                    selected_file = excel_files[0]
                    return {
                        "data_source_type": "excel",
                        "workspace_file": selected_file,
                        "uploaded_file": None,
                        "use_default_file": False,
                        "section": section,
                        "subsection": "Hung Threads Data"
                    }
            # Return None if no folder or files found - this will trigger our graceful handling
            return None

        # Handle Daily Exceptions sections
        elif section in ["prd_online_exceptions", "prd_batch_exceptions", "prd_batch_runtime",
                        "uat_online_exceptions", "uat_batch_exceptions", "uat_batch_runtime"]:
            daily_exceptions_path = monitoring_data_path / "Daily Exceptions"
            
            if daily_exceptions_path.exists():
                # Map section keys to Excel sheet names
                section_to_sheet_map = {
                    "prd_online_exceptions": "PRD Online",
                    "prd_batch_exceptions": "PRD Batch", 
                    "prd_batch_runtime": "PRD Batch Runtime",
                    "uat_online_exceptions": "UAT Online",
                    "uat_batch_exceptions": "UAT Batch",
                    "uat_batch_runtime": "UAT Batch Runtime"
                }
                
                # Find the most recent Excel file in the Daily Exceptions folder
                excel_files = []
                for pattern in ['*.xlsx', '*.xls']:
                    excel_files.extend(daily_exceptions_path.glob(pattern))
                
                # Filter out temporary files
                excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                
                if excel_files:
                    # Sort by modification time to get the most recent file
                    most_recent_file = max(excel_files, key=lambda f: f.stat().st_mtime)
                    sheet_name = section_to_sheet_map.get(section)
                    
                    if sheet_name:
                        return {
                            "data_source_type": "excel",
                            "workspace_file": most_recent_file,
                            "uploaded_file": None,
                            "use_default_file": False,
                            "section": section,
                            "sheet_name": sheet_name,
                            "subsection": sheet_name
                        }
        
        # Handle Batch Status sections
        elif section in ["uat_batch_status", "prd_batch_status"]:
            batch_status_path = monitoring_data_path / "Batch Status"
            
            if batch_status_path.exists():
                # Map section keys to Excel file names
                section_to_file_map = {
                    "uat_batch_status": "UAT.xlsx",
                    "prd_batch_status": "Production.xlsx"
                }
                
                file_name = section_to_file_map.get(section)
                if file_name:
                    file_path = batch_status_path / file_name
                    
                    if file_path.exists():
                        return {
                            "data_source_type": "excel",
                            "workspace_file": file_path,
                            "uploaded_file": None,
                            "use_default_file": False,
                            "section": section,
                            "subsection": section
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
                    
                    # Check if a specific sheet is required (e.g., for Daily Exceptions)
                    if sidebar_config.get("sheet_name"):
                        # Use the specified sheet directly
                        specified_sheet = sidebar_config["sheet_name"]
                        df = loader.load_data(sheet_name=specified_sheet)
                    else:
                        # Get available sheets and show selection
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
            
            # Map actual file names to display names for benefit issuance
            file_to_display_map = {
                # Daily files
                "FAP Daily Issuance": "FAP Payments",
                "FIP Daily Issuance": "FIP Payments (EBT & Warrants)",
                "SDA Daily Client Payments": "SDA Client Payments (EBT & Warrants)",
                # Weekly files
                "CDC Warrants": "CDC Warrants",
                "SER Warrants": "SER Warrants", 
                "SDA Provider Payments": "SDA Provider Payments",
                "RAP-RCA Client Payments": "RAP/RCA Client Payments",
                "SSP Client Warrants": "SSP Client Warrants",
                "Vendoring Payments": "Vendoring Payments",
                # Monthly files
                "FAP Payroll": "FAP Payroll",
                "Cash Payroll": "Cash Payroll"
            }
            
            # Convert file names to display names
            display_names = []
            for file_name in file_names:
                display_name = file_to_display_map.get(file_name, file_name)
                display_names.append(display_name)
            
            return sorted(display_names)
        
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

    def get_online_ora_errors_files(self) -> List[str]:
        """Get available files in ORA Errors folder for Online errors."""
        workspace_path = Path(__file__).parent
        ora_errors_path = workspace_path / "Monitoring Data Files" / "ORA Errors"
        
        if ora_errors_path.exists():
            # Look specifically for Online files
            for pattern in ['Online.xlsx', 'Online.xls']:
                file_path = ora_errors_path / pattern
                if file_path.exists() and not file_path.name.startswith('~$'):
                    return ["Online"]
        
        return []

    def get_batch_ora_errors_files(self) -> List[str]:
        """Get available files in ORA Errors folder for Batch errors."""
        workspace_path = Path(__file__).parent
        ora_errors_path = workspace_path / "Monitoring Data Files" / "ORA Errors"
        
        if ora_errors_path.exists():
            # Look specifically for Batch files
            for pattern in ['Batch.xlsx', 'Batch.xls']:
                file_path = ora_errors_path / pattern
                if file_path.exists() and not file_path.name.startswith('~$'):
                    return ["Batch"]
        
        return []

    def render_breadcrumb_navigation(self, selected_section: str = None, selected_subsection: str = None) -> None:
        """Render breadcrumb navigation for better user orientation."""
        if not selected_section or selected_section == 'summary':
            # On home page - no breadcrumb needed
            return
            
        # Create breadcrumb trail
        breadcrumb_items = ["🏠 Home"]
        
        # Add section to breadcrumb
        section_info = self.get_section_display_info(selected_section)
        if section_info:
            breadcrumb_items.append(f"{section_info['icon']} {section_info['name']}")
        
        # Add subsection if exists
        if selected_subsection:
            subsection_info = self.get_subsection_display_info(selected_section, selected_subsection)
            if subsection_info:
                breadcrumb_items.append(f"{subsection_info.get('icon', '📄')} {subsection_info['name']}")
        
        # Create clickable breadcrumb navigation
        breadcrumb_html = '<div style="margin-bottom: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #17a2b8;">'
        
        for i, item in enumerate(breadcrumb_items):
            if i == 0:  # Home link
                breadcrumb_html += f'<a href="#" onclick="return false;" style="color: #17a2b8; text-decoration: none; font-weight: bold;">{item}</a>'
            else:
                breadcrumb_html += f' <span style="color: #6c757d; margin: 0 8px;">→</span> '
                if i == len(breadcrumb_items) - 1:  # Current page (not clickable)
                    breadcrumb_html += f'<span style="color: #333; font-weight: bold;">{item}</span>'
                else:  # Intermediate links
                    breadcrumb_html += f'<span style="color: #6c757d;">{item}</span>'
        
        breadcrumb_html += '</div>'
        st.markdown(breadcrumb_html, unsafe_allow_html=True)
        
        # Add navigation buttons
        if selected_section or selected_subsection:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                if st.button("← Back", key="breadcrumb_back", help="Go back to previous level"):
                    # Define sections that auto-select subsections (single subsection sections)
                    single_subsection_sections = ['error_counts', 'user_impact', 'online_ora_errors', 'batch_ora_errors']
                    # Define sections that are batch status subsections
                    batch_subsections = ['uat_batch_status', 'prd_batch_status']
                    
                    if selected_section in batch_subsections:
                        # For batch status subsections, go back to batch status selection
                        st.session_state.selected_section = 'batch_status'
                        st.session_state.selected_subsection = None
                    elif selected_subsection and selected_section in single_subsection_sections:
                        # For single subsection sections, go directly back to home
                        st.session_state.selected_section = 'summary'
                        st.session_state.selected_subsection = None
                    elif selected_subsection:
                        # Go back to section selection page
                        st.session_state.selected_subsection = None
                    elif selected_section:
                        # Go back to home (summary)
                        st.session_state.selected_section = 'summary'
                        st.session_state.selected_subsection = None
                    st.rerun()
            
            with col2:
                if st.button("🏠 Home", key="breadcrumb_home", help="Return to home page"):
                    # Return to summary home page
                    st.session_state.selected_section = 'summary'
                    st.session_state.selected_subsection = None
                    st.session_state.benefit_category = None  # Reset benefit category
                    st.rerun()

    def get_section_display_info(self, section_key: str) -> dict:
        """Get display information for a section."""
        section_map = {
            "summary": {"icon": "📋", "name": "System Summary"},
            "user_impact": {"icon": "👥", "name": "User Impact"},
            "error_counts": {"icon": "🚨", "name": "100 Error Counts"},
            "online_ora_errors": {"icon": "💻", "name": "Online ORA Errors"},
            "batch_ora_errors": {"icon": "📊", "name": "Batch ORA Errors"},
            "correspondence_tango": {"icon": "📧", "name": "Correspondence"},
            "benefit_issuance": {"icon": "📈", "name": "Benefit Issuance"},
            "batch_status": {"icon": "⚙️", "name": "Batch Status"},
            "daily_exceptions": {"icon": "⚠️", "name": "Daily Exceptions"},
            "miscellaneous_bridges": {"icon": "🔗", "name": "Other Critical Processes"}
        }
        return section_map.get(section_key, {"icon": "📄", "name": section_key.replace('_', ' ').title()})

    def get_subsection_display_info(self, section_key: str, subsection_key: str) -> dict:
        """Get display information for a subsection."""
        subsection_maps = {
            "batch_status": {
                "uat_batch_status": {"icon": "🧪", "name": "UAT Environment"},
                "prd_batch_status": {"icon": "🏭", "name": "Production Environment"}
            },
            "daily_exceptions": {
                "prd_online_exceptions": {"icon": "🌐", "name": "PRD Online Exceptions"},
                "prd_batch_exceptions": {"icon": "⚙️", "name": "PRD Batch Exceptions"},
                "prd_batch_runtime": {"icon": "⏱️", "name": "PRD Batch Runtime"},
                "uat_online_exceptions": {"icon": "🧪", "name": "UAT Online Exceptions"},
                "uat_batch_exceptions": {"icon": "🔧", "name": "UAT Batch Exceptions"},
                "uat_batch_runtime": {"icon": "⌚", "name": "UAT Batch Runtime"}
            }
        }
        
        section_subsections = subsection_maps.get(section_key, {})
        return section_subsections.get(subsection_key, {"icon": "📄", "name": subsection_key.replace('_', ' ').title()})

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
        1️⃣ Set date on Home page<br>
        2️⃣ Click any dashboard section<br>
        3️⃣ View data for selected date
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current date filter status
        selected_date = self.get_selected_date()
        st.sidebar.markdown("### � Current Date Filter")
        st.sidebar.info(f"� **Viewing:** {selected_date.strftime('%B %d, %Y')}\n\n🏠 Go to Home page to change date")
        
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
        
        /* Basic button styling - no forced colors, just layout */
        .stButton > button {
            text-align: left !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Grey hover effect for all buttons */
        .stButton > button:hover {
            background-color: #6c757d !important;
            color: white !important;
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
        
        # Auto-expand parent section if a subsection is selected
        if current_section_key in ["uat_batch_status", "prd_batch_status"]:
            st.session_state.expanded_sections.add("batch_status")
        elif current_section_key in ["prd_online_exceptions", "prd_batch_exceptions", "prd_batch_runtime", "uat_online_exceptions", "uat_batch_exceptions", "uat_batch_runtime"]:
            st.session_state.expanded_sections.add("daily_exceptions")
        
        selected_section = current_section_key
        selected_subsection = current_subsection_key
        
        # Add quick navigation section at top
        st.sidebar.markdown("### ⚡ Quick Access")
        
        # Quick navigation buttons for high-priority sections
        quick_nav_col1, quick_nav_col2 = st.sidebar.columns(2)
        
        with quick_nav_col1:
            if st.button("🏠 Home", key="quick_home", use_container_width=True, help="Return to welcome page"):
                # Clear all session state to return to welcome
                for key in list(st.session_state.keys()):
                    if key.startswith(('selected_', 'expanded_')):
                        del st.session_state[key]
                st.rerun()
                
        with quick_nav_col2:
            if st.button("📋 Summary", key="quick_summary", use_container_width=True, help="Go to System Summary"):
                st.session_state.selected_section = "summary"
                st.session_state.selected_subsection = None
                if 'expanded_sections' not in st.session_state:
                    st.session_state.expanded_sections = set()
                st.rerun()
        
        # Add current location indicator
        current_section = st.session_state.get('selected_section')
        current_subsection = st.session_state.get('selected_subsection')
        
        if current_section:
            section_info = self.get_section_display_info(current_section)
            location_text = f"📍 **Current:** {section_info['icon']} {section_info['name']}"
            if current_subsection:
                subsection_info = self.get_subsection_display_info(current_section, current_subsection)
                location_text += f" → {subsection_info['icon']} {subsection_info['name']}"
            st.sidebar.info(location_text)
        else:
            st.sidebar.success("📍 **Current:** 🏠 Welcome Page")
        
        st.sidebar.markdown("---")
        
        # Add keyboard shortcuts info
        with st.sidebar.expander("⌨️ Keyboard Shortcuts", expanded=False):
            st.markdown("""
            **Navigation:**
            - `Home` - Return to welcome page
            - `S` - Go to System Summary
            - `Esc` - Go back one level
            
            **Quick Access:**
            - `B` - Batch Status
            - `U` - User Impact
            - `E` - Error Counts
            - `C` - Correspondence
            """)
        
        st.sidebar.markdown("---")
        
        st.sidebar.subheader("🗂️ All Sections")
        
        # Define sections organized by priority groups
        high_priority_sections = [
            {"key": "summary", "icon": "📋", "name": "System Summary", "has_subsections": False, "color": "#17a2b8", "active_color": "#138496", "priority": "overview"},
            {"key": "batch_status", "icon": "⚙️", "name": "Batch Status", "has_subsections": True, "color": "#795548", "active_color": "#5d4037", "priority": "critical"},
            {"key": "correspondence_tango", "icon": "�", "name": "Correspondence", "has_subsections": True, "color": "#2196f3", "active_color": "#1976d2", "priority": "critical"},
            {"key": "user_impact", "icon": "�", "name": "User Impact", "has_subsections": True, "color": "#4caf50", "active_color": "#388e3c", "priority": "critical"},
            {"key": "error_counts", "icon": "�", "name": "100 Error Counts", "has_subsections": True, "color": "#ffffff", "active_color": "#f0f0f0", "priority": "critical"}
        ]
        
        business_intelligence_sections = [
            {"key": "benefit_issuance", "icon": "📈", "name": "Benefit Issuance", "has_subsections": True, "color": "#ff9800", "active_color": "#f57c00", "priority": "business"},
            {"key": "daily_exceptions", "icon": "⚠️", "name": "Daily Exceptions", "has_subsections": True, "color": "#9c27b0", "active_color": "#7b1fa2", "priority": "business"},
            {"key": "miscellaneous_bridges", "icon": "🔗", "name": "Other Critical Processes", "has_subsections": True, "color": "#008b8b", "active_color": "#006064", "priority": "business"}
        ]
        
        # Combine all sections for rendering
        all_sections = high_priority_sections + business_intelligence_sections
        
        # Render high priority sections first
        st.sidebar.markdown("##### 🔴 High Priority Monitoring")
        for i, section in enumerate(high_priority_sections):
            self.render_section_navigation(section, i)
        
        st.sidebar.markdown("---")
        
        # Render business intelligence sections
        st.sidebar.markdown("##### 📊 Business Intelligence & Processing")  
        for i, section in enumerate(business_intelligence_sections, start=len(high_priority_sections)):
            self.render_section_navigation(section, i)
            
        # Return navigation result
        return {
            "selected_section": st.session_state.get('selected_section'),
            "selected_period": st.session_state.get('selected_period', 'daily'),
            "selected_subsection": st.session_state.get('selected_subsection')
        }

    def render_section_navigation(self, section: dict, index: int) -> None:
        """Render navigation for a single section."""
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
                               key=f"expand_{section_key}_{index}", 
                               help="Click to expand/collapse subsections",
                               use_container_width=True, 
                               type="secondary"):
                # Toggle functionality for expand/collapse
                if is_expanded:
                    st.session_state.expanded_sections.discard(section_key)
                else:
                    st.session_state.expanded_sections.add(section_key)
                
                # Set this as the selected section
                st.session_state.selected_section = section_key
                st.session_state.selected_subsection = None
                st.session_state.benefit_category = None  # Reset benefit category navigation
                st.rerun()
            
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            # Show subsections if expanded
            if is_expanded:
                self.render_subsections(section_key, section_color)
        else:
            # Simple button for sections without subsections
            button_text = f"**{section_icon} {section_name}**"
            is_active = section_key == st.session_state.get('selected_section')
            
            if is_active:
                bg_color = f"{active_color}CC"
                border_color = active_color
            else:
                bg_color = f"{section_color}99"
                border_color = section_color
                
            st.sidebar.markdown(f"""
            <div style="margin: 0px 0px 0px 0px !important; padding: 1px 2px 1px 2px !important; background-color: {bg_color}; border-radius: 4px; border-left: 3px solid {border_color}; margin-bottom: 0px !important; margin-top: 0px !important;">
            """, unsafe_allow_html=True)
            
            if st.sidebar.button(button_text,
                               key=f"select_{section_key}_{index}",
                               help=f"Navigate to {section_name}",
                               use_container_width=True,
                               type="secondary"):
                st.session_state.selected_section = section_key
                st.session_state.selected_subsection = None
                st.session_state.benefit_category = None  # Reset benefit category navigation
                if 'expanded_sections' not in st.session_state:
                    st.session_state.expanded_sections = set()
                st.rerun()
                
            st.sidebar.markdown('</div>', unsafe_allow_html=True)

    def render_subsection_selection_page(self, section_key: str) -> None:
        """Render a page for selecting subsections within a section."""
        section_info = self.get_section_display_info(section_key)
        
        st.markdown(f"# {section_info['icon']} {section_info['name']}")
        st.markdown("Please select a specific environment or subsection to view detailed monitoring data.")
        st.markdown("---")
        
        if section_key == "batch_status":
            st.markdown("### 🎯 Select Environment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # UAT Environment Card
                st.markdown("""
                <div style="border: 2px solid #FF7043; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FFF3E0;">
                    <h4>🧪 UAT Environment</h4>
                    <p>Monitor UAT batch job status and failures for testing environment</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open UAT Batch Status", key="select_uat_batch", use_container_width=True):
                    st.session_state.selected_section = "uat_batch_status"
                    st.session_state.selected_subsection = None
                    st.rerun()
            
            with col2:
                # Production Environment Card  
                st.markdown("""
                <div style="border: 2px solid #1976D2; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E3F2FD;">
                    <h4>🏭 Production Environment</h4>
                    <p>Monitor Production batch job status and critical failures</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Production Batch Status", key="select_prd_batch", use_container_width=True):
                    st.session_state.selected_section = "prd_batch_status"
                    st.session_state.selected_subsection = None
                    st.rerun()
                    
        elif section_key == "daily_exceptions":
            st.markdown("### 🎯 Select Exception Type")
            
            # PRD Exceptions
            st.markdown("#### 🏭 Production Environment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🌐 PRD Online Exceptions", key="select_prd_online", use_container_width=True):
                    st.session_state.selected_subsection = "prd_online_exceptions"
                    st.rerun()
            with col2:
                if st.button("⚙️ PRD Batch Exceptions", key="select_prd_batch_exc", use_container_width=True):
                    st.session_state.selected_subsection = "prd_batch_exceptions"
                    st.rerun()
            with col3:
                if st.button("⏱️ PRD Batch Runtime", key="select_prd_runtime", use_container_width=True):
                    st.session_state.selected_subsection = "prd_batch_runtime"
                    st.rerun()
            
            st.markdown("#### 🧪 UAT Environment")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                if st.button("🧪 UAT Online Exceptions", key="select_uat_online", use_container_width=True):
                    st.session_state.selected_subsection = "uat_online_exceptions" 
                    st.rerun()
            with col5:
                if st.button("🔧 UAT Batch Exceptions", key="select_uat_batch_exc", use_container_width=True):
                    st.session_state.selected_subsection = "uat_batch_exceptions"
                    st.rerun()
            with col6:
                if st.button("⌚ UAT Batch Runtime", key="select_uat_runtime", use_container_width=True):
                    st.session_state.selected_subsection = "uat_batch_runtime"
                    st.rerun()
                    
        elif section_key == "benefit_issuance":
            # Check if we're at category level or subsection level
            benefit_category = st.session_state.get('benefit_category', None)
            
            if benefit_category is None:
                # First level: Show only Daily, Weekly, Monthly categories
                st.markdown("### 🎯 Select Benefit Issuance Category")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Daily Benefit Issuance
                    st.markdown("""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #F1F8E9; height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h4 style="margin: 0 0 15px 0; color: #2E7D32;">📅 Daily Issuance</h4>
                            <p style="margin: 0; color: #4A4A4A; line-height: 1.4;">Daily FAP, FIP, and SDA benefit processing dashboards and monitoring</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View Daily Dashboards", key="select_daily_category", use_container_width=True):
                        st.session_state.benefit_category = "daily"
                        st.rerun()
                
                with col2:
                    # Weekly Benefit Issuance
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FFF8E1; height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h4 style="margin: 0 0 15px 0; color: #E65100;">📊 Weekly Issuance</h4>
                            <p style="margin: 0; color: #4A4A4A; line-height: 1.4;">Weekly warrants, provider payments, and special program dashboards</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View Weekly Dashboards", key="select_weekly_category", use_container_width=True):
                        st.session_state.benefit_category = "weekly"
                        st.rerun()
                
                with col3:
                    # Monthly Benefit Issuance
                    st.markdown("""
                    <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E3F2FD; height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h4 style="margin: 0 0 15px 0; color: #1565C0;">📈 Monthly Issuance</h4>
                            <p style="margin: 0; color: #4A4A4A; line-height: 1.4;">Monthly payroll processing and summary report dashboards</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View Monthly Dashboards", key="select_monthly_category", use_container_width=True):
                        st.session_state.benefit_category = "monthly"
                        st.rerun()
                        
            elif benefit_category == "daily":
                # Second level: Show Daily subsections
                st.markdown("### 📅 Daily Benefit Issuance Dashboards")
                
                # Back button
                if st.button("⬅️ Back to Categories", key="back_to_categories_daily"):
                    st.session_state.benefit_category = None
                    st.rerun()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # FAP Payments
                    st.markdown("""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #F1F8E9;">
                        <h5>🍯 FAP Payments</h5>
                        <p>Daily FAP benefit issuance data</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View FAP Payments", key="select_fap_payments", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "daily"
                        st.session_state.selected_subsection = "FAP Payments"
                        st.rerun()
                
                with col2:
                    # FIP Payments
                    st.markdown("""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #F1F8E9;">
                        <h5>👨‍👩‍👧‍👦 FIP Payments</h5>
                        <p>Daily FIP EBT & Warrants data</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View FIP Payments", key="select_fip_payments", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "daily"
                        st.session_state.selected_subsection = "FIP Payments (EBT & Warrants)"
                        st.rerun()
                
                with col3:
                    # SDA Client Payments
                    st.markdown("""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #F1F8E9;">
                        <h5>🏠 SDA Client Payments</h5>
                        <p>Daily SDA EBT & Warrants data</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View SDA Client", key="select_sda_client", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "daily"
                        st.session_state.selected_subsection = "SDA Client Payments (EBT & Warrants)"
                        st.rerun()
                        
            elif benefit_category == "weekly":
                # Second level: Show Weekly subsections
                st.markdown("### 📊 Weekly Benefit Issuance Dashboards")
                
                # Back button
                if st.button("⬅️ Back to Categories", key="back_to_categories_weekly"):
                    st.session_state.benefit_category = None
                    st.rerun()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CDC Warrants
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>🏥 CDC Warrants</h5>
                        <p>Weekly CDC warrant processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View CDC Warrants", key="select_cdc_warrants", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "CDC Warrants"
                        st.rerun()
                    
                    # SER Warrants
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>🎓 SER Warrants</h5>
                        <p>Weekly SER warrant processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View SER Warrants", key="select_ser_warrants", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "SER Warrants"
                        st.rerun()
                
                with col2:
                    # SDA Provider Payments
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>🏠 SDA Provider</h5>
                        <p>Weekly SDA provider payments</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View SDA Provider", key="select_sda_provider", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "SDA Provider Payments"
                        st.rerun()
                    
                    # RAP/RCA Client Payments
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>🔄 RAP/RCA Client</h5>
                        <p>Weekly RAP/RCA client payments</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View RAP/RCA", key="select_rap_rca", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "RAP/RCA Client Payments"
                        st.rerun()
                
                with col3:
                    # SSP Client Warrants
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>💰 SSP Client</h5>
                        <p>Weekly SSP client warrants</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View SSP Client", key="select_ssp_client", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "SSP Client Warrants"
                        st.rerun()
                    
                    # Vendoring Payments
                    st.markdown("""
                    <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #FFF8E1;">
                        <h5>🏪 Vendoring Payments</h5>
                        <p>Weekly vendoring payment processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View Vendoring", key="select_vendoring", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "weekly"
                        st.session_state.selected_subsection = "Vendoring Payments"
                        st.rerun()
                        
            elif benefit_category == "monthly":
                # Second level: Show Monthly subsections
                st.markdown("### 📈 Monthly Benefit Issuance Dashboards")
                
                # Back button
                if st.button("⬅️ Back to Categories", key="back_to_categories_monthly"):
                    st.session_state.benefit_category = None
                    st.rerun()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # FAP Payroll
                    st.markdown("""
                    <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #E3F2FD;">
                        <h5>🍯 FAP Payroll</h5>
                        <p>Monthly FAP payroll processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View FAP Payroll", key="select_fap_payroll", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "monthly"
                        st.session_state.selected_subsection = "FAP Payroll"
                        st.rerun()
                
                with col2:
                    # Cash Payroll
                    st.markdown("""
                    <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #E3F2FD;">
                        <h5>💵 Cash Payroll</h5>
                        <p>Monthly cash payroll processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 View Cash Payroll", key="select_cash_payroll", use_container_width=True):
                        st.session_state.selected_section = "benefit_issuance"
                        st.session_state.selected_period = "monthly"
                        st.session_state.selected_subsection = "Cash Payroll"
                        st.rerun()
                    
        elif section_key == "miscellaneous_bridges":
            st.markdown("### 🎯 Select Critical Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mass Update
                st.markdown("""
                <div style="border: 2px solid #673AB7; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #F3E5F5;">
                    <h4>🔄 Mass Update</h4>
                    <p>Monitor mass update processes and batch operations</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Mass Update Dashboard", key="select_mass_update", use_container_width=True):
                    st.session_state.selected_subsection = "mass_update"
                    st.rerun()
                
                # Interfaces
                st.markdown("""
                <div style="border: 2px solid #3F51B5; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E8EAF6;">
                    <h4>🔗 Interfaces</h4>
                    <p>Monitor system interfaces and data exchange processes</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Interfaces Dashboard", key="select_interfaces", use_container_width=True):
                    st.session_state.selected_subsection = "interfaces"
                    st.rerun()
            
            with col2:
                # Hung Threads
                st.markdown("""
                <div style="border: 2px solid #FF5722; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FBE9E7;">
                    <h4>🧵 Hung Threads</h4>
                    <p>Monitor system performance and hung thread detection</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Hung Threads Dashboard", key="select_hung_threads", use_container_width=True):
                    st.session_state.selected_subsection = "hung_threads"
                    st.rerun()
                    
        elif section_key == "miscellaneous_bridges":
            st.markdown("### 🎯 Select Critical Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mass Update
                st.markdown("""
                <div style="border: 2px solid #673AB7; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #F3E5F5;">
                    <h4>🔄 Mass Update</h4>
                    <p>Monitor mass update processes and batch operations</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Mass Update Dashboard", key="select_mass_update", use_container_width=True):
                    st.session_state.selected_subsection = "mass_update"
                    st.rerun()
                
                # Interfaces
                st.markdown("""
                <div style="border: 2px solid #3F51B5; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E8EAF6;">
                    <h4>🔗 Interfaces</h4>
                    <p>Monitor system interfaces and data exchange processes</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Interfaces Dashboard", key="select_interfaces", use_container_width=True):
                    st.session_state.selected_subsection = "interfaces"
                    st.rerun()
            
            with col2:
                # Hung Threads
                st.markdown("""
                <div style="border: 2px solid #FF5722; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FBE9E7;">
                    <h4>⚠️ Hung Threads</h4>
                    <p>Monitor and track hung thread issues and resolution</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Hung Threads Dashboard", key="select_hung_threads", use_container_width=True):
                    st.session_state.selected_subsection = "hung_threads"
                    st.rerun()
                
                # Extra Batch Connections
                st.markdown("""
                <div style="border: 2px solid #607D8B; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #ECEFF1;">
                    <h4>🔌 Extra Batch Connections</h4>
                    <p>Monitor extra batch connections and connection management</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Extra Batch Connections Dashboard", key="select_extra_batch", use_container_width=True):
                    st.session_state.selected_subsection = "extra_batch_connections"
                    st.rerun()
                    
        elif section_key == "correspondence_tango":
            st.markdown("### 🎯 Select Correspondence Type")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Tango Monitoring
                st.markdown("""
                <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E3F2FD;">
                    <h4>📨 Tango Monitoring</h4>
                    <p>Monitor Tango system status and performance metrics</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open Tango Monitoring Dashboard", key="select_tango_monitoring", use_container_width=True):
                    st.session_state.selected_subsection = "Tango Monitoring"
                    st.rerun()
            
            with col2:
                # View History Screen Validation
                st.markdown("""
                <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E8F5E8;">
                    <h4>📋 View History Screen Validation</h4>
                    <p>Track view history screen validation processes and results</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Click Here to Open View History Dashboard", key="select_view_history", use_container_width=True):
                    st.session_state.selected_subsection = "View History Screen Validation"
                    st.rerun()
                    
        elif section_key == "error_counts":
            st.markdown("### 🎯 Select Error Count Type")
            
            # Daily 100 Error Counts
            st.markdown("""
            <div style="border: 2px solid #F44336; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FFEBEE;">
                <h4>🚨 Daily 100 Error Counts</h4>
                <p>Monitor daily session timeouts and critical system errors</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Click Here to Open Daily Error Counts Dashboard", key="select_daily_errors", use_container_width=True):
                st.session_state.selected_section = "error_counts"
                st.session_state.selected_subsection = "Daily 100 Error Counts"
                st.rerun()
                
        elif section_key == "user_impact":
            st.markdown("### 🎯 Select User Impact Type")
            
            # Daily User Impact Status
            st.markdown("""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #E8F5E8;">
                <h4>👥 Daily User Impact Status</h4>
                <p>Monitor daily user experience metrics and error percentages</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Click Here to Open Daily User Impact Dashboard", key="select_daily_user_impact", use_container_width=True):
                st.session_state.selected_section = "user_impact"
                st.session_state.selected_subsection = "Daily User Impact Status"
                st.rerun()
                
        elif section_key == "online_ora_errors":
            st.markdown("### 🎯 Online ORA Errors")
            
            # Online ORA Errors
            st.markdown("""
            <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #FFF3E0;">
                <h4>💻 Online ORA Errors</h4>
                <p>Monitor online Oracle database errors and exceptions</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Click Here to Open Online ORA Errors Dashboard", key="select_online_ora_errors", use_container_width=True):
                st.session_state.selected_section = "online_ora_errors"
                st.session_state.selected_subsection = "Online"
                st.rerun()
                
        elif section_key == "batch_ora_errors":
            st.markdown("### 🎯 Batch ORA Errors")
            
            # Batch ORA Errors
            st.markdown("""
            <div style="border: 2px solid #9C27B0; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #F3E5F5;">
                <h4>📊 Batch ORA Errors</h4>
                <p>Track batch processing Oracle database errors</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Click Here to Open Batch ORA Errors Dashboard", key="select_batch_ora_errors", use_container_width=True):
                st.session_state.selected_section = "batch_ora_errors"
                st.session_state.selected_subsection = "Batch"
                st.rerun()

    def render_subsections(self, section_key: str, section_color: str) -> None:
        """Render subsections for a given section."""
        if section_key == "batch_status":
            # Batch Status subsections
            subsections = [
                {"key": "uat_batch_status", "icon": "🧪", "name": "UAT Environment"},
                {"key": "prd_batch_status", "icon": "🏭", "name": "Production Environment"}
            ]
            
            st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
            for subsection in subsections:
                is_selected = st.session_state.get('selected_subsection') == subsection["key"]
                button_style = "primary" if is_selected else "secondary"
                
                if st.sidebar.button(f"  └─ {subsection['icon']} {subsection['name']}", 
                                   key=f"subsection_{subsection['key']}", 
                                   help=f"Navigate to {subsection['name']}",
                                   use_container_width=True,
                                   type=button_style):
                    st.session_state.selected_section = section_key
                    st.session_state.selected_subsection = subsection["key"]
                    st.rerun()
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
        elif section_key == "daily_exceptions":
            # Daily Exceptions subsections  
            subsections = [
                {"key": "prd_online_exceptions", "icon": "🌐", "name": "PRD Online Exceptions"},
                {"key": "prd_batch_exceptions", "icon": "⚙️", "name": "PRD Batch Exceptions"},
                {"key": "prd_batch_runtime", "icon": "⏱️", "name": "PRD Batch Runtime"},
                {"key": "uat_online_exceptions", "icon": "🧪", "name": "UAT Online Exceptions"},
                {"key": "uat_batch_exceptions", "icon": "🔧", "name": "UAT Batch Exceptions"},
                {"key": "uat_batch_runtime", "icon": "⌚", "name": "UAT Batch Runtime"}
            ]
            
            st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
            for subsection in subsections:
                is_selected = st.session_state.get('selected_subsection') == subsection["key"]
                button_style = "primary" if is_selected else "secondary"
                
                if st.sidebar.button(f"  └─ {subsection['icon']} {subsection['name']}", 
                                   key=f"subsection_{subsection['key']}", 
                                   help=f"Navigate to {subsection['name']}",
                                   use_container_width=True,
                                   type=button_style):
                    st.session_state.selected_section = section_key
                    st.session_state.selected_subsection = subsection["key"]
                    st.rerun()
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # For other sections, show a placeholder or file-based navigation
            st.sidebar.markdown('<div style="margin-left: 15px; color: #666;">', unsafe_allow_html=True)
            st.sidebar.write("📄 Select from available files")
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
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
                            # Check if prev_section is a subsection of this section
                            is_subsection = False
                            if section_key == "batch_status" and prev_section in ["uat_batch_status", "prd_batch_status"]:
                                is_subsection = True
                            elif section_key == "daily_exceptions" and prev_section in ["prd_online_exceptions", "prd_batch_exceptions", "prd_batch_runtime", "uat_online_exceptions", "uat_batch_exceptions", "uat_batch_runtime"]:
                                is_subsection = True
                            
                            if not is_subsection:
                                # Clear all section-related state when switching sections
                                for key in list(st.session_state.keys()):
                                    if key.startswith(('selected_', 'current_', 'clicked_', 'data_', 'df_')):
                                        del st.session_state[key]
                        
                        # Collapse all other sections and expand this one
                        st.session_state.expanded_sections.clear()
                        st.session_state.expanded_sections.add(section_key)
                        
                        # Only set selected_section to main section if not currently on a valid subsection
                        current_section = st.session_state.get('selected_section', None)
                        if section_key == "batch_status" and current_section in ["uat_batch_status", "prd_batch_status"]:
                            # Keep the subsection selected
                            pass
                        elif section_key == "daily_exceptions" and current_section in ["prd_online_exceptions", "prd_batch_exceptions", "prd_batch_runtime", "uat_online_exceptions", "uat_batch_exceptions", "uat_batch_runtime"]:
                            # Keep the subsection selected  
                            pass
                        else:
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
                            {"key": "monthly", "icon": "📉", "name": "Monthly"}
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
                                        # Daily files
                                        "FAP Payments": "💳",
                                        "FIP Payments (EBT & Warrants)": "🏦", 
                                        "SDA Client Payments (EBT & Warrants)": "💰",
                                        # Weekly files
                                        "CDC Warrants": "🎫",
                                        "SER Warrants": "📝",
                                        "SDA Provider Payments": "🏥",
                                        "RAP/RCA Client Payments": "👥",
                                        "SSP Client Warrants": "🏛️",
                                        "Vendoring Payments": "🛒"
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
                    
                    elif section_key == "online_ora_errors":
                        # Get files from ORA Errors folder for Online
                        available_files = self.get_online_ora_errors_files()
                        
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        if available_files:
                            file_icons = {
                                "Online": "💻"
                            }
                            
                            # Hierarchical file buttons with tree symbols
                            for i, file_name in enumerate(available_files):
                                icon = file_icons.get(file_name, "📄")
                                tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                
                                if st.sidebar.button(f"　{tree_symbol} {icon} {file_name}", 
                                                   key=f"file_online_ora_errors_{file_name}",
                                                   help=f"Click to analyze {file_name}",
                                                   use_container_width=True,
                                                   type="secondary"):
                                    st.session_state.selected_section = section_key
                                    st.session_state.selected_subsection = file_name
                                    st.rerun()

                        else:
                            st.sidebar.markdown('<div style="color: orange; font-size: 12px;">⚠️ No files available</div>', unsafe_allow_html=True)
                        
                        st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    elif section_key == "batch_ora_errors":
                        # Get files from ORA Errors folder for Batch
                        available_files = self.get_batch_ora_errors_files()
                        
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        if available_files:
                            file_icons = {
                                "Batch": "📊"
                            }
                            
                            # Hierarchical file buttons with tree symbols
                            for i, file_name in enumerate(available_files):
                                icon = file_icons.get(file_name, "📄")
                                tree_symbol = "└─" if i == len(available_files) - 1 else "├─"
                                
                                if st.sidebar.button(f"　{tree_symbol} {icon} {file_name}", 
                                                   key=f"file_batch_ora_errors_{file_name}",
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
                        # Other Critical Processes subsections
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
                        
                        # Define the exception subsections in the specified order
                        exception_sections = [
                            {"key": "prd_online_exceptions", "icon": "🌐", "name": "PRD Online Exceptions"},
                            {"key": "prd_batch_exceptions", "icon": "💻", "name": "PRD Batch Exceptions"},
                            {"key": "prd_batch_runtime", "icon": "⏱️", "name": "PRD Batch Runtime"},
                            {"key": "uat_online_exceptions", "icon": "🧪", "name": "UAT Online Exceptions"},
                            {"key": "uat_batch_exceptions", "icon": "🔬", "name": "UAT Batch Exceptions"},
                            {"key": "uat_batch_runtime", "icon": "⏰", "name": "UAT Batch Runtime"}
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
                    
                    elif section_key == "batch_status":
                        # Batch Status subsections
                        st.sidebar.markdown('<div style="margin-left: 15px;" class="sub-menu-container">', unsafe_allow_html=True)
                        
                        # Define the batch status subsections
                        batch_sections = [
                            {"key": "uat_batch_status", "icon": "🧪", "name": "UAT"},
                            {"key": "prd_batch_status", "icon": "🏭", "name": "Production"}
                        ]
                        
                        # Hierarchical subsection buttons with tree symbols
                        for i, batch_section in enumerate(batch_sections):
                            batch_key = batch_section["key"]
                            batch_icon = batch_section["icon"]
                            batch_name = batch_section["name"]
                            
                            # Use tree symbols for hierarchy (last item gets └─, others get ├─)
                            tree_symbol = "└─" if i == len(batch_sections) - 1 else "├─"
                            
                            # Check if this is the currently selected dashboard
                            is_active = (st.session_state.get('selected_section') == batch_key)
                            
                            # Add active styling if selected
                            if is_active:
                                st.sidebar.markdown('<div data-active="true" style="background-color: #e3f2fd; border-radius: 4px; margin: 1px 0;">', unsafe_allow_html=True)
                            
                            if st.sidebar.button(f"　{tree_symbol} {batch_icon} {batch_name}", 
                                               key=f"batch_{batch_key}",
                                               help=f"Click to analyze {batch_name} Batch Status",
                                               use_container_width=True,
                                               type="secondary"):
                                # Clear all data state when selecting a new subsection
                                for key in list(st.session_state.keys()):
                                    if key.startswith(('data_', 'df_', 'clicked_', 'current_data')):
                                        del st.session_state[key]
                                
                                st.session_state.selected_section = batch_key
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
                "FAP Payments": "💳",
                "FIP Payments (EBT & Warrants)": "🏦", 
                "SDA Client Payments (EBT & Warrants)": "💰"
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
                "online_ora_errors": "💻 Online ORA Errors",
                "batch_ora_errors": "📊 Batch ORA Errors",
                "user_impact": "👥 User Impact",
                "mass_update": "🔄 Mass Update",
                "interfaces": "🔗 Interfaces",
                "extra_batch_connections": "⚡ Extra Batch Connections",
                "hung_threads": "🧵 Hung Threads",
                "data_warehouse": "🏢 Data Warehouse",
                "consolidated_inquiry": "🔍 Consolidated Inquiry",
                "miscellaneous_bridges": "🔗 Other Critical Processes",
                "daily_exceptions": "⚠️ Daily Exceptions",
                "prd_online_exceptions": "🌐 PRD Online Exceptions",
                "prd_batch_exceptions": "� PRD Batch Exceptions",
                "prd_batch_runtime": "⏱️ PRD Batch Runtime",
                "uat_online_exceptions": "🧪 UAT Online Exceptions",
                "uat_batch_exceptions": "🔬 UAT Batch Exceptions",
                "uat_batch_runtime": "⏰ UAT Batch Runtime",
                "batch_status": "⚙️ Batch Status",
                "uat_batch_status": "🧪 UAT Batch Status",
                "prd_batch_status": "🏭 Production Batch Status"
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
        • Data shows only selected date<br>
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
            
            ⚙️ **Batch Status** - UAT & Production batch job monitoring
            
            � **Correspondence** - Tango monitoring & uploads
            
            �👥 **User Impact** - Daily user impact status & tracking
            
            🚨 **100 Error Counts** - Session timeouts & system errors
            """)
            
        with col2:
            st.info("""
            **📊 Business Intelligence & Processing**
            
            📈 **Benefit Issuance** - FAP, FIP, SDA tracking
            
            ⚠️ **Daily Exceptions** - Exception monitoring & resolution
            
            🔧 **Other Critical Processes** - Bridge system monitoring
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
                    "FAP Payments",
                    "FIP Payments (EBT & Warrants)", 
                    "SDA Client Payments (EBT & Warrants)",
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
                "title": "Other Critical Processes",
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
            "prd_online_exceptions": {
                "title": "PRD Online Exceptions",
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
            "prd_batch_exceptions": {
                "title": "PRD Batch Exceptions",
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
            "prd_batch_runtime": {
                "title": "PRD Batch Runtime",
                "icon": "⏱️",
                "color": "#ff9800",
                "description": "Monitor batch process runtime performance in production environment.",
                "features": [
                    "Runtime performance tracking",
                    "Production batch timing",
                    "Performance optimization",
                    "Execution time analysis"
                ]
            },
            "uat_online_exceptions": {
                "title": "UAT Online Exceptions", 
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
            "uat_batch_exceptions": {
                "title": "UAT Batch Exceptions",
                "icon": "🔬",
                "color": "#8bc34a", 
                "description": "Track batch process exceptions in UAT environment.",
                "features": [
                    "UAT batch monitoring",
                    "Testing batch analysis",
                    "Pre-production validation",
                    "Batch testing metrics"
                ]
            },
            "uat_batch_runtime": {
                "title": "UAT Batch Runtime",
                "icon": "⏰",
                "color": "#9c27b0",
                "description": "Monitor batch process runtime performance in UAT environment.",
                "features": [
                    "UAT runtime monitoring",
                    "Testing batch timing",
                    "Performance validation",
                    "Pre-production timing analysis"
                ]
            },
            "batch_status": {
                "title": "Batch Status",
                "icon": "⚙️",
                "color": "#795548",
                "description": "Monitor batch job status and performance across UAT and Production environments.",
                "features": [
                    "UAT Environment monitoring",
                    "Production Environment tracking", 
                    "Previous working day data",
                    "Failed batch job analysis"
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
        st.markdown(f"""
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
                Click <strong>"{info['icon']} {info['title']}"</strong> in the sidebar to begin exploring your data!
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
        
        # Get current selections from session state
        selected_section = st.session_state.get('selected_section')
        selected_subsection = st.session_state.get('selected_subsection')  
        selected_period = st.session_state.get('selected_period', 'daily')
        
        # Show date filter status if a specific date is selected
        from datetime import datetime
        selected_date = self.get_selected_date()
        recent_date_str = self.get_most_recent_weekday_date()
        default_date = datetime.strptime(recent_date_str, '%Y-%m-%d').date() if recent_date_str != "No recent data" else datetime.now().date()
        
        if selected_date != default_date:
            st.info(f"📅 **Viewing data for: {selected_date.strftime('%B %d, %Y')}** | Return home to change date")
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Remove empty rows from the dataframe
        df = remove_empty_rows(df)
        
        # Apply standardized date formatting to all date columns
        df = format_dataframe_dates(df)
        
        # Always filter by the selected date only - no historical data
        original_df = df.copy()
        df = self.filter_data_by_selected_date(df)
        
        # Note: Individual dashboard functions will handle empty data messaging
        # Don't show duplicate message here
        
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
            },
            "online_ora_errors": {
                "daily": "Online ORA Errors Dashboard",
                "weekly": "Online ORA Errors Dashboard",
                "monthly": "Online ORA Errors Dashboard",
                "yearly": "Online ORA Errors Dashboard"
            },
            "batch_ora_errors": {
                "daily": "Batch ORA Errors Dashboard",
                "weekly": "Batch ORA Errors Dashboard",
                "monthly": "Batch ORA Errors Dashboard",
                "yearly": "Batch ORA Errors Dashboard"
            },
            "mass_update": {
                "daily": "🔄 Mass Update Dashboard",
                "weekly": "🔄 Mass Update Dashboard",
                "monthly": "🔄 Mass Update Dashboard",
                "yearly": "🔄 Mass Update Dashboard"
            },
            "interfaces": {
                "daily": "🔗 Interfaces Dashboard",
                "weekly": "🔗 Interfaces Dashboard",
                "monthly": "🔗 Interfaces Dashboard",
                "yearly": "🔗 Interfaces Dashboard"
            },
            "hung_threads": {
                "daily": "🧵 Hung Threads Dashboard",
                "weekly": "🧵 Hung Threads Dashboard",
                "monthly": "🧵 Hung Threads Dashboard",
                "yearly": "🧵 Hung Threads Dashboard"
            },
            "extra_batch_connections": {
                "daily": "🔌 Extra Batch Connections Dashboard",
                "weekly": "🔌 Extra Batch Connections Dashboard",
                "monthly": "🔌 Extra Batch Connections Dashboard",
                "yearly": "🔌 Extra Batch Connections Dashboard"
            },
            "data_warehouse": {
                "daily": "🏢 Data Warehouse Dashboard",
                "weekly": "🏢 Data Warehouse Dashboard",
                "monthly": "🏢 Data Warehouse Dashboard",
                "yearly": "🏢 Data Warehouse Dashboard"
            },
            "consolidated_inquiry": {
                "daily": "🔍 Consolidated Inquiry Dashboard",
                "weekly": "🔍 Consolidated Inquiry Dashboard",
                "monthly": "🔍 Consolidated Inquiry Dashboard",
                "yearly": "🔍 Consolidated Inquiry Dashboard"
            }
        }
        
        # Get title and render colored header for all sections
        title = section_titles.get(selected_section, {}).get(selected_period, f"{selected_section.replace('_', ' ').title()} Dashboard" if selected_section else "Dashboard")
        
        # Section color mapping
        section_colors = {
            "benefit_issuance": "#2196f3",
            "correspondence_tango": "#4caf50", 
            "error_counts": "#f44336",
            "online_ora_errors": "#ff9800",
            "batch_ora_errors": "#9c27b0",
            "user_impact": "#ff9800",
            "mass_update": "#9c27b0",
            "interfaces": "#607d8b", 
            "hung_threads": "#795548",
            "extra_batch_connections": "#ff5722",
            "data_warehouse": "#3f51b5",
            "consolidated_inquiry": "#4caf50",
            "batch_status": "#795548",
            "uat_batch_status": "#FF7043",
            "prd_batch_status": "#1976D2",
            "summary": "#17a2b8"
        }
        
        # Section icon mapping
        section_icons = {
            "benefit_issuance": "📈",
            "correspondence_tango": "📧", 
            "error_counts": "🚨",
            "online_ora_errors": "💻",
            "batch_ora_errors": "📊",
            "user_impact": "👥",
            "mass_update": "🔄",
            "interfaces": "🔗", 
            "hung_threads": "🧵",
            "extra_batch_connections": "⚡",
            "data_warehouse": "🏢",
            "consolidated_inquiry": "🔍",
            "batch_status": "⚙️",
            "uat_batch_status": "🧪",
            "prd_batch_status": "🏭",
            "summary": "📋"
        }
        
        section_color = section_colors.get(selected_section, "#1f4e79")
        section_icon = section_icons.get(selected_section, "📊")
        
        # Render colored header for all sections
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {section_color} 0%, {section_color}CC 100%);
            color: white;
            padding: 25px 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        ">
            <div style="display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 2.5rem; margin-right: 15px;">{section_icon}</span>
                <h1 style="
                    margin: 0;
                    font-size: 2rem;
                    font-weight: 700;
                    color: white !important;
                ">{title}</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
            "online_ora_errors": self.render_online_ora_errors_content,
            "batch_ora_errors": self.render_batch_ora_errors_content,
            "user_impact": self.render_user_impact_content,
            "mass_update": self.render_mass_update_content,
            "interfaces": self.render_interfaces_content,
            "extra_batch_connections": self.render_extra_batch_connections_content,
            "hung_threads": self.render_hung_threads_content,
            "data_warehouse": self.render_data_warehouse_content,
            "consolidated_inquiry": self.render_consolidated_inquiry_content,
            "miscellaneous_bridges": self.render_other_critical_processes_content,
            "daily_exceptions": self.render_daily_exceptions_content,
            "prd_online_exceptions": self.render_prd_online_exceptions_content,
            "prd_batch_exceptions": self.render_prd_batch_exceptions_content,
            "prd_batch_runtime": self.render_prd_batch_runtime_content,
            "uat_online_exceptions": self.render_uat_online_exceptions_content,
            "uat_batch_exceptions": self.render_uat_batch_exceptions_content,
            "uat_batch_runtime": self.render_uat_batch_runtime_content,
            "batch_status": self.render_batch_status_content,
            "uat_batch_status": self.render_uat_batch_status_content,
            "prd_batch_status": self.render_prd_batch_status_content
        }
        
        handler = section_handlers.get(selected_section)
        if handler:
            handler(filtered_by_period_df, selected_period)
        else:
            st.error(f"Handler not implemented for section: {selected_section}")

    def render_summary_home_page(self) -> None:
        """Render System Summary as the main home page with navigation cards."""
        
        # Keep only grey hover effect for navigation buttons
        st.markdown("""
        <style>
        /* Grey hover effect for all buttons */
        .stButton > button:hover {
            background-color: #6c757d !important;
            background: #6c757d !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced header using Streamlit native components with custom styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1976d2 0%, #1565c0 50%, #0d47a1 100%); color: white; padding: 40px 30px; border-radius: 15px; text-align: center; margin: 0 0 30px 0; box-shadow: 0 8px 32px rgba(25, 118, 210, 0.3); border: 2px solid rgba(255, 255, 255, 0.1);">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <div style="background: rgba(255, 255, 255, 0.15); border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; margin-right: 25px; font-size: 2.5rem;">🏠</div>
                <div style="text-align: left;">
                    <h1 style="margin: 0 0 10px 0; font-size: 3rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">MDHHS Bridges</h1>
                    <p style="margin: 0; font-size: 1.3rem; color: rgba(255, 255, 255, 0.9); font-weight: 400; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">Monitoring & Operations Status Center</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add description using native Streamlit info box
        st.info("🎯 **Your Command Center:** Monitor system health, track performance metrics, and access detailed reports all in one place.\n\n📊 Real-time monitoring • 🔍 Detailed analytics • ⚡ Quick insights • 🛡️ System oversight")
        
        # Date selection functionality
        from datetime import datetime, timedelta
        
        # Get the most recent weekday date as default
        recent_date = self.get_most_recent_weekday_date()
        default_date = datetime.strptime(recent_date, '%Y-%m-%d').date() if recent_date != "No recent data" else datetime.now().date()
        
        # Initialize selected date in session state if not exists
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = default_date
        
        # Data Date status card with current selection
        current_date_display = st.session_state.selected_date.strftime('%Y-%m-%d')
        status_cards_html = f"""
        <div style="display: flex; justify-content: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%); color: white; padding: 24px 40px; border-radius: 12px; text-align: center; box-shadow: 0 4px 16px rgba(23, 162, 184, 0.3); min-width: 300px;">
                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 8px;">📅 Current Data Date</div>
                <div style="font-size: 1.4rem; font-weight: 500; opacity: 0.95;">{current_date_display}</div>
            </div>
        </div>
        """
        st.markdown(status_cards_html, unsafe_allow_html=True)
        
        # Date picker section
        st.markdown("### 📅 Select Different Date")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Date picker
            selected_date = st.date_input(
                "Choose a date to view data from:",
                value=st.session_state.selected_date,
                min_value=datetime(2020, 1, 1).date(),
                max_value=datetime.now().date(),
                help="Select any date to view historical data across all dashboards"
            )
            
            # Update session state if date changed
            if selected_date != st.session_state.selected_date:
                st.session_state.selected_date = selected_date
                st.success(f"✅ Date updated to {selected_date.strftime('%Y-%m-%d')}. This will apply to all dashboards.")
                st.rerun()
            
            # Reset to latest date button
            if st.button("🔄 Reset to Latest Available Date", use_container_width=True):
                st.session_state.selected_date = default_date
                st.success(f"✅ Date reset to {default_date.strftime('%Y-%m-%d')}")
                st.rerun()
        
        st.markdown("---")
        
        # Show alert if viewing historical data
        from datetime import datetime
        selected_date = self.get_selected_date()
        recent_date_str = self.get_most_recent_weekday_date()
        default_date = datetime.strptime(recent_date_str, '%Y-%m-%d').date() if recent_date_str != "No recent data" else datetime.now().date()
        
        if selected_date != default_date:
            st.warning(f"📅 **Historical View Active**: All dashboards will show data for **{selected_date.strftime('%B %d, %Y')}**. Use the date picker above to change or reset to latest data.")
        
        self.render_dashboard_navigation_cards()

    def render_dashboard_navigation_cards(self) -> None:
        """Render navigation cards for all dashboard sections."""
        
        # Define dashboard sections for navigation
        dashboard_sections = [
            {
                "key": "batch_status", 
                "name": "Batch Status",
                "icon": "⚙️",
                "description": "Monitor UAT & Production batch job status and failures",
                "priority": "high"
            },
            {
                "key": "correspondence_tango",
                "name": "Correspondence", 
                "icon": "📧",
                "description": "Track Tango monitoring & file upload status",
                "priority": "high"
            },
            {
                "key": "user_impact",
                "name": "User Impact",
                "icon": "👥", 
                "description": "Monitor user experience metrics & error percentages",
                "priority": "high"
            },
            {
                "key": "error_counts",
                "name": "100 Error Counts",
                "icon": "🚨",
                "description": "Track session timeouts & critical system errors",
                "priority": "high"
            },
            {
                "key": "online_ora_errors",
                "name": "Online ORA Errors",
                "icon": "💻",
                "description": "Monitor online Oracle database errors and exceptions",
                "priority": "high"
            },
            {
                "key": "batch_ora_errors",
                "name": "Batch ORA Errors",
                "icon": "📊",
                "description": "Track batch processing Oracle database errors",
                "priority": "high"
            },
            {
                "key": "benefit_issuance",
                "name": "Benefit Issuance",
                "icon": "📈", 
                "description": "Monitor FAP, FIP, SDA processing & issuance status",
                "priority": "business"
            },
            {
                "key": "daily_exceptions",
                "name": "Daily Exceptions",
                "icon": "⚠️",
                "description": "Track online & batch exceptions across environments",
                "priority": "business"
            },
            {
                "key": "miscellaneous_bridges",
                "name": "Other Critical Processes",
                "icon": "🔗",
                "description": "Monitor critical system processes & operations",
                "priority": "business"
            }
        ]
        
        # Create status cards in a grid layout
        st.markdown("### 🎛️ Bridges M&O Summary")
        
        # Display the selected date consistently across all dashboards
        selected_date = self.get_selected_date()
        st.markdown(f"**Data as of:** {selected_date.strftime('%B %d, %Y')}")
        st.markdown("")  # Add spacing
        
        # Create status cards in properly aligned rows of 3 cards each
        # Calculate number of rows needed (7 sections = 3 + 3 + 1)
        total_sections = len(dashboard_sections)
        sections_per_row = 3
        
        # Process sections in batches of 3
        for row_idx in range(0, total_sections, sections_per_row):
            # Create columns for this row
            cols = st.columns(sections_per_row, gap="medium")
            
            # Fill columns with cards
            for col_idx in range(sections_per_row):
                section_idx = row_idx + col_idx
                if section_idx < total_sections:
                    section = dashboard_sections[section_idx]
                    
                    with cols[col_idx]:
                        # Get status for this section
                        try:
                            status, status_color, status_text = self.get_section_status(section["key"])
                        except Exception as e:
                            status, status_color, status_text = ("normal", "#6c757d", "Status monitoring available")
                        
                        # Render the status card with navigational button
                        self.render_status_card(
                            title=section["name"],
                            icon=section["icon"],
                            description=section["description"],
                            status=status,
                            color=status_color,
                            status_text=status_text
                        )
                        
                        # Add navigation button below each card
                        if st.button(
                            f"📊 View {section['name']} Dashboard", 
                            key=f"nav_to_{section['key']}", 
                            help=f"Navigate to {section['name']} section",
                            use_container_width=True
                        ):
                            st.session_state.selected_section = section["key"]
                            st.session_state.selected_subsection = None
                            st.rerun()        # Separate sections by priority for the old navigation (keep this as fallback)
        high_priority = [s for s in dashboard_sections if s["priority"] == "high"]
        business_intel = [s for s in dashboard_sections if s["priority"] == "business"]
        
        # Add some spacing and instructions
        st.markdown("---")
        st.info("""
        **💡 How to use the Summary Dashboard:**
        
        - **Green cards** indicate normal operations
        - **Yellow cards** suggest attention needed  
        - **Red cards** require immediate action
        - Click the button below each summary card to access that section's dashboard
        - Each dashboard provides detailed monitoring data and analytics
        """)

    def render_section_cards(self, sections: list) -> None:
        """Render cards for a list of sections."""
        # Create cards in rows of 2
        for i in range(0, len(sections), 2):
            col1, col2 = st.columns(2, gap="large")
            
            # First card
            if i < len(sections):
                with col1:
                    self.render_navigation_card(sections[i])
            
            # Second card (if exists)
            if i + 1 < len(sections):
                with col2:
                    self.render_navigation_card(sections[i + 1])

    def render_navigation_card(self, section: dict) -> None:
        """Render a single navigation card with navigation button below."""
        # Get status for this section
        try:
            status, status_color, status_text = self.get_section_status(section["key"])
        except:
            status, status_color, status_text = ("normal", "#6c757d", "Status available")
        
        # Determine card styling based on status
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
        
        # Render the card content
        st.markdown(f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: {bg_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 15px;">{section['icon']}</span>
                <h4 style="margin: 0; color: #333;">{section['name']}</h4>
            </div>
            <p style="color: #666; margin-bottom: 12px; font-size: 14px;">
                {section['description']}
            </p>
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="margin-right: 8px;">{status_icon}</span>
                <span style="color: {status_color}; font-weight: bold; font-size: 13px;">
                    {status_text}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Button below the card with section-specific text
        if st.button(
            f"📊 Click Here to Open {section['name']}",
            key=f"nav_card_{section['key']}",
            help=f"Navigate to {section['name']} monitoring dashboard",
            use_container_width=True
        ):
            st.session_state.selected_section = section["key"]
            st.session_state.selected_subsection = None
            st.rerun()

    def render_summary_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Legacy method - redirects to new home page."""
        self.render_summary_home_page()
        

    def get_section_status(self, section_key: str):
        """Get status for a dashboard section. Returns (status, color, text)."""
        
        try:
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
                return self.get_other_critical_processes_status()
            elif section_key == "batch_status":
                return self.get_batch_status_summary()
            else:
                # Fallback for any unknown sections
                return ("normal", "#6c757d", "Status monitoring available")
        except Exception as e:
            # Return error status if any exception occurs
            return ("warning", "#ffc107", f"Status check failed: {str(e)}")
    
    def get_error_counts_status(self):
        """Get status for 100 Error Counts using Total Count with specific thresholds."""
        try:
            # Load data directly using ExcelDataLoader
            from src.data_loader import ExcelDataLoader
            error_counts_path = Path(__file__).parent / "Monitoring Data Files" / "100 Error Counts" / "Daily 100 Error Counts.xlsx"
            
            if not error_counts_path.exists():
                return ("warning", "#ffc107", "No error data file found")
            
            loader = ExcelDataLoader(str(error_counts_path))
            df = loader.load_data()
            
            if df is None or df.empty:
                return ("warning", "#ffc107", "No error data available")
            
            # Apply date filtering
            filtered_df = self.filter_data_by_selected_date(df)
            
            if filtered_df.empty:
                selected_date = self.get_selected_date()
                return ("normal", "#6c757d", f"No data for {selected_date}")
            
            # Get the most recent row (should be only one after date filtering)
            if len(filtered_df) > 0:
                recent_data = filtered_df.iloc[0]
                
                # Get the Total Count value using our existing method
                total_count = self.calculate_total_error_count(recent_data)
                
                if total_count is None:
                    return ("warning", "#ffc107", "Total Count not found")
                
                # Apply specific thresholds: Red ≥750, Yellow 650-749, Green ≤649
                if total_count >= 750:
                    return ("critical", "#dc3545", f"{total_count} errors detected")
                elif total_count >= 650:
                    return ("warning", "#ffc107", f"{total_count} errors detected")
                else:  # <= 649
                    return ("normal", "#28a745", f"{total_count} errors detected")
            else:
                return ("warning", "#ffc107", "No processed data available")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error loading data: {str(e)}")
    
    def get_user_impact_status(self):
        """Get status for User Impact based on 0 Errors % with 89%/90% thresholds."""
        try:
            # Load data directly using ExcelDataLoader
            from src.data_loader import ExcelDataLoader
            user_impact_path = Path(__file__).parent / "Monitoring Data Files" / "User Impact" / "Daily User Impact Status.xlsx"
            
            if not user_impact_path.exists():
                return ("warning", "#ffc107", "No user impact data file found")
            
            loader = ExcelDataLoader(str(user_impact_path))
            df = loader.load_data()
            
            if df is None or df.empty:
                return ("warning", "#ffc107", "No user impact data available")
            
            # Apply date filtering
            filtered_df = self.filter_data_by_selected_date(df)
            
            if filtered_df.empty:
                selected_date = self.get_selected_date()
                return ("normal", "#6c757d", f"No data for {selected_date}")
            
            # Process data to add calculated percentage columns (same as the dashboard)
            processed_df = self.add_user_impact_percentage_columns(filtered_df.copy())
            
            # Get the most recent row (should be only one after date filtering)
            if len(processed_df) > 0:
                recent_data = processed_df.iloc[0]
                
                # Get the 0 Errors % value using our existing method
                zero_errors_pct = self.get_zero_errors_percentage(recent_data)
                
                if zero_errors_pct is None:
                    return ("warning", "#ffc107", "0 Errors % calculation failed")
                
                # Apply the 89%/90% thresholds as originally designed
                if zero_errors_pct < 89:
                    return ("critical", "#dc3545", f"Low success rate: {zero_errors_pct:.1f}%")
                elif zero_errors_pct >= 89 and zero_errors_pct <= 90:
                    return ("warning", "#ffc107", f"Moderate success rate: {zero_errors_pct:.1f}%")
                else:  # > 90%
                    return ("normal", "#28a745", f"Good success rate: {zero_errors_pct:.1f}%")
            else:
                return ("warning", "#ffc107", "No processed data available")
                
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
            
            # Get data for the selected date
            selected_date = self.get_selected_date()
            filtered_df = self.filter_data_by_selected_date(df)
            
            if filtered_df.empty:
                return ("normal", "#6c757d", f"No correspondence data for {selected_date}")
            
            # Find the "Number of Files not sent to CPC" column
            files_not_sent_col = None
            for col in df.columns:
                if "Number of Files not sent to CPC" in str(col) or "files not sent" in str(col).lower():
                    files_not_sent_col = col
                    break
            
            if files_not_sent_col is not None and files_not_sent_col in filtered_df.columns:
                files_not_sent = filtered_df[files_not_sent_col].sum()
                if files_not_sent > 10:
                    return ("critical", "#dc3545", f"{files_not_sent} files not sent")
                elif files_not_sent > 0:
                    return ("warning", "#ffc107", f"{files_not_sent} files not sent")
                else:
                    return ("normal", "#28a745", "All files sent")
            else:
                return ("normal", "#28a745", "Data available")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Error loading data: {str(e)}")
    
    def get_benefit_issuance_status(self):
        """Get status for Benefit Issuance section."""
        try:
            # Check for benefit issuance data files
            bi_path = Path(__file__).parent / "Monitoring Data Files" / "BI Monitoring"
            
            if not bi_path.exists():
                return ("warning", "#ffc107", "Data not available")
            
            # Check for different time period folders
            folders = ["Daily", "Weekly", "Monthly", "Yearly"]
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
        """Get status for Daily Exceptions section based on PRD Online and PRD Batch exception counts."""
        try:
            from pathlib import Path
            
            # Get summary data for PRD Online and PRD Batch exceptions
            prd_online_summary = get_daily_exceptions_summary("prd_online_exceptions")
            prd_batch_summary = get_daily_exceptions_summary("prd_batch_exceptions")
            
            # Get the most recent counts (first row since data is sorted newest first)
            prd_online_count = 0
            prd_batch_count = 0
            
            if not prd_online_summary.empty:
                count_col = [col for col in prd_online_summary.columns if col != "Date"][0]
                prd_online_count = prd_online_summary.iloc[0][count_col] if len(prd_online_summary) > 0 else 0
            
            if not prd_batch_summary.empty:
                count_col = [col for col in prd_batch_summary.columns if col != "Date"][0]
                prd_batch_count = prd_batch_summary.iloc[0][count_col] if len(prd_batch_summary) > 0 else 0
            
            # Apply thresholds - if EITHER PRD Online OR PRD Batch exceeds thresholds
            max_count = max(prd_online_count, prd_batch_count)
            
            if max_count > 15:
                return ("critical", "#dc3545", f"High PRD exceptions: Online({prd_online_count}) Batch({prd_batch_count})")
            elif max_count >= 11:
                return ("warning", "#ffc107", f"Moderate PRD exceptions: Online({prd_online_count}) Batch({prd_batch_count})")
            elif max_count <= 10:
                return ("normal", "#28a745", f"Normal PRD exceptions: Online({prd_online_count}) Batch({prd_batch_count})")
            else:
                return ("normal", "#28a745", f"PRD exceptions: Online({prd_online_count}) Batch({prd_batch_count})")
                
        except Exception as e:
            return ("warning", "#ffc107", "Unable to check exception data")
    
    def get_other_critical_processes_status(self):
        """Get status for Other Critical Processes section."""
        try:
            # Check for various critical process data
            processes = ["Mass Update", "Interfaces", "Extra Batch Connections", "Hung Threads"]
            
            # Since no real data source is configured yet, return data not available
            return ("warning", "#ffc107", "Data not available")
                
        except Exception as e:
            return ("warning", "#ffc107", "Data not available")

    def get_batch_status_summary(self):
        """Get status summary for Batch Status based on UAT and Production failed jobs."""
        try:
            from pathlib import Path
            
            # Check Production batch status first (higher priority)
            prd_path = Path(__file__).parent / "Monitoring Data Files" / "Batch Status" / "Production.xlsx"
            uat_path = Path(__file__).parent / "Monitoring Data Files" / "Batch Status" / "UAT.xlsx"
            
            prd_failed_jobs = 0
            uat_failed_jobs = 0
            
            # Check Production data
            if prd_path.exists():
                try:
                    from src.data_loader import ExcelDataLoader
                    loader = ExcelDataLoader(str(prd_path))
                    prd_df = loader.load_data()
                    
                    if not prd_df.empty:
                        filtered_prd = self.filter_batch_status_data(prd_df)
                        prd_failed_jobs = len(filtered_prd)
                except:
                    pass
            
            # Check UAT data
            if uat_path.exists():
                try:
                    from src.data_loader import ExcelDataLoader
                    loader = ExcelDataLoader(str(uat_path))
                    uat_df = loader.load_data()
                    
                    if not uat_df.empty:
                        filtered_uat = self.filter_batch_status_data(uat_df)
                        uat_failed_jobs = len(filtered_uat)
                except:
                    pass
            
            # Apply priority logic: Production failures take precedence
            if prd_failed_jobs > 0:
                return ("critical", "#dc3545", f"{prd_failed_jobs} Failed Jobs in PRD")
            elif uat_failed_jobs > 0:
                return ("warning", "#ffc107", f"{uat_failed_jobs} Failed Jobs in UAT")
            else:
                return ("normal", "#28a745", "No Failed Jobs")
                
        except Exception as e:
            return ("warning", "#ffc107", f"Status check failed: {str(e)}")
    
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
            
            return target_date.strftime("%Y-%m-%d")
            
        except Exception:
            return "No recent data"
    
    def get_selected_date(self):
        """Get the currently selected date from session state or default to most recent."""
        from datetime import datetime, timedelta
        
        if 'selected_date' in st.session_state:
            return st.session_state.selected_date
        else:
            # Fallback to most recent weekday
            recent_date_str = self.get_most_recent_weekday_date()
            if recent_date_str != "No recent data":
                return datetime.strptime(recent_date_str, '%Y-%m-%d').date()
            else:
                return datetime.now().date()
    
    def get_selected_date_formatted(self, format_type="display"):
        """Get the selected date in various formats for data filtering."""
        selected_date = self.get_selected_date()
        
        if format_type == "display":
            return selected_date.strftime("%B %d, %Y")
        elif format_type == "filename":
            return selected_date.strftime("%Y-%m-%d")
        elif format_type == "filter":
            return selected_date.strftime("%Y-%m-%d")
        else:
            return selected_date.strftime("%Y-%m-%d")
    
    def filter_data_by_selected_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to show only data for the selected date or week."""
        if df is None or df.empty:
            return df
            
        # Get selected date directly from session state to ensure it's current
        if 'selected_date' in st.session_state:
            selected_date = st.session_state.selected_date
        else:
            # Fallback to most recent weekday
            from datetime import datetime
            recent_date_str = self.get_most_recent_weekday_date()
            if recent_date_str != "No recent data":
                selected_date = datetime.strptime(recent_date_str, '%Y-%m-%d').date()
            else:
                selected_date = datetime.now().date()
        
        # Check if we're in a weekly or monthly dashboard
        selected_period = st.session_state.get('selected_period', 'daily')
        is_weekly_dashboard = selected_period == 'weekly'
        is_monthly_dashboard = selected_period == 'monthly'
        
        # Try to find date columns in the dataframe
        date_columns = []
        week_start_col = None
        week_end_col = None
        week_of_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(date_word in col_lower for date_word in ['date', 'day', 'time', 'created', 'updated', 'week']):
                date_columns.append(col)
                
                # Identify specific week columns
                if 'week start' in col_lower:
                    week_start_col = col
                elif 'week end' in col_lower:
                    week_end_col = col
                elif 'week of' in col_lower:
                    week_of_col = col
        
        # If no obvious date columns found, return empty dataframe (no data for selected date)
        if not date_columns:
            return pd.DataFrame(columns=df.columns)
        
        # Filter by the selected date or week
        filtered_df = df.copy()
        found_matching_data = False
        
        if is_weekly_dashboard:
            # For weekly dashboards, check if this is benefit issuance section
            selected_section = st.session_state.get('selected_section', '')
            
            if selected_section == 'benefit_issuance':
                # Special logic for Weekly Benefit Issuance: check for Week Start/End columns first
                if week_start_col and week_end_col:
                    # Use separate Week Start and Week End columns
                    try:
                        # Convert columns to datetime
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[week_start_col]):
                            filtered_df[week_start_col] = pd.to_datetime(filtered_df[week_start_col], errors='coerce')
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[week_end_col]):
                            filtered_df[week_end_col] = pd.to_datetime(filtered_df[week_end_col], errors='coerce')
                        
                        # Filter for rows where selected date falls between week start and end
                        mask = (filtered_df[week_start_col].dt.date <= selected_date) & \
                               (filtered_df[week_end_col].dt.date >= selected_date)
                        temp_filtered = filtered_df[mask]
                        
                        if not temp_filtered.empty:
                            filtered_df = temp_filtered
                            found_matching_data = True
                            
                    except Exception as e:
                        print(f"Error filtering by Week Start/End columns: {e}")
                
                # If no Week Start/End columns or filtering failed, try Week Of column
                if not found_matching_data:
                    for date_col in date_columns:
                        try:
                            # Convert the date column to datetime if it's not already
                            if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                                filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                            
                            # Check if this is a "Week Of" column specifically
                            col_lower = date_col.lower()
                            if 'week of' in col_lower:
                                # For "Week Of" columns, look for rows where the week range contains our selected date
                                mask = pd.Series([False] * len(filtered_df))
                                
                                for idx, week_data in filtered_df[date_col].items():
                                    if pd.isna(week_data):
                                        continue
                                    try:
                                        week_str = str(week_data).strip()
                                        
                                        # Handle the format: "27 Oct 2025 to 31 Oct 2025"
                                        found_match = False
                                        
                                        # Check if it's a date range with "to"
                                        if ' to ' in week_str.lower():
                                            parts = week_str.split(' to ')
                                            if len(parts) == 2:
                                                start_str = parts[0].strip()
                                                end_str = parts[1].strip()
                                                
                                                start_date = pd.to_datetime(start_str, errors='coerce')
                                                end_date = pd.to_datetime(end_str, errors='coerce')
                                                
                                                if pd.notna(start_date) and pd.notna(end_date):
                                                    if start_date.date() <= selected_date <= end_date.date():
                                                        found_match = True
                                        
                                        # Check if it contains multiple dates (comma-separated)
                                        elif ',' in week_str:
                                            dates_in_week = [d.strip() for d in week_str.split(',')]
                                            for date_str in dates_in_week:
                                                try:
                                                    date_obj = pd.to_datetime(date_str, errors='coerce')
                                                    if pd.notna(date_obj) and date_obj.date() == selected_date:
                                                        found_match = True
                                                        break
                                                except:
                                                    continue
                                        
                                        # Check if it's a single date (the start of the week)
                                        else:
                                            date_obj = pd.to_datetime(week_str, errors='coerce')
                                            if pd.notna(date_obj):
                                                # Check if selected date falls in the same week as this date
                                                week_start_date = date_obj.date()
                                                selected_datetime = datetime.combine(selected_date, datetime.min.time())
                                                week_start_datetime = datetime.combine(week_start_date, datetime.min.time())
                                                
                                                # Calculate if they're in the same week (within 4 days for weekdays)
                                                days_diff = abs((selected_datetime - week_start_datetime).days)
                                                if days_diff <= 4:  # Within the same week (Mon-Fri)
                                                    found_match = True
                                        
                                        if found_match:
                                            mask.iloc[idx] = True
                                            
                                    except Exception as e:
                                        # Debug: log parsing errors
                                        print(f"Error parsing week data '{week_data}': {e}")
                                        continue
                                
                                temp_filtered = filtered_df[mask]
                                
                                # If we have matching data, use it
                                if not temp_filtered.empty:
                                    filtered_df = temp_filtered
                                    found_matching_data = True
                                    break
                                    
                        except Exception:
                            continue
            else:
                # For other weekly dashboards (non-benefit issuance), use standard week calculation
                from datetime import datetime, timedelta
                selected_datetime = datetime.combine(selected_date, datetime.min.time())
                days_since_monday = selected_datetime.weekday()
                week_start = selected_datetime - timedelta(days=days_since_monday)
                week_end = week_start + timedelta(days=4)  # Friday (only weekdays)
                
                for date_col in date_columns:
                    try:
                        # Convert the date column to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                        
                        # Filter for weekdays within the week
                        mask = (filtered_df[date_col].dt.date >= week_start.date()) & \
                               (filtered_df[date_col].dt.date <= week_end.date()) & \
                               (filtered_df[date_col].dt.dayofweek < 5)  # Only weekdays (0-4)
                        temp_filtered = filtered_df[mask]
                        
                        # If we have matching data, use it
                        if not temp_filtered.empty:
                            filtered_df = temp_filtered
                            found_matching_data = True
                            break
                            
                    except Exception:
                        continue
        elif is_monthly_dashboard:
            # For monthly dashboards, filter for the month containing the selected date
            for date_col in date_columns:
                try:
                    # Convert the date column to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                    
                    from datetime import datetime
                    import calendar
                    
                    # Convert selected_date to datetime for calculation
                    selected_datetime = datetime.combine(selected_date, datetime.min.time())
                    
                    # Calculate the start and end of the month for the selected date
                    month_start = selected_datetime.replace(day=1)
                    last_day = calendar.monthrange(selected_datetime.year, selected_datetime.month)[1]
                    month_end = selected_datetime.replace(day=last_day)
                    
                    # Filter for dates within this month
                    mask = (filtered_df[date_col].dt.date >= month_start.date()) & \
                           (filtered_df[date_col].dt.date <= month_end.date())
                    temp_filtered = filtered_df[mask]
                    
                    # If we have matching data, use it
                    if not temp_filtered.empty:
                        filtered_df = temp_filtered
                        found_matching_data = True
                        break
                        
                except Exception:
                    continue
        else:
            # For daily dashboards, filter for the exact selected date
            for date_col in date_columns:
                try:
                    # Convert the date column to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                    
                    mask = filtered_df[date_col].dt.date == selected_date
                    temp_filtered = filtered_df[mask]
                    
                    # If we have matching data, use it
                    if not temp_filtered.empty:
                        filtered_df = temp_filtered
                        found_matching_data = True
                        break
                        
                except Exception:
                    continue
        
        # Return filtered data if found, otherwise empty DataFrame
        return filtered_df if found_matching_data else pd.DataFrame(columns=df.columns)
    
    def get_filtered_dashboard_data(self, section: str, period: str = "daily") -> pd.DataFrame:
        """Get the same filtered data that sub-dashboards use."""
        import pandas as pd
        
        try:
            # Load the raw data
            result = self.auto_load_excel_file(section, period)
            if not result or result.get("df") is None:
                return pd.DataFrame()
            
            df = result["df"]
            
            # Apply the same filtering logic as sub-dashboards
            from src.data_loader import sort_dataframe_by_date, format_dataframe_dates
            
            # Sort by date columns (latest to oldest) - same as sub-dashboards
            df = sort_dataframe_by_date(df, ascending=False)
            
            # Apply standardized date formatting - same as sub-dashboards
            df = format_dataframe_dates(df)
            
            # Filter by selected date only - same as sub-dashboards
            df = self.filter_data_by_selected_date(df)
            
            return df
        except Exception as e:
            st.error(f"❌ Error in get_filtered_dashboard_data: {str(e)}")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def calculate_date_based_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics based on the same filtered data used by sub-dashboards."""
        from datetime import datetime, timedelta
        import numpy as np
        import pandas as pd
        
        # Get selected date directly from session state to ensure it's current
        if 'selected_date' in st.session_state:
            selected_date = st.session_state.selected_date
        else:
            # Fallback to most recent weekday
            from datetime import datetime
            recent_date_str = self.get_most_recent_weekday_date()
            if recent_date_str != "No recent data":
                selected_date = datetime.strptime(recent_date_str, '%Y-%m-%d').date()
            else:
                selected_date = datetime.now().date()
        
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        
        metrics = {
            "total_errors": "Loading...",
            "error_change": "",
            "active_processes": "Loading...", 
            "process_change": "",
            "system_health": "Loading...",
            "health_change": "",
            "data_freshness": "Loading...",
            "freshness_indicator": "",
            "user_impact": "Loading...",
            "user_impact_change": ""
        }
        
        try:
            # Get filtered error data - same as sub-dashboards
            filtered_errors = self.get_filtered_dashboard_data("error_counts", "daily")
            
            if not filtered_errors.empty:
                # Count total error records for selected date
                total_errors = len(filtered_errors)
                metrics["total_errors"] = f"{total_errors:,}"
                
                # Calculate sum of error counts if there are numeric columns
                numeric_cols = filtered_errors.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Exclude obvious non-error columns
                    error_cols = [col for col in numeric_cols if not any(skip in col.lower() for skip in ['id', 'index', 'year', 'month', 'day'])]
                    if error_cols:
                        total_error_count = filtered_errors[error_cols].sum().sum()
                        metrics["total_errors"] = f"{int(total_error_count):,}" if total_error_count > total_errors else f"{total_errors:,}"
                
                metrics["error_change"] = "📊"  # Show data indicator instead of comparison
            else:
                metrics["total_errors"] = "0"
                metrics["error_change"] = "📊"
            
            # Get filtered batch data - same as sub-dashboards
            filtered_batch = self.get_filtered_dashboard_data("batch_status", "daily")
                
            if not filtered_batch.empty:
                active_processes = len(filtered_batch)
                metrics["active_processes"] = f"{active_processes:,}"
                
                # Calculate success rate for system health
                success_indicators = ['success', 'complete', 'ok', 'pass', 'good']
                error_indicators = ['error', 'fail', 'abort', 'timeout', 'bad']
                
                # Check for status columns
                status_cols = [col for col in filtered_batch.columns if 'status' in col.lower()]
                
                if status_cols:
                    status_col = status_cols[0]
                    success_count = 0
                    error_count = 0
                    
                    for _, row in filtered_batch.iterrows():
                        status_val = str(row[status_col]).lower()
                        if any(indicator in status_val for indicator in success_indicators):
                            success_count += 1
                        elif any(indicator in status_val for indicator in error_indicators):
                            error_count += 1
                    
                    total_with_status = success_count + error_count
                    if total_with_status > 0:
                        success_rate = (success_count / total_with_status) * 100
                        metrics["system_health"] = f"{success_rate:.0f}%"
                        
                        if success_rate >= 95:
                            metrics["health_change"] = "✅"
                        elif success_rate >= 80:
                            metrics["health_change"] = "⚠️"
                        else:
                            metrics["health_change"] = "🔴"
                    else:
                        metrics["system_health"] = "Unknown"
                        metrics["health_change"] = "❓"
                else:
                    metrics["system_health"] = "N/A"
                    metrics["health_change"] = "❓"
                        
                metrics["process_change"] = "📊"
            else:
                metrics["active_processes"] = "0"
                metrics["system_health"] = "No Data"
                metrics["health_change"] = "❓"
            
            # Get filtered user impact data - same as sub-dashboards
            filtered_user_impact = self.get_filtered_dashboard_data("user_impact", "daily")
            
            if not filtered_user_impact.empty:
                # Count affected users/records
                user_impact_count = len(filtered_user_impact)
                
                # Look for numeric columns that might indicate impact volume
                numeric_cols = filtered_user_impact.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Exclude obvious non-impact columns
                    impact_cols = [col for col in numeric_cols if not any(skip in col.lower() for skip in ['id', 'index', 'year', 'month', 'day'])]
                    if impact_cols:
                        total_impact = filtered_user_impact[impact_cols].sum().sum()
                        if total_impact > user_impact_count:
                            user_impact_count = int(total_impact)
                
                # Add user impact to metrics (replace one of the existing metrics or add as 5th metric)
                metrics["user_impact"] = f"{user_impact_count:,}"
                metrics["user_impact_change"] = "📊"
            else:
                metrics["user_impact"] = "0"
                metrics["user_impact_change"] = "📊"
            
            # Data freshness indicator
            recent_date_str = self.get_most_recent_weekday_date()
            if recent_date_str != "No recent data":
                default_date = datetime.strptime(recent_date_str, '%Y-%m-%d').date()
                if selected_date == default_date:
                    metrics["data_freshness"] = "Latest"
                    metrics["freshness_indicator"] = "🟢"
                else:
                    days_old = (default_date - selected_date).days
                    metrics["data_freshness"] = f"{days_old}d ago"
                    if days_old <= 1:
                        metrics["freshness_indicator"] = "🟡"
                    elif days_old <= 7:
                        metrics["freshness_indicator"] = "🟠"
                    else:
                        metrics["freshness_indicator"] = "🔴"
            else:
                metrics["data_freshness"] = "Unknown"
                metrics["freshness_indicator"] = "❓"
                
        except Exception as e:
            # If there are any errors, show safe defaults
            st.error(f"🚨 Error in summary metrics calculation: {str(e)}")
            st.write(f"🔍 Exception details: {type(e).__name__}")
            import traceback
            st.write(f"🔍 Traceback: {traceback.format_exc()}")
            metrics = {
                "total_errors": "Error",
                "error_change": "",
                "active_processes": "Error",
                "process_change": "",
                "system_health": "Error", 
                "health_change": "",
                "data_freshness": "Error",
                "freshness_indicator": "❓",
                "user_impact": "Error",
                "user_impact_change": ""
            }
        
        return metrics
    
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
            if len(df) > 0:
                return df.iloc[-1]
            else:
                return None
        
        # Convert date column to datetime for comparison
        date_col = date_columns[0]
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if df_copy.empty:
                if len(df) > 0:
                    return df.iloc[-1]
                else:
                    return None
            
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
            if len(df) > 0:
                return df.iloc[-1]
            else:
                return None
    
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
        
        # Calculate dynamic font size based on text length
        if len(status_text) > 50:
            status_font_size = "0.65rem"
            status_line_height = "1.1"
        elif len(status_text) > 35:
            status_font_size = "0.7rem"  
            status_line_height = "1.1"
        elif len(status_text) > 25:
            status_font_size = "0.75rem"
            status_line_height = "1.2"
        else:
            status_font_size = "0.85rem"
            status_line_height = "1.2"
        
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
            overflow: hidden;
            position: relative;
        ">
            <div style="flex: 1; overflow: hidden;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem; margin-right: 8px;">{icon}</span>
                    <h4 style="margin: 0; color: #333; font-size: 1.1rem; line-height: 1.2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;">{title}</h4>
                </div>
                <p style="margin: 5px 0; color: #666; font-size: 0.85rem; line-height: 1.3; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">{description}</p>
            </div>
            <div style="display: flex; align-items: center; margin-top: auto; min-height: 35px; overflow: hidden;">
                <span style="margin-right: 8px; flex-shrink: 0; display: flex; align-items: center; height: 100%;">{status_icon}</span>
                <span style="
                    color: {color}; 
                    font-weight: bold; 
                    font-size: {status_font_size}; 
                    line-height: {status_line_height}; 
                    word-wrap: break-word; 
                    word-break: break-word;
                    hyphens: auto;
                    overflow: hidden; 
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    flex: 1;
                    max-height: 30px;
                    align-self: center;
                ">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


    def check_and_display_no_data_message(self, df: pd.DataFrame, selected_period: str = None) -> bool:
        """Check if dataframe is empty and display consistent no data message.
        Returns True if data is empty, False if data exists."""
        if df.empty:
            st.warning("⚠️ No Data available for this time period")
            return True
        return False

    def render_benefit_issuance_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Benefit Issuance specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Debug: Show what data was found after filtering for weekly dashboards
        if selected_period == 'weekly' and df.empty:
            st.warning("⚠️ No Data available for this time period")
            
            # Show available week ranges in debug
            try:
                original_result = self.auto_load_excel_file(st.session_state.get('selected_section'), selected_period)
                if original_result and original_result.get("df") is not None:
                    orig_df = original_result["df"]
                    
                    # Check for Week Start/End columns
                    week_start_col = None
                    week_end_col = None
                    week_of_col = None
                    
                    for col in orig_df.columns:
                        col_lower = col.lower()
                        if 'week start' in col_lower:
                            week_start_col = col
                        elif 'week end' in col_lower:
                            week_end_col = col
                        elif 'week of' in col_lower:
                            week_of_col = col
                    
                    if week_start_col and week_end_col:
                        st.write(f"📊 Found Week Start/End columns: '{week_start_col}' and '{week_end_col}'")
                        # Show sample data from both columns
                        sample_data = orig_df[[week_start_col, week_end_col]].dropna().head(5)
                        st.write("Sample week ranges:")
                        for idx, row in sample_data.iterrows():
                            start_date = pd.to_datetime(row[week_start_col], errors='coerce')
                            end_date = pd.to_datetime(row[week_end_col], errors='coerce')
                            if pd.notna(start_date) and pd.notna(end_date):
                                st.write(f"  • {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
                    
                    elif week_of_col:
                        st.write(f"📊 Available data in column '{week_of_col}':")
                        unique_weeks = orig_df[week_of_col].dropna().unique()
                        for i, week in enumerate(unique_weeks[:10]):  # Show first 10
                            st.write(f"  {i+1}. `{week}` (type: {type(week).__name__})")
                        if len(unique_weeks) > 10:
                            st.write(f"  ... and {len(unique_weeks) - 10} more entries")
                    
                    selected_date = self.get_selected_date()
                    st.write(f"🎯 Looking for week containing: **{selected_date.strftime('%Y-%m-%d')}** ({selected_date.strftime('%A')})")
                    
                    # Show what the system would calculate as the week range
                    from datetime import datetime, timedelta
                    selected_datetime = datetime.combine(selected_date, datetime.min.time())
                    days_since_monday = selected_datetime.weekday()
                    week_start = selected_datetime - timedelta(days=days_since_monday)
                    week_end = week_start + timedelta(days=4)
                    st.write(f"📅 Expected week range: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
                        
            except Exception as e:
                st.error(f"Debug error: {e}")
        elif selected_period == 'weekly' and not df.empty:
            st.success(f"✅ Found {len(df)} row(s) of data for the selected week!")
        elif df.empty:
            st.warning("⚠️ No Data available for this time period")
        
        # Display the selected date for clarity
        selected_date = self.get_selected_date()
        
        if selected_period == 'weekly':
            # For weekly dashboards, show the weekday range
            from datetime import datetime, timedelta
            selected_datetime = datetime.combine(selected_date, datetime.min.time())
            days_since_monday = selected_datetime.weekday()  # Monday = 0, Sunday = 6
            week_start = selected_datetime - timedelta(days=days_since_monday)
            week_end = week_start + timedelta(days=4)  # Friday (only weekdays)
            
            st.markdown(f"**📅 Data for week of:** {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')} (weekdays only, containing selected date: {selected_date.strftime('%Y-%m-%d')})")
        elif selected_period == 'monthly':
            # For monthly dashboards, show the month range
            from datetime import datetime
            import calendar
            selected_datetime = datetime.combine(selected_date, datetime.min.time())
            month_start = selected_datetime.replace(day=1)
            last_day = calendar.monthrange(selected_datetime.year, selected_datetime.month)[1]
            month_end = selected_datetime.replace(day=last_day)
            
            st.markdown(f"**📅 Data for month of:** {month_start.strftime('%B %Y')} ({month_start.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}) (containing selected date: {selected_date.strftime('%Y-%m-%d')})")
        else:
            st.markdown(f"**📅 Data as of:** {selected_date.strftime('%Y-%m-%d')}")
        st.markdown("---")
        
        # Special preprocessing for Vendoring Payments BEFORE filtering
        if selected_subsection and "Vendoring Payments" in selected_subsection:
            df = self.preprocess_vendoring_payments(df)
        
        # Filters in main area with expander
        with st.expander("🔍 **Data Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
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
    
    def preprocess_vendoring_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Vendoring Payments data to handle empty rows before filtering."""
        import pandas as pd
        
        # Apply standard processing first
        sorted_df = sort_dataframe_by_date(df, ascending=False)
        formatted_df = format_dataframe_dates(sorted_df)
        display_df = formatted_df.copy()
        
        # Apply currency formatting
        for col in display_df.columns:
            col_lower = col.lower()
            is_currency_col = ('amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower))
            is_variance_col = 'variance' in col_lower
            if is_currency_col and not is_variance_col:
                try:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                except:
                    pass
        
        # Apply percentage formatting
        display_df = format_percentage_columns(display_df)
        
        # Find date column (Week Of or Date)
        date_col = None
        for col in display_df.columns:
            if 'week of' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        # Find all non-date columns
        data_cols = [col for col in display_df.columns if col != date_col]
        
        if date_col and data_cols:
            # Check each row - if all data columns are null/empty, restructure
            rows_to_update = []
            for idx, row in display_df.iterrows():
                all_empty = True
                for col in data_cols:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() != '' and val != 0:
                        all_empty = False
                        break
                if all_empty:
                    rows_to_update.append(idx)
            
            if rows_to_update:
                # If ALL rows are empty, simplify to just date + status columns
                if len(rows_to_update) == len(display_df):
                    # All rows empty - create simplified structure
                    new_columns = [date_col, 'Status']
                    new_data = []
                    for idx in range(len(display_df)):
                        new_data.append([display_df.loc[idx, date_col], 'No Vendoring Payments'])
                    display_df = pd.DataFrame(new_data, columns=new_columns)
                else:
                    # Mixed data - keep original structure but mark empty rows
                    first_data_col = None
                    for col in data_cols:
                        if 'benefit' in col.lower():
                            first_data_col = col
                            break
                    if not first_data_col:
                        first_data_col = data_cols[0]
                    
                    # Convert columns to object type to avoid dtype warnings
                    for col in data_cols:
                        display_df[col] = display_df[col].astype('object')
                    
                    for idx in rows_to_update:
                        display_df.loc[idx, first_data_col] = 'No Vendoring Payments'
                        # Clear other data columns for empty rows
                        for col in data_cols:
                            if col != first_data_col:
                                display_df.loc[idx, col] = ''
        
        return display_df
    
    def render_benefit_issuance_table(self, df: pd.DataFrame, title: str) -> None:
        """Render benefit issuance table with variance highlighting."""
        import pandas as pd
        
        # Additional aggressive empty row removal for table display
        df = remove_empty_rows(df)
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Check if dataframe has variance-related columns, specifically "Variance in #Benefits"
        variance_columns = []
        benefits_variance_col = None
        
        # Look for all variance-related columns
        for col in df.columns:
            col_lower = col.lower().strip()
            # Check for any variance columns
            if ('variance' in col_lower):
                variance_columns.append(col)
                # Set the benefits variance column if this matches benefits patterns
                if ('benefit' in col_lower or '#benefit' in col_lower or 'benefits' in col_lower or '#benefits' in col_lower or
                    '# benefit' in col_lower or 'num benefit' in col_lower):
                    benefits_variance_col = col
        
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
            # Create a copy of the dataframe for styling
            display_df = df.copy()
            
            # Format currency columns (Amt issued fields) - exclude variance columns
            for col in display_df.columns:
                col_lower = col.lower()
                # Check for currency columns but exclude variance columns
                is_currency_col = ('amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower))
                is_variance_col = 'variance' in col_lower
                if is_currency_col and not is_variance_col:
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
            

            
            # Handle null/empty variance columns - fill with meaningful text ONLY if truly null/empty
            for col in variance_columns:
                if col in display_df.columns:
                    # Only replace truly null values, preserve existing text like "Below 10%", "Above 10%", etc.
                    display_df[col] = display_df[col].fillna('N/A')
                    # Only replace empty strings, not text values
                    mask = (display_df[col] == '') & (display_df[col].notna())
                    display_df[col] = display_df[col].where(~mask, 'N/A')
            
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
            
            # Display with custom styling
            display_clean_dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            # If no variance columns found, still apply currency formatting
            display_df = df.copy()
            
            # Format currency columns (Amt issued fields) - exclude variance columns
            for col in display_df.columns:
                col_lower = col.lower()
                # Check for currency columns but exclude variance columns
                is_currency_col = ('amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower))
                is_variance_col = 'variance' in col_lower
                if is_currency_col and not is_variance_col:
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
            
            # Handle any variance columns that might have been missed in detection
            for col in display_df.columns:
                col_lower = col.lower().strip()
                if 'variance' in col_lower:
                    # Only replace truly null/empty values, preserve existing text
                    display_df[col] = display_df[col].fillna('N/A')
                    # Only replace empty strings, not text values
                    mask = (display_df[col] == '') & (display_df[col].notna())
                    display_df[col] = display_df[col].where(~mask, 'N/A')
            
            # Use regular table
            display_clean_dataframe(display_df, use_container_width=True, hide_index=True)
    
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
            display_clean_dataframe(df, use_container_width=True, hide_index=True)
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
            
            # Show available dates 
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
            
            # Display with custom styling
            display_clean_dataframe(styled_df, use_container_width=True, hide_index=True)
            
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
            display_clean_dataframe(upload_status_formatted, use_container_width=True, hide_index=True)
        
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
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df, selected_period):
            return
        
        # Display the selected date for clarity
        selected_date = self.get_selected_date()
        st.markdown(f"**📅 Data as of:** {selected_date.strftime('%Y-%m-%d')}")
        st.markdown("---")
        
        # Filters in main area with expander
        with st.expander("🔍 **Correspondence Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        # Apply filters
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
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
            display_clean_dataframe(df.head(3), hide_index=True)
    
    def render_error_counts_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render 100 Error Counts specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df, selected_period):
            return
        
        with st.expander("🚨 **Error Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
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
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df, selected_period):
            return
        
        with st.expander("👥 **Impact Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
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
    
    def render_online_ora_errors_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Online ORA Errors specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df, selected_period):
            return
        
        with st.expander("💻 **Online ORA Error Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus
        st.header("📋 Data Table")
        file_display_name = f"{selected_subsection} ORA Errors Data"
        self.render_ora_errors_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.render_ora_errors_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Error Trends", "🔍 Error Analysis", "📈 Resolution Status"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="online_ora_")
        with tab2:
            self.table_component.summary_stats(filtered_df)
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="online_ora_")
    
    def render_batch_ora_errors_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Batch ORA Errors specific content."""
        
        selected_subsection = st.session_state.get('selected_subsection')
        
        # Only show data if a specific file is selected
        if not selected_subsection:
            # Show selection message when no file is selected
            st.info("👆 Click on a dashboard item in the sidebar to view its contents")
            return
        
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df, selected_period):
            return
        
        with st.expander("📊 **Batch ORA Error Filters**", expanded=False):
            filters = self.create_inline_filters(df)
        
        filtered_df = self.filter_component.apply_filters(df, filters)
        
        # Remove empty rows to clean up display
        filtered_df = remove_empty_rows(filtered_df)
        
        # Sort by date columns (latest to oldest)
        filtered_df = sort_dataframe_by_date(filtered_df, ascending=False)
        
        filtered_df = format_dataframe_dates(filtered_df)
        
        # Data Table - Main Focus
        st.header("📋 Data Table")
        file_display_name = f"{selected_subsection} ORA Errors Data"
        self.render_ora_errors_table(filtered_df, file_display_name)
        
        # Key Metrics - Below Data Table
        st.header(f"📊 Key Metrics")
        self.render_ora_errors_metrics(filtered_df)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Error Trends", "🔍 Error Analysis", "📈 Resolution Status"])
        
        with tab1:
            self.render_charts(filtered_df, selected_period, key_prefix="batch_ora_")
        with tab2:
            self.table_component.summary_stats(filtered_df)
        with tab3:
            self.render_custom_analysis(filtered_df, key_prefix="batch_ora_")

    def render_mass_update_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Mass Update specific content."""
        # Check if user wants to view data
        if st.session_state.get('show_mass_update_data', False):
            # Apply date filtering to ensure we show data for the selected date
            df = self.filter_data_by_selected_date(df)
            
            if df is None or df.empty:
                st.info("📋 No Mass Update data found. This could indicate:")
                st.markdown("""
                - No Excel files in the Mass Update folder
                - Files are empty or in an unsupported format
                - No data matching the selected date range
                
                **Next Steps:**
                - Check the Monitoring Data Files folder structure
                - Verify Excel files contain data
                - Try adjusting the date range filter
                """)
                
                if st.button("← Back to Mass Update Overview", key="back_to_mass_update_home"):
                    st.session_state.show_mass_update_data = False
                    st.rerun()
                return
            
            self.render_generic_section_content(df, selected_period, "Mass Update", "🔄")
            
            if st.button("← Back to Mass Update Overview", key="back_to_mass_update_home2"):
                st.session_state.show_mass_update_data = False
                st.rerun()
        else:
            # Show section home page with information
            self.render_section_home_page("mass_update")
            
            # Add button to view data
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📊 View Mass Update Data", key="view_mass_update_data", use_container_width=True, type="primary"):
                    st.session_state.show_mass_update_data = True
                    st.rerun()
    
    def render_interfaces_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Interfaces specific content."""
        # Check if user wants to view data
        if st.session_state.get('show_interfaces_data', False):
            # Apply date filtering to ensure we show data for the selected date
            df = self.filter_data_by_selected_date(df)
            
            if df is None or df.empty:
                st.info("📋 No Interfaces data found. This could indicate:")
                st.markdown("""
                - No Excel files in the Interfaces folder
                - Files are empty or in an unsupported format
                - No data matching the selected date range
                
                **Next Steps:**
                - Check the Monitoring Data Files folder structure
                - Verify Excel files contain data
                - Try adjusting the date range filter
                """)
                
                if st.button("← Back to Interfaces Overview", key="back_to_interfaces_home"):
                    st.session_state.show_interfaces_data = False
                    st.rerun()
                return
            
            self.render_generic_section_content(df, selected_period, "Interfaces", "🔗")
            
            if st.button("← Back to Interfaces Overview", key="back_to_interfaces_home2"):
                st.session_state.show_interfaces_data = False
                st.rerun()
        else:
            # Show section home page with information
            self.render_section_home_page("interfaces")
            
            # Add button to view data
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📊 View Interfaces Data", key="view_interfaces_data", use_container_width=True, type="primary"):
                    st.session_state.show_interfaces_data = True
                    st.rerun()
    
    def render_extra_batch_connections_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Extra Batch Connections specific content."""
        # Check if user wants to view data
        if st.session_state.get('show_extra_batch_data', False):
            # Apply date filtering to ensure we show data for the selected date
            df = self.filter_data_by_selected_date(df)
            
            if df is None or df.empty:
                st.info("📋 No Extra Batch Connections data found. This could indicate:")
                st.markdown("""
                - No Excel files in the Extra Batch Connections folder
                - Files are empty or in an unsupported format
                - No data matching the selected date range
                
                **Next Steps:**
                - Check the Monitoring Data Files folder structure
                - Verify Excel files contain data
                - Try adjusting the date range filter
                """)
                
                if st.button("← Back to Extra Batch Connections Overview", key="back_to_extra_batch_home"):
                    st.session_state.show_extra_batch_data = False
                    st.rerun()
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
            
            # Data Table - Main Focus with highlighting
            st.header("📋 Extra Connections Created Data")
            self.render_extra_batch_connections_table(filtered_df, "Extra Connections Created Data")
            
            # Key Metrics - Below Data Table
            st.header(f"📊 Key Metrics")
            self.metrics_component.auto_metrics(filtered_df)
            
            if st.button("← Back to Extra Batch Connections Overview", key="back_to_extra_batch_home2"):
                st.session_state.show_extra_batch_data = False
                st.rerun()
        else:
            # Show section home page with information
            self.render_section_home_page("extra_batch_connections")
            
            # Add button to view data
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📊 View Extra Batch Connections Data", key="view_extra_batch_data", use_container_width=True, type="primary"):
                    st.session_state.show_extra_batch_data = True
                    st.rerun()

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
            
            # Display with custom styling
            display_clean_dataframe(styled_df, use_container_width=True, hide_index=True)
            
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
            
            # Use regular table
            display_clean_dataframe(display_df, use_container_width=True, hide_index=True)
    
    def render_hung_threads_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Hung Threads specific content."""
        # Check if user wants to view data
        if st.session_state.get('show_hung_threads_data', False):
            # Apply date filtering to ensure we show data for the selected date
            df = self.filter_data_by_selected_date(df)
            
            if df is None or df.empty:
                st.info("📋 No Hung Threads data found. This could indicate:")
                st.markdown("""
                - No Excel files in the Hung Threads folder
                - Files are empty or in an unsupported format
                - No data matching the selected date range
                - System is healthy with no hung threads (which is good!)
                
                **Next Steps:**
                - Check the Monitoring Data Files folder structure
                - Verify Excel files contain data
                - Try adjusting the date range filter
                """)
                
                if st.button("← Back to Hung Threads Overview", key="back_to_hung_threads_home"):
                    st.session_state.show_hung_threads_data = False
                    st.rerun()
                return
            
            self.render_generic_section_content(df, selected_period, "Hung Threads", "🧵")
            
            if st.button("← Back to Hung Threads Overview", key="back_to_hung_threads_home2"):
                st.session_state.show_hung_threads_data = False
                st.rerun()
        else:
            # Show section home page with information
            self.render_section_home_page("hung_threads")
            
            # Add button to view data
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📊 View Hung Threads Data", key="view_hung_threads_data", use_container_width=True, type="primary"):
                    st.session_state.show_hung_threads_data = True
                    st.rerun()

    def render_data_warehouse_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Data Warehouse specific content."""
        if df is None or df.empty:
            st.markdown("## 🏢 Data Warehouse Overview")
            st.info("📋 No Data Warehouse data found. This could indicate:")
            st.markdown("""
            - No Excel files in the Data Warehouse folder
            - Files are empty or in an unsupported format
            - No data matching the selected date range
            
            **Next Steps:**
            - Check the Monitoring Data Files folder structure
            - Verify Excel files contain data
            - Try adjusting the date range filter
            """)
            return
        
        self.render_generic_section_content(df, selected_period, "Data Warehouse", "🏢")

    def render_consolidated_inquiry_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Consolidated Inquiry specific content."""
        self.render_generic_section_content(df, selected_period, "Consolidated Inquiry", "🔍")

    def render_other_critical_processes_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Other Critical Processes section overview."""
        st.markdown("## 🔗 Other Critical Processes Overview")
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

    def render_daily_exceptions_content(self, section: str, title: str, icon: str) -> None:
        """Render Daily Exceptions content with summary view and clickable date buttons."""
        
        # Check if a specific date is selected for detailed view
        selected_date_key = f"selected_date_{section}"
        selected_date = st.session_state.get(selected_date_key)
        
        if selected_date:
            # Show detailed data for the selected date
            st.subheader(f"{icon} {title} - {selected_date}")
            
            # Back button
            if st.button("← Back to Summary", key=f"back_{section}"):
                st.session_state[selected_date_key] = None
                st.rerun()
            
            # Load and display detailed data for this date
            try:
                from pathlib import Path
                
                section_to_sheet_map = {
                    "prd_online_exceptions": "PRD Online",
                    "prd_batch_exceptions": "PRD Batch", 
                    "prd_batch_runtime": "PRD Batch Runtime",
                    "uat_online_exceptions": "UAT Online",
                    "uat_batch_exceptions": "UAT Batch",
                    "uat_batch_runtime": "UAT Batch Runtime"
                }
                
                daily_exceptions_path = Path("Monitoring Data Files") / "Daily Exceptions"
                excel_file = daily_exceptions_path / f"{selected_date}.xlsx"
                sheet_name = section_to_sheet_map.get(section)
                
                if excel_file.exists() and sheet_name:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    if not df.empty:
                        # Format the data
                        df_formatted = format_dataframe_dates(df)
                        df_sorted = sort_dataframe_by_date(df_formatted, ascending=False)
                        
                        st.success(f"📊 Showing {len(df_sorted)} records from {selected_date}")
                        
                        # Show column information
                        st.markdown(f"**📋 Data Columns:** {', '.join(df_sorted.columns)}")
                        
                        # Display the data with proper table formatting
                        st.markdown("### 📊 Detailed Data")
                        display_height = min(600, max(200, len(df_sorted) * 35 + 100))
                        display_clean_dataframe(
                            df_sorted, 
                            width="stretch", 
                            height=display_height, 
                            hide_index=True
                        )
                    else:
                        st.info("📋 No data available for this date.")
                else:
                    st.error(f"❌ Could not load data for {selected_date}")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        
        else:
            # Show summary view with clickable date buttons
            st.subheader(f"{icon} {title}")
            
            # Get summary data
            summary_df = get_daily_exceptions_summary(section)
            
            if not summary_df.empty:
                st.success(f"📊 Found {len(summary_df)} available dates")
                
                # Get the count column name
                count_columns = [col for col in summary_df.columns if col != "Date"]
                count_column_name = count_columns[0] if count_columns else "Count"
                
                # Display the table with integrated clickable buttons
                st.markdown("### 📋 Available Dates and Counts")
                st.markdown("*Click on any date to view detailed data*")
                
                # Create table headers
                header_col1, header_col2 = st.columns([1, 1])
                with header_col1:
                    st.markdown("**Date**")
                with header_col2:
                    st.markdown(f"**{count_column_name}**")
                
                # Add separator line
                st.markdown("---")
                
                # Create table rows with clickable date buttons integrated
                for index, row in summary_df.iterrows():
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Clickable date button that looks like a table cell
                        if st.button(
                            f"📅 {row['Date']}", 
                            key=f"date_btn_{section}_{index}",
                            use_container_width=True,
                            type="secondary"
                        ):
                            st.session_state[selected_date_key] = row['Date']
                            st.rerun()
                    
                    with col2:
                        # Count value in a formatted container to match button style
                        count_value = row[count_column_name] if count_columns else 0
                        st.markdown(
                            f'<div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 0.25rem; text-align: center; margin: 2px 0; border: 1px solid #dee2e6; font-weight: 500;">{count_value}</div>', 
                            unsafe_allow_html=True
                        )
            else:
                st.info("📋 No Daily Exceptions data found in the Monitoring Data Files/Daily Exceptions folder.")

    def render_prd_online_exceptions_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render PRD Online Exceptions specific content."""
        self.render_daily_exceptions_content("prd_online_exceptions", "PRD Online Exceptions", "🌐")
    
    def render_prd_batch_exceptions_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render PRD Batch Exceptions specific content."""
        self.render_daily_exceptions_content("prd_batch_exceptions", "PRD Batch Exceptions", "💻")
    
    def render_prd_batch_runtime_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render PRD Batch Runtime specific content."""
        self.render_daily_exceptions_content("prd_batch_runtime", "PRD Batch Runtime", "⏱️")
    
    def render_uat_online_exceptions_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render UAT Online Exceptions specific content."""
        self.render_daily_exceptions_content("uat_online_exceptions", "UAT Online Exceptions", "🧪")
    
    def render_uat_batch_exceptions_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render UAT Batch Exceptions specific content."""
        self.render_daily_exceptions_content("uat_batch_exceptions", "UAT Batch Exceptions", "🔬")
    
    def render_uat_batch_runtime_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render UAT Batch Runtime specific content."""
        self.render_daily_exceptions_content("uat_batch_runtime", "UAT Batch Runtime", "⏰")
    
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
        

        

        # Display the main table
        display_clean_dataframe(display_df, use_container_width=True, hide_index=True)
        

    
    def render_error_counts_table(self, df: pd.DataFrame, title: str) -> None:
        """Render Error Counts table with calculated percentage columns."""
        import pandas as pd
        
        # Additional aggressive empty row removal for table display
        df = remove_empty_rows(df)
        
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
        
        # Display the table
        display_clean_dataframe(display_df, use_container_width=True, hide_index=True)
        

        
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
                # Exclude columns that are clearly not dates (like processing times)
                if 'processing' in col_lower or 'load' in col_lower or 'staging' in col_lower:
                    continue
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
                        # Skip columns that are all NaN
                        if df[col].isna().all():
                            continue
                        
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        
                        # Skip columns with NaN min/max or identical values
                        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                            continue
                            
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
        # Apply date filtering to ensure we show data for the selected date
        df = self.filter_data_by_selected_date(df)
        
        # Display the selected date for clarity
        selected_date = self.get_selected_date()
        st.markdown(f"**📅 Data as of:** {selected_date.strftime('%Y-%m-%d')}")
        st.markdown("---")
        
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
            
            # Get selections from session state 
            selected_section = st.session_state.get('selected_section')
            selected_subsection = st.session_state.get('selected_subsection')
            
            # If no section is selected, default to summary (home page)
            if not selected_section:
                selected_section = 'summary'
                st.session_state.selected_section = 'summary'
            
            # Add breadcrumb navigation at the top of main content (only if not on summary page)
            if selected_section != 'summary':
                self.render_breadcrumb_navigation(selected_section, selected_subsection)
            
            # Handle Summary section as home page (no data loading needed)
            if selected_section == 'summary':
                self.render_summary_home_page()
                return
            
            # Check if we should load data (either has subsection OR is a section that loads data directly)
            sections_with_direct_data = ['prd_online_exceptions', 'prd_batch_exceptions', 'prd_batch_runtime',
                                       'uat_online_exceptions', 'uat_batch_exceptions', 'uat_batch_runtime',
                                       'uat_batch_status', 'prd_batch_status']  # Sections that load data without subsections
            
            # Sections that need subsection selection first
            sections_requiring_subsections = ['batch_status', 'daily_exceptions', 'benefit_issuance', 'miscellaneous_bridges', 'correspondence_tango']
            
            # Sections with single subsections - auto-select the subsection
            single_subsection_mappings = {
                'error_counts': 'Daily 100 Error Counts',
                'online_ora_errors': 'Online',
                'batch_ora_errors': 'Batch',
                'user_impact': 'Daily User Impact Status'
            }
            # Check if we need to load data for sections that have data viewing enabled
            data_viewing_sections = ['mass_update', 'interfaces', 'hung_threads', 'extra_batch_connections']
            section_wants_data = any([
                st.session_state.get('show_mass_update_data', False) and selected_section == 'mass_update',
                st.session_state.get('show_interfaces_data', False) and selected_section == 'interfaces', 
                st.session_state.get('show_hung_threads_data', False) and selected_section == 'hung_threads',
                st.session_state.get('show_extra_batch_data', False) and selected_section == 'extra_batch_connections'
            ])
            
            should_load_data = selected_subsection or (selected_section in sections_with_direct_data) or section_wants_data
            
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
                            self.render_main_content_with_data(df, {})
                        elif df is not None and df.empty:
                            st.warning("The loaded data is empty. Please check your Excel file.")
                        else:
                            st.error("Failed to load the Excel file. Please check the file format.")
                    else:
                        # No file found - call render function with empty dataframe for graceful handling
                        import pandas as pd
                        empty_df = pd.DataFrame()
                        self.render_main_content_with_data(empty_df, {})
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                # Handle sections that need subsection selection
                if selected_section in sections_requiring_subsections:
                    # Show subsection selection page
                    self.render_subsection_selection_page(selected_section)
                elif selected_section in single_subsection_mappings:
                    # Auto-select the single subsection and load data
                    auto_subsection = single_subsection_mappings[selected_section]
                    st.session_state.selected_subsection = auto_subsection
                    
                    try:
                        # Auto-detect and load Excel files with the selected subsection
                        file_config = self.auto_load_excel_file(
                            section=selected_section,
                            period=st.session_state.get('selected_period', 'daily'),
                            subsection=auto_subsection
                        )
                        
                        if file_config:
                            # Load data automatically
                            df = self.load_data(file_config)
                            
                            if df is not None and not df.empty:
                                # Render main content with the loaded data
                                self.render_main_content_with_data(df, {})
                            elif df is not None and df.empty:
                                st.warning(f"The loaded data for {auto_subsection} is empty. Please check your Excel file.")
                            else:
                                st.error(f"Failed to load the Excel file for {auto_subsection}. Please check the file format.")
                        else:
                            # No file found - show message
                            st.warning(f"No data file found for {auto_subsection}. Please ensure the Excel file exists in the monitoring data folder.")
                    except Exception as e:
                        st.error(f"Error loading data for {auto_subsection}: {str(e)}")
                else:
                    # For any remaining sections that don't have specific handling
                    st.info("This section is under development. Please navigate back to the home page and select another dashboard section.")
            
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

    def render_ora_errors_table(self, df: pd.DataFrame, title: str) -> None:
        """Render ORA Errors table with appropriate formatting."""
        import pandas as pd
        
        # Additional aggressive empty row removal for table display
        df = remove_empty_rows(df)
        
        # Sort by date columns (latest to oldest)
        df = sort_dataframe_by_date(df, ascending=False)
        
        # Apply date formatting to the dataframe
        df = format_dataframe_dates(df)
        
        # Display table title
        st.subheader(title)
        
        # Create a copy of the dataframe for display
        display_df = df.copy()
        
        if display_df.empty:
            st.info("No ORA errors data available for the selected date range.")
            return
        
        # Display the dataframe with clean formatting
        display_clean_dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Show total record count
        st.caption(f"📊 Total Records: {len(display_df)}")

    def render_ora_errors_metrics(self, df: pd.DataFrame) -> None:
        """Render ORA errors metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        # Count total errors
        total_errors = len(df)
        
        # Look for error severity if available
        severity_high = 0
        if not df.empty:
            # Check if there are severity or error type columns
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'severity' in col_lower or 'level' in col_lower or 'priority' in col_lower:
                    try:
                        severity_high = len(df[df[col].astype(str).str.contains('high|critical|error', case=False, na=False)])
                    except:
                        pass
                    break
        
        with col1:
            delta_text = f"+{total_errors}" if total_errors > 0 else "0"
            st.metric("Total ORA Errors", total_errors, delta=delta_text if total_errors > 0 else None)
        with col2:
            status = "Critical" if total_errors > 50 else "Warning" if total_errors > 10 else "Good"
            st.metric("Error Status", status)
        with col3:
            st.metric("High Severity", severity_high if severity_high > 0 else "None")
        with col4:
            health = "Action Needed" if total_errors > 20 else "Monitoring"
            st.metric("System Health", health)

    def get_previous_working_day(self) -> datetime:
        """Calculate the previous working day (skipping weekends)."""
        today = datetime.now()
        
        # Get previous day
        previous_day = today - timedelta(days=1)
        
        # If previous day is Saturday (5) or Sunday (6), go back further
        while previous_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
            previous_day = previous_day - timedelta(days=1)
        
        return previous_day

    def format_batch_timestamp(self, value):
        """Format timestamp values to DD-MON-YYYY format for batch status."""
        if pd.isna(value) or value is None or str(value).strip() == '' or str(value) == 'nan':
            return value
            
        try:
            value_str = str(value).strip()
            
            # Direct regex match for YYYY-MM-DD HH:MM:SS
            timestamp_match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2}) \d{1,2}:\d{1,2}:\d{1,2}', value_str)
            if timestamp_match:
                year, month, day = timestamp_match.groups()
                month_names = ['', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                month_abbr = month_names[int(month)]
                return f"{int(day):02d}-{month_abbr}-{year}"
            
            # If it's already in DD-MON-YYYY format, keep it
            if re.match(r'\d{1,2}-[A-Z]{3}-\d{4}', value_str):
                return value_str
                
            # Try pandas datetime conversion as fallback
            if hasattr(value, 'strftime'):
                return value.strftime('%d-%b-%Y').upper()
                
            return value_str
        except:
            return str(value)

    def filter_batch_status_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter batch status data to show only records for the selected date."""
        if df is None or df.empty:
            return df
        
        # Use the selected date from the main page instead of hardcoded previous working day
        selected_date = self.get_selected_date()
        
        # Look for date columns in the dataframe
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'day', 'time', 'created', 'updated']):
                date_columns.append(col)
        
        if not date_columns:
            # If no date columns found, return the data as is
            return df
        
        # Use the first date column found
        date_col = date_columns[0]
        
        try:
            # Convert to datetime for comparison
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            
            # Filter for the selected date
            from datetime import datetime
            selected_date_start = datetime.combine(selected_date, datetime.min.time())
            selected_date_end = datetime.combine(selected_date, datetime.max.time())
            
            filtered_df = df_copy[
                (df_copy[date_col] >= selected_date_start) & 
                (df_copy[date_col] <= selected_date_end)
            ]
            
            return filtered_df.dropna(subset=[date_col])
        
        except Exception:
            # If date filtering fails, return original data
            return df

    def render_batch_status_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Batch Status main landing page."""
        self.render_section_home_page("batch_status")

    def render_uat_batch_status_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render UAT Batch Status specific content."""
        self.render_batch_status_environment("uat_batch_status", "UAT Batch Status", "🧪", df)
    
    def render_prd_batch_status_content(self, df: pd.DataFrame, selected_period: str) -> None:
        """Render Production Batch Status specific content."""  
        self.render_batch_status_environment("prd_batch_status", "Production Batch Status", "🏭", df)

    def render_batch_status_environment(self, section: str, title: str, icon: str, df: pd.DataFrame) -> None:
        """Render batch status content for a specific environment (UAT or Production)."""
        
        # Display the selected date consistently
        selected_date = self.get_selected_date()
        st.markdown(f"**📅 Data as of:** {selected_date.strftime('%Y-%m-%d')}")
        st.markdown("---")
        
        # Check for empty data and show consistent message
        if self.check_and_display_no_data_message(df):
            return
        
        # Filter data for the selected date and display immediately
        if df is not None and not df.empty:
            filtered_df = self.filter_batch_status_data(df)
            
            if filtered_df.empty:
                # Summary line for no failed jobs
                st.markdown("**📊 Failed Jobs: 0**")
                
                # No failed batch jobs message
                st.success("✅ **There are no failed batch jobs** for the previous working day.")
            else:
                # Summary line for failed jobs count
                st.markdown(f"**📊 Failed Jobs: {len(filtered_df)}**")
                
                # Display the data table
                st.markdown("## 📋 Batch Job Details")
                
                # Format and display the dataframe with proper timestamp conversion
                df_display = filtered_df.copy()
                
                # Convert ALL columns to proper display format
                for col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: self.format_batch_timestamp(x))
                
                # Don't use sort_dataframe_by_date as it converts dates back to timestamps
                # Instead, just reverse the order if needed
                df_display = df_display.iloc[::-1]
                
                display_clean_dataframe(
                    df_display,
                    use_container_width=True,
                    height=400
                )
                
                # Additional analysis
                if len(df_display.columns) > 0:
                    st.markdown("## 🔍 Quick Analysis")
                    
                    # Show analysis in columns
                    analysis_cols = st.columns(2)
                    with analysis_cols[0]:
                        st.markdown("**Data Summary:**")
                        st.write(f"- Total failed jobs: {len(filtered_df)}")
                        st.write(f"- Data columns: {len(filtered_df.columns)}")
                        st.write(f"- Environment: {title.split()[0]}")
                        
                    with analysis_cols[1]:
                        st.markdown("**Recommended Actions:**")
                        if len(filtered_df) > 10:
                            st.write("🔴 High number of failures - Immediate attention required")
                        elif len(filtered_df) > 5:
                            st.write("🟡 Moderate failures - Investigation recommended")
                        else:
                            st.write("🟢 Low failure count - Monitor trends")
        else:
            # No data available
            selected_date = self.get_selected_date()
            formatted_date = selected_date.strftime('%Y-%m-%d')
            st.warning(f"⚠️ No batch status data available for {formatted_date}")
            
            # Show basic metrics even with no data
            st.markdown("## 📊 Environment Status")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Status", "No Data")
            with col2:
                st.metric("Environment", title.split()[0])
            with col3:
                st.metric("Date Checked", formatted_date)
            with col4:
                st.metric("Status", "Unknown")


def main():
    """Main application entry point."""
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()


