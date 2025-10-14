"""Reusable dashboard components for data visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .config import config


class ChartComponent:
    """Component for creating various chart types."""
    
    def __init__(self):
        """Initialize chart component with configuration."""
        self.chart_config = config.get_chart_config()
        self.color_palette = self.chart_config.get('color_palette', px.colors.qualitative.Set1)
        self.default_height = self.chart_config.get('default_height', 400)
    
    def line_chart(self, df: pd.DataFrame, x: str, y: Union[str, List[str]], 
                   title: str = "", **kwargs) -> go.Figure:
        """Create a line chart.
        
        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name(s) for y-axis
            title: Chart title
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        if isinstance(y, str):
            y = [y]
        
        fig = go.Figure()
        
        for i, y_col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title="Value",
            height=kwargs.get('height', self.default_height),
            template="plotly_white"
        )
        
        return fig
    
    def bar_chart(self, df: pd.DataFrame, x: str, y: str, 
                  title: str = "", color: Optional[str] = None, **kwargs) -> go.Figure:
        """Create a bar chart.
        
        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color grouping
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        if color:
            fig = px.bar(df, x=x, y=y, color=color, title=title,
                        color_discrete_sequence=self.color_palette)
        else:
            fig = px.bar(df, x=x, y=y, title=title,
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=kwargs.get('height', self.default_height),
            template="plotly_white"
        )
        
        return fig
    
    def scatter_plot(self, df: pd.DataFrame, x: str, y: str,
                     title: str = "", color: Optional[str] = None,
                     size: Optional[str] = None, **kwargs) -> go.Figure:
        """Create a scatter plot.
        
        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color grouping
            size: Column name for bubble size
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title,
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=kwargs.get('height', self.default_height),
            template="plotly_white"
        )
        
        return fig
    
    def pie_chart(self, df: pd.DataFrame, values: str, names: str,
                  title: str = "", **kwargs) -> go.Figure:
        """Create a pie chart.
        
        Args:
            df: DataFrame with data
            values: Column name for pie slice values
            names: Column name for pie slice names
            title: Chart title
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        fig = px.pie(df, values=values, names=names, title=title,
                    color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=kwargs.get('height', self.default_height)
        )
        
        return fig
    
    def histogram(self, df: pd.DataFrame, x: str, title: str = "",
                  bins: int = 30, **kwargs) -> go.Figure:
        """Create a histogram.
        
        Args:
            df: DataFrame with data
            x: Column name for histogram
            title: Chart title
            bins: Number of bins
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        fig = px.histogram(df, x=x, nbins=bins, title=title,
                          color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=kwargs.get('height', self.default_height),
            template="plotly_white"
        )
        
        return fig
    
    def box_plot(self, df: pd.DataFrame, y: str, x: Optional[str] = None,
                 title: str = "", **kwargs) -> go.Figure:
        """Create a box plot.
        
        Args:
            df: DataFrame with data
            y: Column name for y-axis (values)
            x: Column name for x-axis (categories)
            title: Chart title
            **kwargs: Additional plotly parameters
            
        Returns:
            Plotly figure object
        """
        fig = px.box(df, x=x, y=y, title=title,
                    color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=kwargs.get('height', self.default_height),
            template="plotly_white"
        )
        
        return fig


class TableComponent:
    """Component for displaying data tables."""
    
    @staticmethod
    def data_table(df: pd.DataFrame, title: str = "", 
                   max_rows: int = 1000, **kwargs) -> None:
        """Display a data table with Streamlit.
        
        Args:
            df: DataFrame to display
            title: Table title
            max_rows: Maximum number of rows to display
            **kwargs: Additional streamlit dataframe parameters
        """
        if title:
            st.subheader(title)
        
        # Import the date formatting function
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from main import sort_dataframe_by_date, format_dataframe_dates, format_percentage_columns
            # Sort by date columns (latest to oldest)
            df = sort_dataframe_by_date(df, ascending=False)
            # Apply date formatting
            df = format_dataframe_dates(df)
            
            # Apply currency formatting (Amt issued fields)
            for col in df.columns:
                col_lower = col.lower()
                if 'amt issued' in col_lower or 'amount issued' in col_lower or ('amt' in col_lower and 'issued' in col_lower):
                    # Convert to numeric if needed and format as currency
                    try:
                        # Convert to numeric, handling any string values
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Format as currency with 2 decimal places
                        df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    except (ValueError, TypeError):
                        # If conversion fails, try to format existing numeric values
                        df[col] = df[col].apply(
                            lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)
                        )
            
            # Apply percentage formatting
            df = format_percentage_columns(df)
        except ImportError:
            # If import fails, continue without formatting
            pass
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        
        # Add enhanced styling for table headers
        st.markdown("""
        <style>
        .stDataFrame table thead th {
            font-weight: bold !important;
            font-size: 16px !important;
            background-color: #f8f9fa !important;
            color: #333 !important;
            text-align: center !important;
            padding: 12px 8px !important;
            border-bottom: 2px solid #dee2e6 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the dataframe with dynamic height based on row count
        display_height = min(600, max(200, len(df) * 35 + 100))  # Dynamic height calculation
        
        # Display the dataframe
        if len(df) > max_rows:
            st.warning(f"Showing first {max_rows} rows of {len(df)} total rows")
            st.dataframe(df.head(max_rows), use_container_width=True, height=display_height, **kwargs)
        else:
            st.dataframe(df, use_container_width=True, height=display_height, **kwargs)
    
    @staticmethod
    def summary_stats(df: pd.DataFrame, title: str = "Summary Statistics") -> None:
        """Display summary statistics for the dataframe.
        
        Args:
            df: DataFrame to analyze
            title: Section title
        """
        if title:
            st.subheader(title)
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe())
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("**Categorical Columns:**")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                st.write(f"- **{col}**: {unique_count} unique values")
                if unique_count <= 10:
                    st.write(f"  Values: {', '.join(map(str, df[col].unique()))}")


class FilterComponent:
    """Component for data filtering controls."""
    
    @staticmethod
    def create_filters(df: pd.DataFrame, exclude_cols: List[str] = None) -> Dict[str, Any]:
        """Create interactive filters for dataframe columns.
        
        Args:
            df: DataFrame to create filters for
            exclude_cols: Columns to exclude from filtering
            
        Returns:
            Dictionary of filter values
        """
        if exclude_cols is None:
            exclude_cols = []
        
        filters = {}
        
        st.sidebar.header("Data Filters")
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if df[col].dtype in ['object', 'category']:
                # Categorical filter
                unique_values = df[col].unique()
                if len(unique_values) <= 50:  # Only show multiselect for reasonable number of options
                    selected_values = st.sidebar.multiselect(
                        f"Filter {col}",
                        options=unique_values,
                        default=unique_values
                    )
                    filters[col] = selected_values
            
            elif df[col].dtype in ['int64', 'float64']:
                # Numeric filter
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                if min_val != max_val:
                    selected_range = st.sidebar.slider(
                        f"Filter {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    filters[col] = selected_range
            
            elif df[col].dtype in ['datetime64[ns]']:
                # Date filter
                min_date = df[col].min().date()
                max_date = df[col].max().date()
                
                selected_date_range = st.sidebar.date_input(
                    f"Filter {col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                filters[col] = selected_date_range
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe.
        
        Args:
            df: Original dataframe
            filters: Dictionary of filter values from create_filters()
            
        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()
        
        for col, filter_value in filters.items():
            if col not in df.columns:
                continue
            
            if df[col].dtype in ['object', 'category']:
                if filter_value:  # If any values selected
                    filtered_df = filtered_df[filtered_df[col].isin(filter_value)]
            
            elif df[col].dtype in ['int64', 'float64']:
                if isinstance(filter_value, tuple) and len(filter_value) == 2:
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[
                        (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
                    ]
            
            elif df[col].dtype in ['datetime64[ns]']:
                if isinstance(filter_value, tuple) and len(filter_value) == 2:
                    start_date, end_date = filter_value
                    filtered_df = filtered_df[
                        (filtered_df[col].dt.date >= start_date) & 
                        (filtered_df[col].dt.date <= end_date)
                    ]
        
        return filtered_df


class MetricsComponent:
    """Component for displaying key metrics."""
    
    @staticmethod
    def display_metrics(df: pd.DataFrame, metric_configs: List[Dict[str, Any]]) -> None:
        """Display key metrics in columns.
        
        Args:
            df: DataFrame to calculate metrics from
            metric_configs: List of metric configurations
                           Each config should have: {'title': str, 'value': str/callable, 'delta': optional}
        """
        if not metric_configs:
            return
        
        cols = st.columns(len(metric_configs))
        
        for i, metric_config in enumerate(metric_configs):
            with cols[i]:
                title = metric_config['title']
                value_config = metric_config['value']
                
                # Calculate value
                if callable(value_config):
                    value = value_config(df)
                elif isinstance(value_config, str) and value_config in df.columns:
                    value = df[value_config].sum()  # Default to sum for numeric columns
                else:
                    value = value_config
                
                # Get delta if provided
                delta = metric_config.get('delta')
                
                st.metric(title, value, delta)
    
    @staticmethod
    def auto_metrics(df: pd.DataFrame) -> None:
        """Automatically generate basic metrics for the dataframe.
        
        Args:
            df: DataFrame to generate metrics for
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.info("No numeric columns found for metrics.")
            return
        
        # Create automatic metrics for first few numeric columns
        metric_configs = []
        for col in numeric_cols[:4]:  # Limit to 4 metrics
            metric_configs.append({
                'title': f"Total {col}",
                'value': lambda x, c=col: f"{x[c].sum():,.0f}" if x[c].sum() > 1000 else f"{x[c].sum():.2f}"
            })
        
        MetricsComponent.display_metrics(df, metric_configs)
