"""Data loading module for Excel files and databases."""

import pandas as pd
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .config import config


class DataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data and return as pandas DataFrame."""
        pass
    
    @abstractmethod
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources (tables, sheets, etc.)."""
        pass


class ExcelDataLoader(DataLoader):
    """Data loader for Excel files."""
    
    def __init__(self, file_path: Optional[str] = None):
        """Initialize Excel data loader.
        
        Args:
            file_path: Path to Excel file. If None, uses config default.
        """
        if file_path is None:
            file_path = config.get('excel.file_path')
        
        self.file_path = Path(file_path) if file_path else None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from Excel file.
        
        Args:
            sheet_name: Name of sheet to load. If None, loads first sheet.
            **kwargs: Additional parameters for pd.read_excel()
            
        Returns:
            pandas DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If sheet doesn't exist or other pandas errors
        """
        if not self.file_path or not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")
        
        try:
            # Determine which sheet to load
            target_sheet = sheet_name or config.get('excel.sheet_name')
            
            # If no specific sheet is requested, get the first sheet
            if target_sheet is None:
                excel_file = pd.ExcelFile(self.file_path)
                if excel_file.sheet_names:
                    target_sheet = excel_file.sheet_names[0]
                else:
                    raise ValueError("No sheets found in Excel file")
            
            # Set parameters
            params = {
                'header': config.get('excel.header_row', 0),
                'sheet_name': target_sheet,
                **kwargs
            }
            
            self.logger.info(f"Loading Excel data from {self.file_path}, sheet: {target_sheet}")
            df = pd.read_excel(self.file_path, **params)
            
            # Handle case where pandas returns a dict (shouldn't happen with specific sheet_name, but just in case)
            if isinstance(df, dict):
                # If it's a dict, take the first sheet's data
                df = list(df.values())[0] if df else pd.DataFrame()
            
            # Clean up data types for better Streamlit compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    col_lower = col.lower()
                    # Skip obvious non-date columns with more specific patterns
                    non_date_keywords = ['error', 'count', 'total', 'number', 'qty', 'quantity', 'amount', 'value', 'rate', 'percent', 'id', 'timeout', 'session', 'connection', 'batch', 'thread']
                    if any(keyword in col_lower for keyword in non_date_keywords):
                        # Convert to string to avoid Arrow serialization issues
                        df[col] = df[col].astype(str)
                    elif any(keyword in col_lower for keyword in ['date', 'time', 'day', 'month', 'year']):
                        # Only try datetime conversion for columns that clearly indicate dates
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            df[col] = df[col].astype(str)
                    else:
                        # Convert other object columns to string to avoid Arrow serialization issues
                        df[col] = df[col].astype(str)
            
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Excel data: {e}")
            raise ValueError(f"Error loading Excel file: {e}")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available sheet names.
        
        Returns:
            List of sheet names in the Excel file
        """
        if not self.file_path or not self.file_path.exists():
            return []
        
        try:
            excel_file = pd.ExcelFile(self.file_path)
            return excel_file.sheet_names
        except Exception as e:
            self.logger.error(f"Error reading sheet names: {e}")
            return []
    
    def set_file_path(self, file_path: str) -> None:
        """Set the Excel file path.
        
        Args:
            file_path: Path to Excel file
        """
        self.file_path = Path(file_path)


class DatabaseDataLoader(DataLoader):
    """Data loader for database connections."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database data loader.
        
        Args:
            connection_string: Database connection string. If None, uses config.
        """
        self.connection_string = connection_string or self._build_connection_string()
        self.engine = None
        self.logger = logging.getLogger(__name__)
        
        if self.connection_string:
            self._create_engine()
    
    def _build_connection_string(self) -> str:
        """Build connection string from configuration."""
        db_config = config.get_database_config()
        db_type = db_config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            return db_config.get('connection_string', 'sqlite:///data/monitoring.db')
        
        # Build connection string for other database types
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database', '')
        username = db_config.get('username', '')
        password = db_config.get('password', '')
        
        if db_type == 'postgresql':
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'mysql':
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'sqlserver':
            return f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        return ""
    
    def _create_engine(self) -> None:
        """Create SQLAlchemy engine."""
        try:
            self.engine = create_engine(self.connection_string)
            self.logger.info("Database engine created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database engine: {e}")
            self.engine = None
    
    def load_data(self, query: Optional[str] = None, table_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from database.
        
        Args:
            query: SQL query to execute. Takes precedence over table_name.
            table_name: Name of table to load (SELECT * FROM table_name)
            **kwargs: Additional parameters for pd.read_sql()
            
        Returns:
            pandas DataFrame with loaded data
            
        Raises:
            ValueError: If no query or table_name provided, or database connection issues
        """
        if not self.engine:
            raise ValueError("Database engine not initialized")
        
        if not query and not table_name:
            raise ValueError("Either query or table_name must be provided")
        
        # Use query if provided, otherwise select all from table
        sql_query = query if query else f"SELECT * FROM {table_name}"
        
        try:
            self.logger.info(f"Executing query: {sql_query[:100]}...")
            df = pd.read_sql(sql_query, self.engine, **kwargs)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error: {e}")
            raise ValueError(f"Database query failed: {e}")
        except Exception as e:
            self.logger.error(f"Error loading database data: {e}")
            raise ValueError(f"Error loading data: {e}")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available tables.
        
        Returns:
            List of table names in the database
        """
        if not self.engine:
            return []
        
        try:
            # Get table names using SQLAlchemy
            with self.engine.connect() as conn:
                # Query for table names (works for most SQL databases)
                if 'sqlite' in self.connection_string:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                elif 'postgresql' in self.connection_string:
                    result = conn.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'"))
                elif 'mysql' in self.connection_string:
                    result = conn.execute(text("SHOW TABLES"))
                else:
                    # Generic approach
                    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"))
                
                tables = [row[0] for row in result]
                return tables
                
        except Exception as e:
            self.logger.error(f"Error getting table names: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def create_loader(source_type: Optional[str] = None) -> DataLoader:
        """Create appropriate data loader based on configuration or type.
        
        Args:
            source_type: Type of data source ('excel' or 'database'). 
                        If None, uses configuration.
            
        Returns:
            DataLoader instance
            
        Raises:
            ValueError: If unsupported source type
        """
        if source_type is None:
            source_type = config.get_data_source_type()
        
        if source_type == 'excel':
            return ExcelDataLoader()
        elif source_type == 'database':
            return DatabaseDataLoader()
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")


# Convenience function
def load_data(source_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Load data using the configured or specified data source.
    
    Args:
        source_type: Type of data source ('excel' or 'database')
        **kwargs: Additional parameters passed to the loader
        
    Returns:
        pandas DataFrame with loaded data
    """
    loader = DataLoaderFactory.create_loader(source_type)
    return loader.load_data(**kwargs)


def get_available_sources(source_type: Optional[str] = None) -> List[str]:
    """Get available data sources (sheets/tables) for the configured source type.
    
    Args:
        source_type: Type of data source ('excel' or 'database')
        
    Returns:
        List of available source names
    """
    loader = DataLoaderFactory.create_loader(source_type)
    return loader.get_available_sources()
