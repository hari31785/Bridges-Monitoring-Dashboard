"""Configuration management for the monitoring dashboard."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the dashboard application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_source_type(self) -> str:
        """Get the current data source type."""
        return self.get('data_source.type', 'excel')
    
    def get_excel_config(self) -> Dict[str, Any]:
        """Get Excel configuration."""
        return self.get('excel', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.get('dashboard', {})
    
    def get_chart_config(self) -> Dict[str, Any]:
        """Get chart configuration."""
        return self.get('charts', {})
    
    def update_data_source(self, source_type: str) -> None:
        """Update the data source type and save configuration.
        
        Args:
            source_type: New data source type ('excel' or 'database')
        """
        if source_type not in ['excel', 'database']:
            raise ValueError("Data source type must be 'excel' or 'database'")
        
        self._config['data_source']['type'] = source_type
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)


# Global configuration instance
config = Config()
