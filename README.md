# Monitoring Dashboard

A comprehensive monitoring dashboard built with Streamlit for tracking various operational metrics and Excel data analysis.

## Features

- 📊 **Interactive Data Visualization**: Multiple chart types with dynamic filtering
- 🎯 **Advanced Conditional Formatting**: 
  - Red highlighting for variance "Above 10%" and connections ≥7
  - Bold red rows for file upload differences > 0
- 💰 **Smart Formatting**: 
  - Currency fields display as $X,XXX.XX
  - Percentage columns with "%" symbol show as percentages (0.04 → 4%)
  - Universal DD-MON-YYYY date formatting
- � **Dynamic Data Loading**: 
  - Tango File Upload Status loads date-specific sheets
  - Expandable navigation with real-time data
- � **Advanced Analytics**: 
  - Correlation analysis, missing data analysis
  - Interactive charts with multiple visualization options

## Dashboard Sections

- **Benefit Issuance**: FAP, FIP, SDA daily tracking with variance analysis
- **Correspondence-Tango**: Dynamic file upload monitoring with clickable dates
- **100 Error Counts**: Session timeout and error tracking
- **User Impact**: Daily user impact status monitoring
- **Extra Batch Connections**: Connection monitoring with alerts

## Project Structure

```
Monitoring Dashboard/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── config/
│   └── config.yaml        # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Data loading modules
│   └── components.py      # Dashboard components
├── data/
│   └── sample_data.xlsx   # Sample Excel file
└── README.md              # This file
```

## Installation

1. **Clone or download** this project to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your data source** (optional):
   - Edit `config/config.yaml` to set your default data source
   - For databases, update the connection parameters

## Quick Start

### Option 1: Excel Files
1. Run the dashboard:
   ```bash
   streamlit run main.py
   ```
2. Upload an Excel file using the sidebar file uploader
3. Select the sheet you want to analyze
4. Explore your data with interactive charts and filters!

### Option 2: Database Connection
1. Update `config/config.yaml` with your database credentials:
   ```yaml
   data_source:
     type: "database"
   
   database:
     type: "postgresql"  # or "mysql", "sqlite", "sqlserver"
     host: "your-host"
     port: 5432
     database: "your-database"
     username: "your-username"
     password: "your-password"
   ```

2. Run the dashboard:
   ```bash
   streamlit run main.py
   ```

3. Select "database" from the data source dropdown
4. Choose your table and start analyzing!

## Configuration

The `config/config.yaml` file allows you to customize:

- **Data Source Settings**: Default file paths, database connections
- **Dashboard Appearance**: Title, theme, layout
- **Chart Settings**: Colors, default height, styling

### Database Support

Supported database types:
- ✅ **PostgreSQL** (`postgresql`)
- ✅ **MySQL** (`mysql`) 
- ✅ **SQLite** (`sqlite`)
- ✅ **SQL Server** (`sqlserver`)

### Excel Support

Supported Excel formats:
- ✅ `.xlsx` files
- ✅ `.xls` files
- ✅ Multiple sheets
- ✅ Custom header rows

## Usage Examples

### Loading Excel Data
```python
from src.data_loader import ExcelDataLoader

loader = ExcelDataLoader("path/to/your/file.xlsx")
df = loader.load_data(sheet_name="Sheet1")
```

### Loading Database Data
```python
from src.data_loader import DatabaseDataLoader

loader = DatabaseDataLoader("postgresql://user:pass@localhost/db")
df = loader.load_data(table_name="your_table")
# Or with custom SQL:
df = loader.load_data(query="SELECT * FROM your_table WHERE date > '2023-01-01'")
```

### Creating Charts
```python
from src.components import ChartComponent

chart = ChartComponent()
fig = chart.line_chart(df, x="date", y=["sales", "profit"], title="Sales Over Time")
```

## Dashboard Features

### 📊 Key Metrics
- Automatic calculation of totals, averages, and counts
- Customizable metric displays
- Support for delta indicators

### 🔍 Interactive Filters
- **Numeric columns**: Range sliders
- **Categorical columns**: Multi-select dropdowns
- **Date columns**: Date range pickers
- Real-time data filtering

### 📈 Chart Types
- **Line Charts**: Time series and trend analysis
- **Bar Charts**: Category comparisons
- **Scatter Plots**: Correlation analysis
- **Pie Charts**: Proportion analysis
- **Histograms**: Distribution analysis
- **Box Plots**: Statistical summaries

### 📋 Data Analysis
- **Data Table**: Sortable, searchable data views
- **Summary Statistics**: Automatic statistical summaries
- **Correlation Analysis**: Correlation matrix with heatmap
- **Missing Data Analysis**: Identify and visualize missing data
- **Top/Bottom Values**: Find extreme values

## Troubleshooting

### Common Issues

1. **"Module not found" error**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Database connection failed**:
   - Check your database credentials in `config.yaml`
   - Ensure the database server is running
   - Verify network connectivity

3. **Excel file not loading**:
   - Check file format (.xlsx or .xls)
   - Ensure file is not corrupted
   - Verify file path permissions

4. **Charts not displaying**:
   - Check that you have numeric columns for most chart types
   - Ensure data is not empty after filtering

### Performance Tips

- For large datasets (>100k rows), consider:
  - Using database queries with LIMIT clauses
  - Pre-filtering data before visualization
  - Using sampling for initial exploration

## Extending the Dashboard

### Adding New Chart Types
1. Add methods to `ChartComponent` class in `src/components.py`
2. Update the chart selection in `main.py`

### Adding New Data Sources
1. Create a new loader class inheriting from `DataLoader`
2. Implement `load_data()` and `get_available_sources()` methods
3. Update `DataLoaderFactory` to include your new loader

### Custom Metrics
1. Modify `MetricsComponent` in `src/components.py`
2. Add custom calculation functions
3. Update the metrics display logic

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

---

**Happy Dashboard Building! 📊✨**
