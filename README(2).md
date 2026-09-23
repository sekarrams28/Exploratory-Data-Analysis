# Exploratory Data Analysis (EDA) Project 🍷📊

A beginner-friendly Exploratory Data Analysis project that analyzes the Wine dataset to uncover patterns, distributions, correlations, possible outliers, and differences between wine classes.

## Project Overview

This project demonstrates:

1. Loading a CSV dataset
2. Inspecting rows, columns, and data types
3. Checking missing values
4. Checking duplicate rows
5. Generating statistical summaries
6. Analyzing class distribution
7. Creating histograms
8. Creating boxplots
9. Calculating correlations
10. Creating a correlation heatmap
11. Creating scatter plots
12. Comparing group-wise averages
13. Generating automatic insights
14. Saving charts and analysis results

## Dataset

The included `wine_dataset.csv` is a CSV copy of the Wine dataset provided by Scikit-learn.

- Rows: 178
- Numerical features: 13
- Target classes: 3

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
EDA-Project/
│
├── eda_project.py
├── wine_dataset.csv
├── README.md
├── .gitignore
│
└── eda_outputs/
```

The `eda_outputs` folder is automatically created when the program runs.

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

```bash
python eda_project.py
```

## Analysis Performed

### Statistical Summary

Uses `df.describe()` to calculate count, mean, standard deviation, minimum, quartiles, and maximum.

### Data Quality

Checks missing values and duplicate rows.

### Visualizations

Creates class distribution, histograms, boxplots, a correlation heatmap, a scatter plot, and a top-correlated-features chart.

### Correlation

Correlation values range from -1 to +1. Correlation indicates association, not causation.

## Generated Output

After running the program, `eda_outputs` contains CSV reports and PNG charts, including:

- `statistical_summary.csv`
- `correlation_matrix.csv`
- `group_wise_means.csv`
- `analyzed_wine_dataset.csv`
- `class_distribution.png`
- `feature_distributions.png`
- `boxplots.png`
- `correlation_heatmap.png`
- `feature_scatter_plot.png`
- `top_correlated_features.png`

## Expected Outcome

You will gain practical experience in:

- Exploratory Data Analysis
- Statistical summaries
- Data quality checking
- Data visualization
- Correlation analysis
- Outlier identification
- Feature comparison
- Insight generation

## Future Improvements

- Streamlit dashboard
- User CSV upload
- Interactive Plotly charts
- Automated PDF report
- Machine learning prediction
- PCA visualization
- Automated data cleaning

## License

This project is created for educational and learning purposes.
