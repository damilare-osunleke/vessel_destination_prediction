import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, f1_score

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def calculate_metrics(actual, predicted, averaging_method):
    """
    Calculate metrics and return them as a DataFrame row with the model name.

    Args:
    actual (array-like): True labels.
    predicted (array-like): Predicted labels.
    averaging_method (str): Method to average the precision and F1-score, e.g., 'micro', 'macro', 'weighted'.
    model_name (str): Name of the model, used to label the row.

    Returns:
    DataFrame: A DataFrame with one row of metrics including the model name.
    """
    metrics = {
        'Accuracy': accuracy_score(actual, predicted),
        'Precision': precision_score(actual, predicted, average=averaging_method, zero_division=0),
        'Recall': recall_score(actual, predicted, average=averaging_method, zero_division=0),
        'F1-Score': f1_score(actual, predicted, average=averaging_method),
        'Balanced Accuracy': balanced_accuracy_score(actual, predicted)
    }
    
    return metrics


def calculate_metrics_by_group(df_group, averaging_method):
    """
    Calculate and return various classification metrics for a given group of data.

    Args:
    df_group (pd.DataFrame): DataFrame containing 'actual_country' and 'predicted_country' columns.
    averaging_method (str): Method to average the precision, recall, and F1-score. e.g. 'micro', 'macro', 'weighted'.

    Returns:
    pd.Series: A Series containing the calculated metrics: accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(df_group['actual_country'], df_group['predicted_country'])
    balanced_accuracy = balanced_accuracy_score(df_group['actual_country'], df_group['predicted_country'])
    precision = precision_score(df_group['actual_country'], df_group['predicted_country'], average=averaging_method, zero_division=0)
    recall = recall_score(df_group['actual_country'], df_group['predicted_country'], average=averaging_method, zero_division=0)
    f1 = f1_score(df_group['actual_country'], df_group['predicted_country'], average = averaging_method, zero_division=0)
    return pd.Series({
        'accuracy': accuracy,
        'bal_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })



def plot_metrics(df, classifications, title):
    """
    Plots accuracy against the number of records for specified classifications.

    Args:
    df (DataFrame): DataFrame containing the metrics.
    classifications (list or str): List of classifications to plot or 'all' for plotting all classifications.
    title (str): The title of the plot.

    Returns:
    fig (Figure): The figure object of the plot.
    """
    sns.set(style="whitegrid") 
    plt.figure(figsize=(12, 6))
    if classifications == 'all':
        filtered_df = df
    else:
        filtered_df = df[df['classification'].isin(classifications)]

    ax = sns.scatterplot(x='count', y='accuracy', data=filtered_df, s=100, marker='o')
    ax.set_xlabel("Number of Trips", fontsize=20, labelpad=10)
    ax.set_ylabel("Accuracy", fontsize=20, labelpad=10)
    plt.title(title, fontsize=24, pad=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    return plt.gcf() 



def plot_accuracy_distribution(df, metric, title, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], figure_size=(10, 6)):
    """
    Plot cumulative accuracy distribution as percentages and display values on the bars with custom font sizes.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the vessel data.
    - metric (str): The metric column name to use for plotting.
    - title (str): Title of the plot.
    - thresholds (list of float): The list of accuracy thresholds to evaluate.
    - figure_size (tuple): The dimensions of the figure to be plotted.

    Returns:
    - fig (Figure): The matplotlib figure object for further manipulation or saving.
    """

    
    df_unique = df.drop_duplicates()
    total_vessels = len(df_unique)
    df_sorted = df_unique.sort_values(by=metric, ascending=True)
    cumulative_percentages = {threshold: (df_sorted[metric] >= threshold).sum() / total_vessels * 100 for threshold in thresholds}
    cumulative_series = pd.Series(cumulative_percentages, name="Percentage of Vessels")

    # Plotting
    fig, ax = plt.subplots(figsize=figure_size)
    bars = cumulative_series.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(title, fontsize=18)  
    ax.set_xlabel('Minimum Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Vessels Meeting Threshold (%)', fontsize=14)  
    ax.tick_params(axis='both', labelsize=14)

    ax.grid(True)
    ax.set_xticklabels([f'>={x}' for x in thresholds], rotation=45)
    ax.set_ylim(0, 100)

    # Add text labels above the bars
    for bar in bars.patches:
        ax.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='center',
                    size=12, xytext=(0, 8),
                    textcoords='offset points') 

    plt.tight_layout()
    return fig
