# %% Import Modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns

from data_processing_helpers import *
from model_helpers import *
from lstm_model import *
from evaluation_helpers import *
import joblib
import pickle
import warnings

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report



# %% Load datasets
################################################################################
print("Loading datasets...")
data_df = pd.read_csv('../temp/data_df.csv')
test_df = pd.read_csv('../temp/test_df.csv')
train_df = pd.read_csv('../temp/train_df.csv')

all_ports = pd.read_csv('../input/all_ports.csv')


# %% Data Preprocessing
################################################################################
# iso2 code for Namibia 'NA' is usually misread by pandas as NaN; this is corrected here
dfs= [data_df, test_df, train_df]
for df in dfs:
    df['country_code_last'] = df['country_code_last'].fillna('NA')
    df['country_code_next'] = df['country_code_next'].fillna('NA')


# Read country from all_ports file
data_df = data_df.merge(all_ports[['country_code', 'country']].drop_duplicates(), left_on='country_code_next', right_on='country_code', how='left')
data_df=data_df.drop(columns=['country_code'])
data_df=data_df.rename(columns={'country':'country_next'})

country_counts = data_df['country_next'].value_counts()
country_counts_df = country_counts.reset_index()
country_counts_df.columns = ['country_next', 'Frequency']
country_counts_df.to_csv('../output/country_counts_all.csv', index=False)

# print("Saving country counts for each vessel type to output/class_freq folder")
classifications =  list(data_df['classification'].unique())
for classific in classifications:
    country_counts = data_df[data_df["classification"]== classific]['country_next'].value_counts()
    country_counts_df = country_counts.reset_index()
    country_counts_df.columns = ['country_next', 'Frequency']
    country_counts_df.to_csv(f'../output/class_freq/country_counts_{classific}.csv', index=False)


# %% Set up the model for evaluation
################################################################################
# print("Loading encoders and model configuration...")
imo_encoder = joblib.load('../output/encoders/imo_encoder.pkl')
classification_encoder = joblib.load('../output/encoders/classification_encoder.pkl')
month_str_encoder = joblib.load('../output/encoders/month_str_encoder.pkl')
country_encoder = joblib.load('../output/encoders/country_encoder.pkl')
tile_encoder = joblib.load('../output/encoders/tile_encoder.pkl')


# Load the trained non_tile and tile categories
with open('../temp/trained_categories_non_tile_columns.pkl', 'rb') as f:
    trained_categories_non_tile_columns = pickle.load(f)

with open('../temp/all_tile_categories.pkl', 'rb') as f:
    all_tile_categories = pickle.load(f)

tile_columns = ['tile_1', 'tile_2', 'tile_3', 'tile_4', 'tile_5', 'tile_6', 'tile_7', 'tile_8', 'tile_9', 'tile_10']
num_tile_features= len(tile_columns)
tile_columns_encoded = [f'{col}_encoded' for col in tile_columns]


# replace unseen values in test_df
test_df = replace_unseen_values(test_df, trained_categories_non_tile_columns, all_tile_categories, tile_columns)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device


print("Setting up model for evaluation...")
# Load model configuration
config = joblib.load('../output/config/model_config.pkl')

model = LSTMWithEmbeddings(
    num_imo_categories=config['num_imo_categories'],
    num_target_categories=config['num_target_categories'],
    num_classification_categories=config['num_classification_categories'],
    num_month_str_categories=config['num_month_str_categories'],
    num_tile_categories=config['num_tile_categories'],
    embedding_dim_imo=config['embedding_dim_imo'],
    embedding_dim_target=config['embedding_dim_target'],
    embedding_dim_classification=config['embedding_dim_classification'],
    embedding_dim_month_str=config['embedding_dim_month_str'],
    embedding_dim_tile=config['embedding_dim_tile'],
    hidden_size=config['hidden_size'],
    num_stacked_layers=config['num_stacked_layers'],
    num_classes=config['num_classes'],
    padding_target=config['padding_target'],
    padding_month_str=config['padding_month_str'],
    padding_tile=config['padding_tile'],
    num_tile_features=config['num_tile_features'],    
    dropout_rate=config['dropout_rate']
)

model.to(device)


# Load the model state (best model with early stopping)
model.load_state_dict(torch.load('../output/models/best_model_val_loss.pt',weights_only=True))
model.eval()


# %% Encodiing the variables
################################################################################
print("Encoding variables and validating features...")
padding_country = padding_target = config['padding_target']
padding_month_str=config['padding_month_str']
padding_tile=config['padding_tile']

test_df[['imo_encoded']] = imo_encoder.transform(test_df[['imo']])
test_df[['classification_encoded']] = classification_encoder.transform(test_df[['classification']])
test_df[['month_str_encoded']] = month_str_encoder.transform(test_df[['month_str']])

test_df[tile_columns_encoded] = tile_encoder.transform(test_df[tile_columns])

test_df['country_code_last_encoded'] = country_encoder.transform(test_df['country_code_last'])
test_df['country_code_next_encoded'] = country_encoder.transform(test_df['country_code_next'])


expected_columns = {
    'imo_encoded': np.number,  # Accepts any numeric type
    'classification_encoded': np.number,  # Accepts any numeric type
    'month_str_encoded': np.number,  # Accepts any numeric type
    'country_code_last_encoded': np.number,  # Accepts any numeric type
    'country_code_next_encoded': np.number,  # Accepts any numeric type
    'tile_1_encoded': np.number,  # Accepts any numeric type
    'tile_2_encoded': np.number,  # Accepts any numeric type
    'tile_3_encoded': np.number,  # Accepts any numeric type
    'tile_4_encoded': np.number,  # Accepts any numeric type
    'tile_5_encoded': np.number,  # Accepts any numeric type
    'tile_6_encoded': np.number,  # Accepts any numeric type
    'tile_7_encoded': np.number,  # Accepts any numeric type
    'tile_8_encoded': np.number,  # Accepts any numeric type
    'tile_9_encoded': np.number,  # Accepts any numeric type
    'tile_10_encoded': np.number  # Accepts any numeric type
    
}

validate_features(test_df, expected_columns)

# %% create sequences
################################################################################
print("Creating input sequences from test data...")
# feature list 
features = ['imo_encoded', 'classification_encoded', 'month_str_encoded', 'country_code_last_encoded',
            'tile_1_encoded', 'tile_2_encoded', 'tile_3_encoded', 'tile_4_encoded', 'tile_5_encoded',
            'tile_6_encoded', 'tile_7_encoded', 'tile_8_encoded', 'tile_9_encoded', 'tile_10_encoded']

target_feature = 'country_code_next_encoded'


lookback = 10
X_test, y_test, imo_encoded_test, classification_encoded_test, country_code_last_encoded_test, month_str_encoded_test, tile_1_encoded_test, tile_2_encoded_test, tile_3_encoded_test, tile_4_encoded_test, tile_5_encoded_test, tile_6_encoded_test, tile_7_encoded_test, tile_8_encoded_test, tile_9_encoded_test, tile_10_encoded_test = create_sequences(
    test_df, padding_month_str, padding_country,padding_tile, features, target_feature, sequence_length=lookback, streak_col='imo')

# Convert sequences to tensors
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

num_features = len(features)

X_test = X_test.reshape((-1, lookback, num_features)) # reshape

print("Inout and output shapes:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# %% Pediction
################################################################################
print("Making predictions on test data...")
with torch.no_grad():
    test_predictions = model(X_test.to(device)).cpu()

test_predicted_classes = torch.argmax(test_predictions, dim=1) # Get the predicted class indices
test_predicted_classes_np = test_predicted_classes.numpy() # Convert to NumPy and reshape for inverse transformation
test_predicted_country = country_encoder.inverse_transform(test_predicted_classes_np) # Apply inverse transformation

y_test_np = y_test.numpy() # Convert to NumPy and reshape for inverse transformation
test_actual_country = country_encoder.inverse_transform(y_test_np) # Apply inverse transformation to get actual labels

# %% Calculate metrics
################################################################################
trained_model_metrics = calculate_metrics(test_actual_country, test_predicted_country, 'weighted')
print(f"Trained model performance metrics: {trained_model_metrics}")
pd.DataFrame([trained_model_metrics]).to_csv('../output/evaluation_results/trained_model_metrics.csv', index=False)
print("Model performance metrics saved to output/evaluation_results folder")


# Decoding and processing the other variables for futher analysis
# Convert to NumPy and reshape for inverse transformation
imo_test_list = imo_encoder.inverse_transform(np.array(imo_encoded_test).reshape(-1, 1))
classification_test_list = classification_encoder.inverse_transform(np.array(classification_encoded_test).reshape(-1, 1))
month_str_test_list = month_str_encoder.inverse_transform(np.array(month_str_encoded_test).reshape(-1, 1))
country_code_last_test_list = [country_encoder.inverse_transform(sublist) for sublist in country_code_last_encoded_test]


# Convert tile lists to NumPy and reshape for inverse transformation
tile_1_encoded_test_array = np.array(tile_1_encoded_test)
tile_2_encoded_test_array = np.array(tile_2_encoded_test)
tile_3_encoded_test_array = np.array(tile_3_encoded_test)
tile_4_encoded_test_array = np.array(tile_4_encoded_test)
tile_5_encoded_test_array = np.array(tile_5_encoded_test)
tile_6_encoded_test_array = np.array(tile_6_encoded_test)
tile_7_encoded_test_array = np.array(tile_7_encoded_test)
tile_8_encoded_test_array = np.array(tile_8_encoded_test)
tile_9_encoded_test_array = np.array(tile_9_encoded_test)
tile_10_encoded_test_array = np.array(tile_10_encoded_test)

# Stack arrays horizontally
encoded_tiles_test = np.column_stack((
    tile_1_encoded_test_array,
    tile_2_encoded_test_array,
    tile_3_encoded_test_array,
    tile_4_encoded_test_array,
    tile_5_encoded_test_array,
    tile_6_encoded_test_array,
    tile_7_encoded_test_array,
    tile_8_encoded_test_array,
    tile_9_encoded_test_array,
    tile_10_encoded_test_array
))

decoded_tiles_test = tile_encoder.inverse_transform(encoded_tiles_test)

# Convert arrays to DataFrame
df1 = pd.DataFrame(imo_test_list, columns=['imo'])
df2 = pd.DataFrame(classification_test_list, columns=['classification'])
df3 = pd.DataFrame(month_str_test_list, columns=['month_str'])
df4 = pd.DataFrame({'country_code_last': country_code_last_test_list})

# Convert decoded_tiles_test to DataFrame
tile_df = pd.DataFrame(decoded_tiles_test, columns=['tile_1', 'tile_2', 'tile_3', 'tile_4', 'tile_5','tile_6', 'tile_7', 'tile_8', 'tile_9', 'tile_10'])

# Concatenate all DataFrames into one
df5 = pd.DataFrame(test_actual_country, columns=['actual_country'])
df6 = pd.DataFrame(test_predicted_country, columns=['predicted_country'])

full_result_df = pd.concat([df1, df2, df3, tile_df,  df4, df5,df6], axis=1)
print("Saving full prediction results to result DataFrame to output folder")
full_result_df.to_csv('../output/full_result_df.csv', index=False)  


################################################################################
# Suppress specific warning from sklearn
warnings.filterwarnings(
    action='ignore', 
    message="y_pred contains classes not in y_true",
    category=UserWarning,
    module='sklearn.metrics._classification' 
)

warnings.filterwarnings(
    action='ignore',
    message="A single label was found in 'y_true' and 'y_pred'.*",
    category=UserWarning,
    module='sklearn'
)

# Suppress specific warnings by message
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive, and thus cannot be shown')
print()

# %% Baseline Models
################################################################################
print("Creating baseline models and calculating metrics for these models...")


# Most frequent class baseline
most_frequent_class = train_df['country_code_next'].value_counts().idxmax()
frequent_class_baseline_predicted = np.full_like(test_predicted_country, most_frequent_class)
most_frequent_class_metrics = calculate_metrics(test_actual_country, frequent_class_baseline_predicted, 'weighted')
print(f"Most Frequent Class Baseline Metrics: ")
print(most_frequent_class_metrics)

trained_model_df = pd.DataFrame([trained_model_metrics], index=['Trained_model'])
most_freq_class_model_df  = pd.DataFrame([most_frequent_class_metrics], index=['Most_frequent_class_model'])
df_combined = pd.concat([trained_model_df, most_freq_class_model_df]).reset_index()
df_combined.rename(columns={'index': 'Model'}, inplace=True)


# most frequent country by imo baseline
################################################################################
most_frequent_country_imo = train_df.groupby('imo')['country_code_next'].agg(lambda x: x.mode()[0]).reset_index()
most_frequent_country_imo= most_frequent_country_imo.rename(columns = {"country_code_next":"most_frequent_country"})

full_result_sub_df = full_result_df[['imo', 'predicted_country']]
full_result_sub_df = pd.merge(left= full_result_sub_df,  right= most_frequent_country_imo, how= 'left', on= 'imo')
most_frequent_class_by_imo = full_result_sub_df['most_frequent_country'].to_numpy()

most_frequent_class_by_imo_metrics = calculate_metrics(test_actual_country,most_frequent_class_by_imo, 'weighted')
print("Most Frequent Class by IMO Baseline Metrics: ")
print(most_frequent_class_by_imo_metrics)

most_freq_class_by_imo_model_df  = pd.DataFrame([most_frequent_class_by_imo_metrics], index=['Most_frequent_class_by_imo_model'])
df_combined_2 = pd.concat([trained_model_df, most_freq_class_by_imo_model_df]).reset_index()
df_combined_2.rename(columns={'index': 'Model'}, inplace=True)


# Last Observation Carried Forward (LOCF) baseline
################################################################################
full_result_df['country_locf'] = full_result_df['country_code_last'].apply(lambda x: x[-1])
locf_class = full_result_df['country_locf'].to_numpy()
locf_metrics = calculate_metrics(test_actual_country,locf_class, 'weighted')
print("Last Observation Carried Forward (LOCF) Baseline Metrics: ")
print(locf_metrics)

locf_model_df  = pd.DataFrame([locf_metrics], index=['Locf_model'])
df_combined_3 = pd.concat([trained_model_df, locf_model_df]).reset_index()
df_combined_3.rename(columns={'index': 'Model'}, inplace=True)


# Uniform Random Class Baseline
################################################################################
# Calculate the distribution of the classes
np.random.seed(42)
class_distribution = train_df['country_code_next'].value_counts(normalize=True)

# Generate random predictions based on the distribution for a new dataset
uniform_random_baseline_df = pd.DataFrame({
    'uniform_random_class': np.random.choice(class_distribution.index, size= len(test_actual_country), p=class_distribution.values)
})

uniform_random_class = uniform_random_baseline_df['uniform_random_class'].to_numpy()
uniform_random_class_metrics = calculate_metrics(test_actual_country,uniform_random_class, 'weighted')
print("Uniform Random Class Metrics")
print(uniform_random_class_metrics)

uniform_random_model_df  = pd.DataFrame([uniform_random_class_metrics], index=['Uniform_random_model'])
df_combined_4 = pd.concat([trained_model_df, uniform_random_model_df]).reset_index()
df_combined_4.rename(columns={'index': 'Model'}, inplace=True)


# Weighted Random Class Baseline
################################################################################
unique_classes = train_df['country_code_next'].unique()
weighted_random_baseline_df = pd.DataFrame({
    'weighted_random_class': np.random.choice(unique_classes, size=len(test_actual_country), p=[1/len(unique_classes)]*len(unique_classes))
})
weighted_random_class = weighted_random_baseline_df['weighted_random_class'].to_numpy()

weighted_random_class_metrics = calculate_metrics(test_actual_country,weighted_random_class, 'weighted')
print("Weighted Random Class Metrics")
print(weighted_random_class_metrics)

weighted_random_model_df  = pd.DataFrame([weighted_random_class_metrics], index=['Weighted_random_model'])
df_combined_5 = pd.concat([trained_model_df, weighted_random_model_df]).reset_index()
df_combined_5.rename(columns={'index': 'Model'}, inplace=True)
df_combined_5

# combine all baselines with trained model
################################################################################
df_combined_6 = pd.concat([df_combined, df_combined_2, df_combined_3, df_combined_4, df_combined_5]).drop_duplicates("Model").reset_index(drop=True)

print("Saving baseline model metrics to output/evaluation_results folder")
df_combined_6.to_csv('../output/evaluation_results/baselines.csv', index=False)
print()

# %% Metrics by vessel classification
################################################################################
print("Additional Analysis of model performance")
print()
print("Calculating performance metrics by vessel classification:")
imo_count_df = data_df['imo'].value_counts().reset_index()
classification_count_df = data_df['classification'].value_counts().reset_index()
class_count_df = data_df['country_code_next'].value_counts().reset_index()
cols= full_result_df.columns.tolist() 

# Group by 'classification' and calculate metrics
metrics_by_classification = full_result_df.groupby('classification')[cols].apply(calculate_metrics_by_group,averaging_method = 'weighted').reset_index()
metrics_by_classification = pd.merge(left = metrics_by_classification, right= classification_count_df, how= 'left', on= "classification")
classification_order = ['Small', 'Feeder', 'Feedermax',  'Panamax', 'New-Panamax', 'Post-Panamax', 'ULCV']
metrics_by_classification['classification'] = pd.Categorical(metrics_by_classification['classification'], categories=classification_order, ordered=True)
metrics_by_classification= metrics_by_classification.sort_values('classification')

print("Saving metrics by vessel classification to output/evaluation_results folder")
metrics_by_classification.to_csv('../output/evaluation_results/metrics_by_classification.csv', index=False)
print()

################################################################################
# Metrics by vessel IMO
print("Calculating performance metrics by vessel classification:")
metrics_by_imo = full_result_df.groupby('imo')[cols].apply(calculate_metrics_by_group, averaging_method = 'weighted' ).reset_index()
metrics_by_imo = pd.merge(left = metrics_by_imo, right= imo_count_df, how= 'left', on= "imo")
metrics_by_imo = pd.merge(left = metrics_by_imo, right= test_df[["imo","classification"]].drop_duplicates("imo"), how= 'left', on= "imo")



# Metrics by Class (Country)
print("Calculating performance metrics by class i.e Country:")
metrics_by_class = full_result_df.groupby('actual_country')[cols].apply(calculate_metrics_by_group, averaging_method = 'weighted' ).reset_index()
metrics_by_class = pd.merge(left = metrics_by_class, right= class_count_df, how= 'left',left_on= 'actual_country', right_on= "country_code_next")
metrics_by_class= metrics_by_class.drop(columns="country_code_next")
metrics_by_class = metrics_by_class.merge(all_ports[['country_code', 'country']].drop_duplicates(), left_on='actual_country', right_on='country_code', how='left')
metrics_by_class= metrics_by_class.drop(columns="country_code")


################################################################################
# Accuracy by Number of Trips 
print("Plotting accuracy metrics by number of IMO trips:")
fig = plot_metrics(metrics_by_imo, "all", "Accuracy by No of Trips")
print("Saving accuracy metric by number of trips plot to output/figures folder")
fig.savefig("../output/figures/accuracy_by_no_trips.png", format='png', dpi=300)

print("Plotting accuracy metrics by number of IMO trips for each classification:")
print("Saving the plots to output/figures folder")
for classific in metrics_by_imo["classification"].unique():
    fig = plot_metrics(metrics_by_imo, [classific], f"Accuracy by No of Trips for {classific}")
    fig.savefig(f"../output/figures/accuracy_by_no_trips_for_{classific}.png", format='png', dpi=300)

################################################################################
# Accuracy Distribution across Thresholds for Vessels
print("Plotting accuracy distribution across thresholds for vessels:")
fig = plot_accuracy_distribution(metrics_by_imo, "bal_accuracy","Vessel: Accuracy Distribution across Thresholds")
fig.show()
plt.tight_layout()
fig.savefig("../output/figures/accuracy_distribution_Thresholds_vessels.png", format='png', dpi=300)


for classific in metrics_by_imo["classification"].unique():
    df = metrics_by_imo[metrics_by_imo["classification"]== classific]
    fig = plot_accuracy_distribution(df, "accuracy", f"Accuracy Distribution across Thresholds for {classific}")
    # Suppress specific warnings by message
    fig.show()
    fig.savefig(f'../output/figures/accuracy_distribution_thresholds_class_for_{classific}.png')  # Save the figure if desired

print("Saving accuracy distribution plots to output/figures folder")
print()


# Accuracy Distribution across Thresholds for Vessels
################################################################################
print("Plotting accuracy distribution across thresholds for target classes i.e countries:")
fig = plot_accuracy_distribution(metrics_by_class, "accuracy"," Actual Class: Accuracy Distribution across Thresholds")
fig.show()
fig.savefig(f'../output/figures/accuracy_distribution_thresholds_class.png')  # Save the figure if desired
print("Saving accuracy distribution plots to output/figures folder")
Print("All evaluation completed successfully!")

# %%
