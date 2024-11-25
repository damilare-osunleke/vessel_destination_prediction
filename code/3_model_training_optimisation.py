# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
from data_processing_helpers import *
from model_helpers import *
from lstm_model import *

from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


import optuna
from optuna.visualization import plot_optimization_history
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
import seaborn as sns

# Read input data
################################################################################
print("Reading input data...")
trips_summary_df= pd.read_csv("../temp/trips_summary_df.csv")

# iso2 code for Namibia 'NA' is usually misread by pandas as NaN; this is corrected here
trips_summary_df.loc[trips_summary_df['locode_last'].str[:2] == 'NA', 'country_code_last'] = 'NA'
trips_summary_df.loc[trips_summary_df['locode_next'].str[:2] == 'NA', 'country_code_next'] = 'NA'


print("Preprocessing data and splitting into train, validation and test datasets...")
# Split the 'tiles' into different columns
trips_summary_df['tiles_split'] = trips_summary_df['tiles'].str.split(', ')
for i in range(1, 11):
    trips_summary_df[f'tile_{i}'] = trips_summary_df['tiles_split'].apply(lambda x: x[i-1] if len(x) >= i else np.nan) # fill with NaN if there are less than 10 tiles

trips_summary_df.drop('tiles_split', axis=1, inplace=True)

# Extract ionly vessels with more than 20 trips
trips_summary_df['imo_count'] = trips_summary_df.groupby('imo')['imo'].transform('count')
data_df = trips_summary_df[trips_summary_df["imo_count"]>=20].reset_index(drop=True) # filter out the ships with less than 20 trips

non_tile_columns = ['imo', 'classification', 'country_code_last', 'country_code_next', 'month_str']
tile_columns = ['tile_1', 'tile_2', 'tile_3', 'tile_4', 'tile_5', 'tile_6', 'tile_7', 'tile_8', 'tile_9', 'tile_10']
data_df = data_df[non_tile_columns + tile_columns] # select only the required columns

# Replace NaN tiles with 'nil' 
data_df[tile_columns] = data_df[tile_columns].fillna('nil')

# split into trian, val and test datasets using the function train_val_test_split
train_df, val_df, test_df = train_val_test_split(data_df)

# save datasets 
print("Saving datasets to temp folder...")
data_df.to_csv("../temp/data_df.csv", index=False)
train_df.to_csv("../temp/train_df.csv", index=False)
val_df.to_csv("../temp/val_df.csv", index=False)
test_df.to_csv("../temp/test_df.csv", index=False)


################################################################################
print("Processing datasets for feature encoding")
# Collect known categories and numerical values for each column from train_df
trained_categories_non_tile_columns = {column: set(train_df[column].unique()) for column in train_df[non_tile_columns].columns}
all_tile_categories = np.append(np.unique(train_df[tile_columns].values.ravel()),'unseen')


# Replace unseen values in validation and test datasets
val_df = replace_unseen_values(val_df, trained_categories_non_tile_columns, all_tile_categories, tile_columns)
test_df = replace_unseen_values(test_df, trained_categories_non_tile_columns, all_tile_categories, tile_columns)


# Save categories for tile and non-tile columns
with open('../temp/trained_categories_non_tile_columns.pkl', 'wb') as f:
    pickle.dump(trained_categories_non_tile_columns, f)

with open('../temp/all_tile_categories.pkl', 'wb') as f:
    pickle.dump(all_tile_categories, f)


print("Encoding the features...")
# Encode the columns
num_tile_features= len(tile_columns)
tile_columns_encoded = [f'{col}_encoded' for col in tile_columns]

# Initialize the Ordinal Encoders
imo_encoder = OrdinalEncoder()
classification_encoder = OrdinalEncoder()
month_str_encoder = OrdinalEncoder()

tile_encoder = OrdinalEncoder(categories=[all_tile_categories for _ in tile_columns])

country_encoder = LabelEncoder()
train_countries = pd.concat([train_df['country_code_last'], train_df['country_code_next']])

################################################################################
print("Fitting and transforming the encoders on the training data...")
# Fit the encoder on the training data and transform the data
train_df[['imo_encoded']] = imo_encoder.fit_transform(train_df[['imo']])
train_df[['classification_encoded']] = classification_encoder.fit_transform(train_df[['classification']])
train_df[['month_str_encoded']] = month_str_encoder.fit_transform(train_df[['month_str']])

train_df[tile_columns_encoded] = tile_encoder.fit_transform(train_df[tile_columns])

country_encoder.fit(train_countries.unique())
train_df['country_code_last_encoded'] = country_encoder.transform(train_df['country_code_last'])
train_df['country_code_next_encoded'] = country_encoder.transform(train_df['country_code_next'])

print("Transforming the validation and test data using the fitted encoders...")
# transform the validation and test data using the fitted encoders
val_df[['imo_encoded']] = imo_encoder.transform(val_df[['imo']])
val_df[['classification_encoded']] = classification_encoder.transform(val_df[['classification']])
val_df[['month_str_encoded']] = month_str_encoder.transform(val_df[['month_str']])

val_df[tile_columns_encoded] = tile_encoder.transform(val_df[tile_columns])

val_df['country_code_last_encoded'] = country_encoder.transform(val_df['country_code_last'])
val_df['country_code_next_encoded'] = country_encoder.transform(val_df['country_code_next'])


test_df[['imo_encoded']] = imo_encoder.transform(test_df[['imo']])
test_df[['classification_encoded']] = classification_encoder.transform(test_df[['classification']])
test_df[['month_str_encoded']] = month_str_encoder.transform(test_df[['month_str']])

test_df[tile_columns_encoded] = tile_encoder.transform(test_df[tile_columns])

test_df['country_code_last_encoded'] = country_encoder.transform(test_df['country_code_last'])
test_df['country_code_next_encoded'] = country_encoder.transform(test_df['country_code_next'])


################################################################################
print("Setting padding values for sequence creation...")
# Retrieve max indices from each encoder and add 1 for the padding value which will be used when creating indexes
padding_month_str = len(month_str_encoder.categories_[0])
padding_tile = len(tile_encoder.categories_[0])
padding_country = int(train_df['country_code_last_encoded'].max()) + 1

print("Padding values:")
print("Month Str:", padding_month_str)
print("Tile:", padding_tile)
print("country:", padding_country)


# A dictionary of all features and their expected types (numeric for all encoded features)
print("Validating the encoded features using function validate_features...")
expected_columns = {
    'imo_encoded': np.number,  
    'classification_encoded': np.number,  
    'month_str_encoded': np.number,  
    'country_code_last_encoded': np.number,  
    'country_code_next_encoded': np.number,  
    'tile_1_encoded': np.number,  
    'tile_2_encoded': np.number,  
    'tile_3_encoded': np.number,  
    'tile_4_encoded': np.number,  
    'tile_5_encoded': np.number,  
    'tile_6_encoded': np.number,  
    'tile_7_encoded': np.number,  
    'tile_8_encoded': np.number,  
    'tile_9_encoded': np.number, 
    'tile_10_encoded': np.number  
    
}

validate_features(train_df, expected_columns)
validate_features(val_df, expected_columns)
validate_features(test_df, expected_columns)

print("Creating sequences and targets for training, validation, and testing...")
# feature list
features = ['imo_encoded', 'classification_encoded', 'month_str_encoded', 'country_code_last_encoded',
            'tile_1_encoded', 'tile_2_encoded', 'tile_3_encoded', 'tile_4_encoded', 'tile_5_encoded',
            'tile_6_encoded', 'tile_7_encoded', 'tile_8_encoded', 'tile_9_encoded', 'tile_10_encoded']

target_feature = 'country_code_next_encoded'

# Create sequences and get corresponding imos for training, validation, and testing
lookback = 10
X_train, y_train, imo_encoded_train, classification_encoded_train, country_code_last_encoded_train, month_str_encoded_train, tile_1_encoded_train, tile_2_encoded_train, tile_3_encoded_train, tile_4_encoded_train, tile_5_encoded_train, tile_6_encoded_train, tile_7_encoded_train, tile_8_encoded_train, tile_9_encoded_train, tile_10_encoded_train = create_sequences(
    train_df, padding_month_str, padding_country,padding_tile, features, target_feature, sequence_length=lookback, streak_col='imo')

X_val, y_val, imo_encoded_val, classification_encoded_val, country_code_last_encoded_val, month_str_encoded_val, tile_1_encoded_val, tile_2_encoded_val, tile_3_encoded_val, tile_4_encoded_val, tile_5_encoded_val, tile_6_encoded_val, tile_7_encoded_val, tile_8_encoded_val, tile_9_encoded_val, tile_10_encoded_val = create_sequences(
    val_df, padding_month_str, padding_country,padding_tile, features, target_feature, sequence_length=lookback, streak_col='imo')

X_test, y_test, imo_encoded_test, classification_encoded_test, country_code_last_encoded_test, month_str_encoded_test, tile_1_encoded_test, tile_2_encoded_test, tile_3_encoded_test, tile_4_encoded_test, tile_5_encoded_test, tile_6_encoded_test, tile_7_encoded_test, tile_8_encoded_test, tile_9_encoded_test, tile_10_encoded_test = create_sequences(
    test_df, padding_month_str, padding_country,padding_tile, features, target_feature, sequence_length=lookback, streak_col='imo')

print("Sequence and target creation complete.")
print()

################################################################################
print("Converting the sequences and targets to tensors...")
X_train = torch.tensor(X_train).float()
X_val = torch.tensor(X_val).float()
X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train).long()  # long for classification targets
y_val = torch.tensor(y_val).long()
y_test = torch.tensor(y_test).long()

num_features = len(features)

# reshape inputs
X_train = X_train.reshape((-1, lookback, num_features))
X_val = X_val.reshape((-1, lookback, num_features))
X_test = X_test.reshape((-1, lookback, num_features))


print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device

################################################################################
print("setting up data loaders...")
# Define custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)


# setup data loader instances
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

for _, batch in enumerate(train_loader):
    X_batch, y_batch = batch[0].to(device), batch[1].to(device)

    
    print(X_batch.shape, y_batch.shape)
    break


# Defining model input variables
num_imo_categories = len(imo_encoder.categories_[0])
num_target_categories = len(country_encoder.classes_) + 1  # Plus 1 for padding 
num_classification_categories = len(classification_encoder.categories_[0])
num_month_str_categories = len(month_str_encoder.categories_[0]) + 1  # Plus 1 for padding 
num_tile_categories = len(tile_encoder.categories_[0]) + 1  # Plus 1 for padding

num_classes = num_target_categories

padding_target = padding_country
padding_month_str = padding_month_str
padding_tile = padding_tile



# %% Model training and hyperameter tuning using Optuna
################################################################################
print("Model training and hyperameter tuning using Optuna...")

num_epochs = 50
# Define the objective function for parameter tuning
def objective(trial):
    # Hyperparameters to tune
    embedding_dim = trial.suggest_categorical('embedding_dim', [12, 24, 36])
    hidden_size = trial.suggest_int('hidden_size', 50, 200)
    num_stacked_layers = trial.suggest_int('num_stacked_layers', 1, 4)

    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Replaces suggest_uniform
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    # Instantiate the model with both fixed parameters and those to be tuned by Optuna
    model = LSTMWithEmbeddings(
        num_imo_categories=num_imo_categories,
        num_target_categories=num_target_categories,
        num_classification_categories=num_classification_categories,
        num_month_str_categories=num_month_str_categories,
        num_tile_categories=num_tile_categories,
        embedding_dim_imo=embedding_dim, 
        embedding_dim_target=embedding_dim,
        embedding_dim_classification=embedding_dim,
        embedding_dim_month_str=embedding_dim,
        embedding_dim_tile=embedding_dim,
        hidden_size=hidden_size,
        num_stacked_layers=num_stacked_layers,
        num_classes=num_classes,
        padding_target=padding_target,
        padding_month_str=padding_month_str,
        padding_tile=padding_tile,
        num_tile_features=num_tile_features, 
        dropout_rate=dropout_rate
    )
    model.to(device)


    # Optimizer setup
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    # Training and validation loop
    for epoch in range(num_epochs):

        # Log the current trial and epoch
        print()
        print('............................................')
        print(f"Starting Trial {trial.number + 1} of {total_trials}, Epoch {epoch + 1} of {num_epochs}")

        train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch)
        print()
        val_loss = validate_one_epoch(model, val_loader, loss_function, device, epoch)
        


        # Report intermediate values to Optuna
        trial.report(val_loss, epoch)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break  # Early stopping condition met

        # Handle early stopping if the trial should be pruned
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_loss
    

total_trials = 25

# Create a study with SQLite storage f
study = optuna.create_study(
    study_name="optimize_lstm_params",
    storage="sqlite:///../temp/optimize_lstm_params.db",  # Path to SQLite database
    load_if_exists=True,  
    direction="minimize"
)

study.optimize(objective, n_trials=total_trials)  # Specify the number of trials

# %%
################################################################################
best_params = study.best_params
print(best_params)


# Print the best trial
print('Best trial:')
best_trial = study.best_trial
print(f'  Value: {best_trial.value}')
print('  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')


# Generate the optimization history plot
fig = plot_optimization_history(study)
fig.write_html("../output/figures/optimization_history.html")


# %%
################################################################################
print("Saving the encoders, best model configuration and other artifacts...")
# Save the scaler and encoders
joblib.dump(imo_encoder, '../output/encoders/imo_encoder.pkl')
joblib.dump(classification_encoder, '../output/encoders/classification_encoder.pkl')
joblib.dump(month_str_encoder, '../output/encoders/month_str_encoder.pkl')
joblib.dump(country_encoder, '../output/encoders/country_encoder.pkl')
joblib.dump(tile_encoder, '../output/encoders/tile_encoder.pkl')

# Save any additional configuration as a dictionary
config = {
    'num_imo_categories': num_imo_categories,
    'num_target_categories': num_target_categories,
    'num_classification_categories': num_classification_categories,
    'num_month_str_categories': num_month_str_categories,
    'num_tile_categories': num_tile_categories,
    'embedding_dim_imo': best_params['embedding_dim'],
    'embedding_dim_target': best_params['embedding_dim'],
    'embedding_dim_classification': best_params['embedding_dim'],
    'embedding_dim_month_str': best_params['embedding_dim'],
    'embedding_dim_tile': best_params['embedding_dim'],
    'hidden_size': best_params['hidden_size'],
    'num_stacked_layers': best_params['num_stacked_layers'],
    'num_classes': num_classes,
    'padding_target': padding_target,
    'padding_month_str': padding_month_str,
    'padding_tile': padding_tile,
    'num_tile_features': num_tile_features,
    'dropout_rate': best_params['dropout_rate']
}

joblib.dump(config, '../output/config/model_config.pkl')

print()
print("Hyperparameter tuning complete. Artifacts saved to output folder.")
print("Model training and optimisation completed successfully!")
