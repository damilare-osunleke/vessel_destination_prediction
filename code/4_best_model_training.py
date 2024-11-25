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
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
import seaborn as sns


# %% Load training, validation and test datasets 
################################################################################
print("Loading training, validation and test datasets...")
data_df = pd.read_csv("../temp/data_df.csv")
train_df = pd.read_csv("../temp/train_df.csv")
val_df = pd.read_csv("../temp/val_df.csv")
test_df = pd.read_csv("../temp/test_df.csv")

# iso2 code for Namibia 'NA' is usually misread by pandas as NaN; this is corrected here
dfs= [data_df, train_df, val_df, test_df]
for df in dfs:
    df['country_code_last'] = df['country_code_last'].fillna('NA')
    df['country_code_next'] = df['country_code_next'].fillna('NA')


# Load the encoders
################################################################################
print("Loading the encoders...")
imo_encoder = joblib.load('../output/encoders/imo_encoder.pkl')
classification_encoder = joblib.load('../output/encoders/classification_encoder.pkl')
month_str_encoder = joblib.load('../output/encoders/month_str_encoder.pkl')
country_encoder = joblib.load('../output/encoders/country_encoder.pkl')
tile_encoder = joblib.load('../output/encoders/tile_encoder.pkl')


# Load additional variables
print("Loading additional variables...")
with open('../temp/trained_categories_non_tile_columns.pkl', 'rb') as f:
    trained_categories_non_tile_columns = pickle.load(f)

with open('../temp/all_tile_categories.pkl', 'rb') as f:
    all_tile_categories = pickle.load(f)


# Load best model configuration
config = joblib.load('../output/config/model_config.pkl')

padding_target = padding_country = config['padding_target']
padding_month_str=config['padding_month_str']
padding_tile=config['padding_tile']

################################################################################
print("Preprocessing the data...")
tile_columns = ['tile_1', 'tile_2', 'tile_3', 'tile_4', 'tile_5', 'tile_6', 'tile_7', 'tile_8', 'tile_9', 'tile_10']
num_tile_features= len(tile_columns)
tile_columns_encoded = [f'{col}_encoded' for col in tile_columns]

# Replace unseen values in validation and test datasets
val_df = replace_unseen_values(val_df, trained_categories_non_tile_columns, all_tile_categories, tile_columns)
test_df = replace_unseen_values(test_df, trained_categories_non_tile_columns, all_tile_categories, tile_columns)

# Encode the columns by transforming with the fitted encoders
print("Encoding the features...")
print("Transforming the training data with the fitted encoders...")
train_df[['imo_encoded']] = imo_encoder.transform(train_df[['imo']])
train_df[['classification_encoded']] = classification_encoder.transform(train_df[['classification']])
train_df[['month_str_encoded']] = month_str_encoder.transform(train_df[['month_str']])

train_df[tile_columns_encoded] = tile_encoder.transform(train_df[tile_columns])

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


# A dictionary of all features and their expected types (numeric for all encoded features)
################################################################################
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
print("Converting the sequences and targets to tensors and reshaping the tensors...")
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
print("Setting up data loader...")
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

    print("Data batch shapes for input and target:")
    print(X_batch.shape, y_batch.shape)
    break

# %%
################################################################################
print("Setting up the model...")
# Load the optuna study from SQLite database
study = optuna.load_study(
    study_name="optimize_lstm_params",
    storage="sqlite:///../temp/optimize_lstm_params.db"  # Adjust the path if needed
)
best_params = study.best_params
best_params

# Initialize the model with the loaded configuration
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


################################################################################
print("Training the model with the best hyperparameters...")
learning_rate = best_params['learning_rate']
loss_function = nn.CrossEntropyLoss() 
optimizer = Adam(model.parameters(), lr= learning_rate)

early_stopping = EarlyStopping(patience=5, verbose=True)

training_losses = []
validation_losses = []

num_epochs = 50

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}/{num_epochs}')
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch)
    val_loss = validate_one_epoch(model, val_loader, loss_function, device, epoch)
    print(f'Completed Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("Training complete.")
print("Saving the model and other artifacts...")
# save training and validation losses
losses_df = pd.DataFrame({
    'Training Losses': training_losses,
    'Validation Losses': validation_losses
})

print("Saving the losses to output folder...")
losses_df.to_csv('../output/losses.csv', index=False)

#  %% plot and save the training and validation losses
################################################################################
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Training and Validation Losses')
plt.legend()

# Save the plot
plt.savefig('../output/figures/losses_plot.png')

plt.show()

# SAVE MODEL (best model has already ben saved during training)
print("Saving the best model to output/models folder...")
torch.save(model.state_dict(), '../output/models/final_model_epoch_50.pt')
print()
print("Best model training completed successfully!")