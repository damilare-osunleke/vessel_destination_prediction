import pandas as pd
import h3
import numpy as np

def calculate_streaks_and_gaps(df, col_name_1, col_name_2, col_name_3, col_name_4,n_gap):
    """
    Function to calculate streaks, streak gaps, and streak lengths based on days between records for each 'imo', 
    with an additional unique super_id across the entire dataset based on unique [imo, streak_id] pairs.
    
    Arguments:
    df -- the input dataframe
    col_name_1 -- the name for the 'streak_id' column to be created
    col_name_2 -- the name for the 'streak_gap' column to be created
    col_name_3 -- the name for the 'streak_length' column to be created
    col_name_4 -- the name for the 'super_id' column to be created
    
    Returns:
    df -- dataframe with recalculated streaks, gaps, and streak lengths
    """
    
    # calculate streaks ids
    df['timestamp'] = pd.to_datetime(df['timestamp'])    
    df['date_diff'] = df.groupby('imo')['timestamp'].diff().dt.days # Calculate the difference in days between consecutive dates for each 'imo'   
    df['new_streak'] = df.groupby('imo')['date_diff'].transform(lambda x: (x > n_gap).cumsum()) # new streaks when the day dif > n_gap 
    df[col_name_1] = df.groupby('imo')['new_streak'].transform(lambda x: x.factorize()[0] + 1) # cummulate streak_id for each unique imo
    
    # Calculate streak gaps for each streak
    df[col_name_2] = df['date_diff'] - 1 # Calculate the initial gap for each streak start
    df[col_name_2] = df[col_name_2].where(df['date_diff'] > n_gap)    
    df[col_name_2] = df.groupby(['imo', col_name_1])[col_name_2].transform('ffill') # Propagate the streak gap across the same streak
    df[col_name_3] = df.groupby(['imo', col_name_1])['timestamp'].transform('count') # Calculate the streak length for each streak
    
    # Create a unique super_id for each unique (imo, streak_id) pair
    df['unique_pair'] = df['imo'].astype(str) + '_' + df[col_name_1].astype(str)
    df[col_name_4] = pd.factorize(df['unique_pair'])[0] + 1
    df.drop(columns='unique_pair', inplace=True)  # Cleanup temporary column

    # Drop intermediate columns used for calculations
    df.drop(columns=['date_diff', 'new_streak'], inplace=True)
    
    return df



def get_h3_cell(row, resolution):
    """
    Convert latitude and longitude to H3 cell.

    This function takes a row from a DataFrame, extracts the latitude and longitude,
    and converts them to an H3 cell at the specified resolution.

    Parameters:
    row (pandas.Series): A row from a DataFrame containing 'latitude' and 'longitude' columns.
    resolution (int): The resolution level for the H3 cell.

    Returns:
    str: The H3 cell identifier.
    """
    lat, lng = row["pos_latitude"], row["pos_longitude"]
    cell = h3.geo_to_h3(lat, lng, resolution)
    return cell


def is_neighbor(cell1, cell2, ring_size):
    """
    Determines if one H3 cell is a neighbor of another within a specified ring distance.

    Parameters:
    - cell1 (str): H3 index of the first cell.
    - cell2 (str): H3 index of the second cell.
    - ring (int): The ring distance to check for neighborhood.

    Returns:
    - bool: True if cell2 is within the specified ring distance from cell1, False otherwise.
    """

    if pd.isna(cell1) or pd.isna(cell2):
        return False
    try:
        neighbors = h3.k_ring(cell1, ring_size) # get neighbors at the specified ring distance
        return cell2 in neighbors
    except Exception as e:
        print(f"Error checking neighbors for cells {cell1} and {cell2}: {e}")
        return False


def clean_anomalies(group, ring_size):
    """
    Filters rows in a DataFrame group by ensuring each row's H3 cell is a neighbor
    to its preceding or succeeding row's H3 cell.

    Parameters:
    - group (DataFrame): The group of DataFrame rows to process.
    - ring_size (int): The ring distance to use for neighborhood checks.

    Returns:
    - DataFrame: A DataFrame containing only the rows where each cell is a neighbor
                 to either the previous or next cell.
    """


    cleaned_rows = []
    num_rows = len(group)
    group = group.reset_index(drop=True)

    for idx in range(num_rows):
        current_cell = group.loc[idx, 'cell_h3']
        
        previous_cell = group.loc[idx - 1, 'cell_h3'] if idx > 0 else None # Get the previous cell, handle edge case for the first entry
        next_cell = group.loc[idx + 1, 'cell_h3'] if idx < num_rows - 1 else None # Get the next cell, handle edge case for the last entry

        # Check neighbors: current must be neighbor of either previous or next
        valid_with_previous = is_neighbor(previous_cell, current_cell, ring_size) if previous_cell else False
        valid_with_next = is_neighbor(next_cell, current_cell, ring_size) if next_cell else False

        # Append row to cleaned data if it's a neighbor to either previous or next
        if valid_with_previous or valid_with_next:
            cleaned_rows.append(group.loc[idx])

    return pd.DataFrame(cleaned_rows)


def remove_consecutive_duplicates(group, col_1='cell_h3', col_2='locode_last'):
    """
    Removes consecutive duplicate entries in a DataFrame based on specific columns.    
    i.e. where the specified columns (`col_1` and `col_2`) have the same values
    as the previous row. The function is intended to be used on grouped DataFrame objects where
    such filtering needs to be applied within each group.

    Parameters:
    - group (DataFrame): The DataFrame group.
    - col_1 (str, optional): the first column to check for consecutive duplicates.
    - col_2 (str, optional): the second column to check for consecutive duplicates.

    Returns:
    - DataFrame: A DataFrame with consecutive duplicates removed based on the specified columns.
    """
    
    mask = (group[col_1] != group[col_1].shift()) | (group[col_2] != group[col_2].shift())
    group = group[mask]
    return group


def summarize_trips(df):    
    """
    Summarizes trip data for each vessel, consolidating consecutive records
    into a single record. Identifies new trips by changes in 'locode_last' or shifts in 'imo'.

    Parameters:
    - df (pd.DataFrame): DataFrame with vessel trip details including 'imo', 'locode_last', 
    'locode_last', 'timestamp', 'cell_h3', 'course', 'speed', 'classification', 'month_str'.

    Returns:
    - pd.DataFrame: Summarized trip data including start and end ports, times, unique tiles,
    course, and speed details for each trip.
    """

    df['new_trip'] = (df['locode_last'] != df['locode_last'].shift()) | (df['imo'] != df['imo'].shift())
    df['trip_id'] = df.groupby('imo')['new_trip'].cumsum()  # Create a trip_id within each imo group

    # Aggregating data for each trip within each imo
    trip_summary = df.groupby(['imo', 'trip_id']).agg(
        locode_last=('locode_last', 'first'),
        locode_next=('locode_next', 'first'),
        country_code_last=('country_code_last', 'first'),
        country_code_next=('country_code_next', 'first'),
        departure_time=('timestamp', 'first'),
        arrival_time=('timestamp', 'last'),
        tiles=('cell_h3', lambda x: ', '.join(x.unique())),
        classification=('classification', 'first'),
        month_str=('month_str', 'first')
    ).reset_index()

    trip_summary.drop('trip_id', axis=1, inplace=True)
    return trip_summary




def train_val_test_split(df, train_frac=0.75, val_frac=0.15):
    """
    Splits the data into training, validation, and test sets based on time for each unique identifier (IMO).
    
    Parameters:
    - df (DataFrame): The input DataFrame containing all data.
    - train_frac (float): Fraction of data to be used for training.
    - val_frac (float): Fraction of data to be used for validation.
    
    Returns:
    - train_df (DataFrame): Training data subset.
    - val_df (DataFrame): Validation data subset.
    - test_df (DataFrame): Test data subset.
    """
    train_data = []
    val_data = []
    test_data = []
    grouped_by_imo = df.groupby('imo')

    for imo, group in grouped_by_imo:
        # Calculate indices for splitting data
        train_size = int(len(group) * train_frac)
        val_size = int(len(group) * val_frac)

        # Slice the data into train, validation, and test segments
        train_group = group.iloc[:train_size]
        val_group = group.iloc[train_size:train_size + val_size]
        test_group = group.iloc[train_size + val_size:]
        
        # Collect the split data
        train_data.append(train_group)
        val_data.append(val_group)
        test_data.append(test_group)

    # Concatenate the lists of dataframes to form full datasets
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df



def replace_unseen_values(df, non_tile_features, all_tile_categories, tile_columns):
    """
    Replaces unseen values in a DataFrame for both tile and non-tile features with placeholders.

    For non-tile features, replaces unseen categorical values with 'unseen' and numerical values with 9999999.
    For tile features, unseen values are replaced with 'unseen'.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        non_tile_features (dict): A dictionary mapping non-tile feature columns to sets of known values.
        all_tile_categories (set): A set of all known tile category values.
        tile_columns (list): A list of column names in `df` that are considered tile features.

    Returns:
        pd.DataFrame: The DataFrame with unseen values replaced.
    """
    for column in df.columns:
        if column not in tile_columns:  # This column is a non-tile feature
            if df[column].dtype == 'object':  # Handling object columns
                known_categories = non_tile_features.get(column, set())
                df[column] = df[column].apply(lambda x: 'unseen' if x not in known_categories else x)
            else:  # Handling numerical columns
                known_values = non_tile_features.get(column, set())
                df[column] = df[column].apply(lambda x: 9999999 if x not in known_values else x)
        else:  # Handling tile features
            df[column] = df[column].apply(lambda x: 'unseen' if x not in all_tile_categories else x)
    
    return df



def validate_features(df, expected_columns):
    """
    Ensures that the DataFrame has the expected columns with the correct data types.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (dict): A dictionary mapping column names to their expected data types.
        
    Raises:
        ValueError: If there are missing columns.
        TypeError: If a column has an incorrect type.
    """
    if not set(expected_columns.keys()).issubset(set(df.columns)):
        missing_cols = list(set(expected_columns.keys()) - set(df.columns))
        raise ValueError(f"Missing expected columns: {missing_cols}")

    for col, expected_type in expected_columns.items():
        if not np.issubdtype(df[col].dtype, expected_type):
            raise TypeError(f"Column {col} expected to be of type {expected_type}, but got {df[col].dtype}")




def create_sequences(df, padding_month_str, padding_country, padding_tile, features, target_feature, sequence_length, streak_col='imo'):

    """
    Generates sequences and targets from the given DataFrame grouped by a specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data to process.
        padding_month_str (int): Padding value for the month_str feature in sequences.
        padding_location (str): Padding value for tile features in sequences.
        features (list): List of column names to include in sequences.
        target_feature (str): Column name to use as target for each sequence.
        sequence_length (int): Length of each sequence.
        streak_col (str): Column name to group the DataFrame by for sequence generation.

    Returns:
        tuple: Tuple containing arrays for sequences (X), targets (y), and additional features extracted from the sequences.    """


    X = []
    y = []
    imo_encoded = []
    classification_encoded = []
    country_code_last_encoded = []
    month_str_encoded = []
    tile_1_encoded = []
    tile_2_encoded = []
    tile_3_encoded = []
    tile_4_encoded = []
    tile_5_encoded = []
    tile_6_encoded = []
    tile_7_encoded = []
    tile_8_encoded = []
    tile_9_encoded = []
    tile_10_encoded = []
    
    grouped = df.groupby(streak_col)
    for _, group in grouped:
        if len(group) >= sequence_length:
            for i in range(len(group) - sequence_length + 1):
                seq = group[features].iloc[i:i + sequence_length].values
                target = group.iloc[i + sequence_length -1][target_feature]
                X.append(seq)
                y.append(target)
                
                imo_encoded.append(group.iloc[i + sequence_length - 1][features[0]])  # imo_encoded
                classification_encoded.append(group.iloc[i + sequence_length - 1][features[1]])  # classification_encoded
                month_str_encoded.append(group.iloc[i + sequence_length - 1][features[2]])  # month_str_encoded
                country_code_last_encoded.append(group.iloc[i:i + sequence_length][features[3]].tolist())  # list of all country_code_last_encoded in the sequence

                tile_1_encoded.append(group.iloc[i + sequence_length - 1][features[4]])  # tile_1_encoded
                tile_2_encoded.append(group.iloc[i + sequence_length - 1][features[5]])  # tile_2_encoded
                tile_3_encoded.append(group.iloc[i + sequence_length - 1][features[6]])  # tile_3_encoded
                tile_4_encoded.append(group.iloc[i + sequence_length - 1][features[7]])  # tile_4_encoded
                tile_5_encoded.append(group.iloc[i + sequence_length - 1][features[8]])  # tile_5_encoded
                tile_6_encoded.append(group.iloc[i + sequence_length - 1][features[9]])  # tile_6_encoded
                tile_7_encoded.append(group.iloc[i + sequence_length - 1][features[10]])  # tile_7_encoded
                tile_8_encoded.append(group.iloc[i + sequence_length - 1][features[11]])  # tile_8_encoded
                tile_9_encoded.append(group.iloc[i + sequence_length - 1][features[12]])  # tile_9_encoded
                tile_10_encoded.append(group.iloc[i + sequence_length - 1][features[13]])  # tile_10_encoded
        else:
            # Handle shorter groups by padding
            if len(group) > 0:
                # Prepare padding and actual data
                padding_length = sequence_length - len(group)
                padded_data = np.full((padding_length, len(features)), np.nan)
                
                # Fill padding with custom padding values or replicated last values for specific features
                for idx, feature in enumerate(features):
                    if 'imo' in feature or 'classification' in feature:
                        padded_data[:, idx] = group[feature].iloc[-1]  
                    elif 'month_str' in feature:
                        padded_data[:, idx] = padding_month_str  
                    elif 'tile' in feature:
                        padded_data[:, idx] = padding_tile  
                    elif 'country_code_last_encoded' in feature:
                        padded_data[:, idx] = padding_country  
                    else:
                        print(f"Padding not defined for feature: {feature}")

                # Combine the actual data with padding
                actual_data = group[features].values
                full_sequence = np.vstack((padded_data, actual_data))
                X.append(full_sequence)

                # The target is set from the last row in the available data
                target = group[target_feature].iloc[-1]
                y.append(target)

                imo_encoded.append(group[features[0]].iloc[-1])
                classification_encoded.append(group[features[1]].iloc[-1])
                month_str_encoded.append(group[features[2]].iloc[-1])

                country_code_last_encoded.append(group[features[3]].values.tolist())  # include all available values

                tile_1_encoded.append(group[features[4]].iloc[-1])  # tile_1_encoded
                tile_2_encoded.append(group[features[5]].iloc[-1])  # tile_2_encoded
                tile_3_encoded.append(group[features[6]].iloc[-1])  # tile_3_encoded
                tile_4_encoded.append(group[features[7]].iloc[-1])  # tile_4_encoded
                tile_5_encoded.append(group[features[8]].iloc[-1])  # tile_5_encoded
                tile_6_encoded.append(group[features[9]].iloc[-1])  # tile_6_encoded
                tile_7_encoded.append(group[features[10]].iloc[-1]) # tile_7_encoded
                tile_8_encoded.append(group[features[11]].iloc[-1]) # tile_8_encoded
                tile_9_encoded.append(group[features[12]].iloc[-1]) # tile_9_encoded
                tile_10_encoded.append(group[features[13]].iloc[-1]) # tile_10_encoded
    
    return np.array(X), np.array(y), imo_encoded, classification_encoded, country_code_last_encoded, month_str_encoded, tile_1_encoded, tile_2_encoded, tile_3_encoded, tile_4_encoded, tile_5_encoded, tile_6_encoded, tile_7_encoded, tile_8_encoded, tile_9_encoded, tile_10_encoded

