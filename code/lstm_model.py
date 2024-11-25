import torch
import torch.nn as nn


class LSTMWithEmbeddings(nn.Module):

    """
    Implements an LSTM model with embedding layers, designed to process sequential data that includes 
    numerically-encoded categorical features.

    Args:
        num_imo_categories (int): Number of unique IMO categories.
        num_target_categories (int): Number of target categories.
        num_classification_categories (int): Number of classification categories.
        num_month_str_categories (int): Number of month string categories.
        num_tile_categories (int): Number of tile categories.
        embedding_dim_imo, embedding_dim_target, embedding_dim_classification, embedding_dim_month_str, embedding_dim_tile (int): Embedding dimensions.
        hidden_size (int): Number of features in the LSTM hidden layers.
        num_stacked_layers (int): Number of LSTM layers.
        num_classes (int): Number of output classes.
        padding_target, padding_month_str, padding_tile (int): Padding indices.
        num_tile_features (int): Number of tile features.
        dropout_rate (float): Dropout rate.
    """



    def __init__(self, num_imo_categories, num_target_categories, num_classification_categories, num_month_str_categories, 
                 num_tile_categories, embedding_dim_imo, embedding_dim_target, embedding_dim_classification, embedding_dim_month_str,
                 embedding_dim_tile, hidden_size, num_stacked_layers, num_classes, padding_target, padding_month_str, padding_tile,num_tile_features, dropout_rate):
        super(LSTMWithEmbeddings, self).__init__()

        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.num_tile_features = num_tile_features
        self.embedding_dim_tile = embedding_dim_tile
        
        # Embedding layers
        self.embedding_imo = nn.Embedding(num_imo_categories, embedding_dim_imo)
        self.embedding_target = nn.Embedding(num_target_categories, embedding_dim_target, padding_idx=padding_target)
        self.embedding_classification = nn.Embedding(num_classification_categories, embedding_dim_classification)
        self.embedding_month_str = nn.Embedding(num_month_str_categories, embedding_dim_month_str, padding_idx=padding_month_str)
        self.embedding_tiles = nn.Embedding(num_tile_categories, embedding_dim_tile, padding_idx=padding_tile)
        
        # Update LSTM input size
        self.lstm_input_size = embedding_dim_imo + embedding_dim_target + embedding_dim_classification + embedding_dim_month_str + num_tile_features * embedding_dim_tile
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_stacked_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate) # Dropout        
        self.fc = nn.Linear(hidden_size, num_classes)  # Final output layer
    
    def forward(self, x):
        batch_size = x.size(0)

        # Extract indices from the input tensor
        imo_indices = x[:, :, 0].long()
        target_indices = x[:, :, 3].long()
        classification_indices = x[:, :, 1].long()
        month_str_indices = x[:, :, 2].long()
        tile_indices = x[:, :, 4:14].long()  #  tiles are the last 5 columns in the input tensor
        
        # Embeddings
        imo_embedding = self.embedding_imo(imo_indices)
        target_embedding = self.embedding_target(target_indices)
        classification_embedding = self.embedding_classification(classification_indices)
        month_str_embedding = self.embedding_month_str(month_str_indices)
        tile_embeddings = self.embedding_tiles(tile_indices)

        # Reshape tile_embeddings from [batch, seq_len, 5, embedding_dim_tile] to [batch, seq_len, 5 * embedding_dim_tile]
        tile_embeddings = tile_embeddings.view(batch_size, -1, self.num_tile_features * self.embedding_dim_tile)
        
        # Concatenate embeddings to form LSTM input
        lstm_input = torch.cat((imo_embedding, target_embedding, classification_embedding, month_str_embedding, tile_embeddings), dim=2)

        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Apply dropout to the output of the last time step
        
        # Pass the last time step output through a fully connected layer to get the final class prediction
        output = self.fc(lstm_out)
        return output




