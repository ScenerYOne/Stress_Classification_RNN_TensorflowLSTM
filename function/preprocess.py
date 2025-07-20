import json
import pandas as pd
import os
import pickle
import numpy as np

class preprocess:
    def __init__(self, dataframe, columns):
        self.data = dataframe
        self.columns = columns
    
    def select_columns(self):
        """
        Selects the specified columns from the dataframe.
        """
        self.data = self.data[self.columns]
        return self.data

    def label_encoding(self, mapping_path='category_mapping.json'):
        """
        Encodes categorical columns in the dataframe using label encoding.
        Saves the mapping to a JSON file for later reference.
        
        Args:
            mapping_path (str): Path to save the category mapping JSON file.
        """
        
        mapping = {}
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                # Store the mapping before encoding
                categories = self.data[column].astype('category')
                mapping[column] = {str(i): cat for i, cat in enumerate(categories.cat.categories)}
                
                # Perform the encoding
                self.data[column] = categories.cat.codes
        
        # Save the mapping to a JSON file
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=4)
            
        return self.data, mapping

    def scale_data(self, scaler, exclude_columns=None, save_path='assets/scaler.pkl'):
        """
        Scales the data using the provided scaler, excluding specified columns.
        
        Args:
            scaler: A scaler object (e.g., MinMaxScaler, StandardScaler) from sklearn.
            exclude_columns: List of column names to exclude from scaling (e.g., 'stress', 'id').
            save_path: Path to save the fitted scaler object.
            
        Returns:
            pandas.DataFrame: The scaled dataframe with the same structure as the input.
        """
        
        # Default to empty list if None is provided
        if exclude_columns is None:
            exclude_columns = []
        
        # Store column names to preserve DataFrame structure
        columns = self.data.columns
        
        # Separate columns to scale and columns to exclude
        cols_to_scale = [col for col in columns if col not in exclude_columns]
        
        # Create a copy of the data
        scaled_df = self.data.copy()
        
        # Apply scaling only to columns that should be scaled
        if cols_to_scale:
            scaled_df[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the fitted scaler for later use
            with open(save_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        self.data = scaled_df
        
        return self.data
    def create_sequence(self, seq_length=5):
        """
        Creates sequences for each unique ID in the dataset.
        
        Args:
            seq_length (int): Length of each sequence.
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target stress values.
        """
        store_id = self.data['id'].unique()
        all_X = []
        all_y = []
        
        for i in store_id:
            user_data = self.data[self.data['id'] == i].reset_index(drop=True)
            if len(user_data) > seq_length:
                X, y = self._create_sequence(user_data, seq_length)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
        
        if not all_X:
            return np.array([]), np.array([])
        
        return np.vstack(all_X), np.concatenate(all_y)
    
    def _create_sequence(self, data, seq_length):
        """
        Helper method to create sequences from a dataframe for a single ID.
        
        Args:
            data (pd.DataFrame): Data for a single ID.
            seq_length (int): Length of each sequence.
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target stress values.
        """
        X = []
        y = []
        
        # Create a copy and drop 'id' column for sequences
        data_for_seq = data.drop(columns=['id']).copy()
        
        # Convert to numpy for easier slicing
        data_array = data_for_seq.values
        
        # Get the index of the stress column
        features = data_for_seq.columns.tolist()
        stress_idx = features.index('stress')
        
        for i in range(len(data) - seq_length):
            # Get the sequence
            seq = data_array[i:(i + seq_length), :]
            
            # Check if all stress values in the sequence are the same
            stress_values = seq[:, stress_idx]
            if len(set(stress_values)) == 1:  # All stress values are the same
                # Get the stress value for the next time step
                label = data_array[i + seq_length, stress_idx]
                X.append(seq)
                y.append(label)
        
        if not X:  # If no valid sequences were found
            return np.array([]), np.array([])
            
        # Convert to numpy arrays
        # For X, don't include the stress column for each time step
        X_no_stress = [seq[:, :stress_idx] for seq in X]  # Exclude the stress column
        
        return np.array(X_no_stress), np.array(y)