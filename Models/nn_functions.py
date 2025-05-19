# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:40:59 2023

@author: Oscar Ivan Calderon Hernandez Ph.D Student Politecnico di Torino
"""
"""------------------------------ Function and Class Main File for the Neural Network Definition and Trainning ------------------------------"""


"""
Basic Libraries
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
import random
import h5py
import empymod
import time

"""
Sklearn libraries 
"""

import sklearn.metrics as sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score


from scipy import interpolate

"""
Torch Libraries 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import tqdm
import copy



"""
Personal Libraries
"""
from Functions import em_functions

pi=math.pi

exp=math.exp(1)

i_num=1j

mu0=4*pi*(10**(-7))



#%%


"""
------------------------------ Revision 1.0 ------------------------------
"""

"""
Notes: 
Naming Scheme:


"""

#%%

#Some functions

def gpu_memory_info():
    print(f"Total allocated memory in the GPU: {torch.cuda.memory_allocated()/1024**2} MB")    
    print(f"Total reserved memory in the GPU: {torch.cuda.memory_reserved()/1024**2} MB")
    
def clear_gpu_memory():
    torch.cuda.empty_cache()

def Models_Subset(data, samples, training=False, train_ratio=0.8, seed=42):
    
    np.random.seed(seed)  # Ensure reproducibility
    
    random.seed(seed)
    
    data = np.array(data)  # Convert to NumPy array if it's a list
    
    indices = np.random.permutation(len(data))  # Shuffle indices

    split_idx = int(len(data) * train_ratio)
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    if training:
        
        train_idx=random.sample(train_idx.tolist(), samples)
        
        train_idx=sorted(train_idx)
        
        data=data[train_idx].tolist()
        
    else:
    
        test_idx=random.sample(test_idx.tolist(), samples)
        
        test_idx=sorted(test_idx)
        
        data=data[test_idx].tolist()

    return data


#%% Functions to prepare a MT dataset into a training set for a given neural network

class LSTM_Dataset(Dataset):
    def __init__(self, data, targets, lengths):
        self.data = data
        self.targets = targets
        self.lengths = lengths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.lengths[idx]


class Models_Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def collate_fn(batch):
    """
    Collate function to handle variable-length sequences.
    Returns:
        - X_padded: Padded input data of shape (batch_size, max_seq_len, input_size).
        - y_padded: Padded target data of shape (batch_size, max_seq_len).
        - seq_len: Tensor of sequence lengths of shape (batch_size,).
    """
    # Unpack the batch
    data, targets = zip(*batch)
    
    # Get sequence lengths
    seq_len = torch.tensor([len(x) for x in data], dtype=torch.long)
    
    # Pad sequences in the batch
    X_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    y_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return X_padded, y_padded, seq_len


def collate_fn_inversion(batch):
    """
    Collate function to handle variable-length sequences.
    Returns:
        - X_padded: Padded input data of shape (batch_size, max_seq_len, input_size).
        - y_padded: Padded target data of shape (batch_size, max_seq_len).
        - seq_len: Tensor of sequence lengths of shape (batch_size,).
    """
    # Unpack the batch
    data, targets = zip(*batch)
    
    # Get sequence lengths
    seq_len = torch.tensor([len(x) for x in data], dtype=torch.long)

    output_seq_len = torch.tensor([len(y) for y in targets], dtype=torch.long)
    
    # Pad sequences in the batch
    X_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    y_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return X_padded, y_padded, seq_len, output_seq_len


def Prepare_NN_Dataset(Models, input_feature_names, target_feature_names, normalization="None"):
    """
    Prepares the input and target data for neural network training.
    
    Args:
        Models: A list of profile objects containing impedance and depth data.
        input_feature_names: A list of feature names to extract from each profile.
        target_feature_names: A list of target feature names to extract from each profile.
        normalized: A boolean flag indicating whether to normalize the data. If True, data will be normalized.
        z_score: If True, Z-score normalization will be applied.
        min_max: If True, Min-Max normalization will be applied.
    
    Returns:
        X: A tensor of input features (normalized or not).
        y: A tensor of target values (normalized or not).
        indices: A list of tuples indicating the start and end indices for each profile.
    """
    Input_NN_Data = []
    Target_NN_Data = []

    for profile in Models:
        data_length = len(getattr(profile, input_feature_names[0]))
        
        # Pre-extract all the features for the profile to avoid repeated calls to getattr
        input_data = np.array([ [getattr(profile, feature)[data_point] for feature in input_feature_names] 
                               for data_point in range(data_length)])
        output_data = np.array([ [getattr(profile, feature)[data_point] for feature in target_feature_names] 
                                for data_point in range(data_length)])

        # Extend Input and Target Data
        Input_NN_Data.append(input_data)
        Target_NN_Data.append(output_data)

    
    # Convert lists of arrays into one big NumPy array
    Input_NN_Data = np.vstack(Input_NN_Data)
    Target_NN_Data = np.vstack(Target_NN_Data)
    
    # If normalization is requested
    if normalization.lower() != "none":
        
        if normalization=="z_score":
        
            # Compute mean and std for Z-score normalization
            Input_NN_Data_mean = np.mean(Input_NN_Data, axis=0)
            Input_NN_Data_std = np.std(Input_NN_Data, axis=0)
                
            Target_NN_Data_mean = np.mean(Target_NN_Data, axis=0)
            Target_NN_Data_std = np.std(Target_NN_Data, axis=0)
                
            # Normalize the dataset using Z-score formula (data - mean) / std
            Normalized_Input_NN_Data = (Input_NN_Data - Input_NN_Data_mean) / Input_NN_Data_std 
            Normalized_Output_NN_Data = (Target_NN_Data - Target_NN_Data_mean) / Target_NN_Data_std
                
            Input_NN_Data = Normalized_Input_NN_Data       
            Target_NN_Data = Normalized_Output_NN_Data
            
            Normalization_Parameters = {
                'input_mean': Input_NN_Data_mean.tolist(),
                'input_std': Input_NN_Data_std.tolist(),
                'target_mean': Target_NN_Data_mean.tolist(),
                'target_std': Target_NN_Data_std.tolist()
            }
            
        elif normalization.lower()=="min-max":           
            # Compute min and max for Min-Max normalization
            Input_NN_Data_min = np.min(Input_NN_Data, axis=0)
            Input_NN_Data_max = np.max(Input_NN_Data, axis=0)
            
            Target_NN_Data_min = np.min(Target_NN_Data, axis=0)
            Target_NN_Data_max = np.max(Target_NN_Data, axis=0)
            
            # Normalize the dataset using Min-Max formula (data - min) / (max - min)
            Normalized_Input_NN_Data = (Input_NN_Data - Input_NN_Data_min) / (Input_NN_Data_max - Input_NN_Data_min)
            Normalized_Output_NN_Data = (Target_NN_Data - Target_NN_Data_min) / (Target_NN_Data_max - Target_NN_Data_min)
            
            # Update the normalized data
            Input_NN_Data = Normalized_Input_NN_Data
            Target_NN_Data = Normalized_Output_NN_Data
            
            # Store normalization parameters
            Normalization_Parameters = {
            'input_min': Input_NN_Data_min.tolist(),
            'input_max': Input_NN_Data_max.tolist(),
            'target_min': Target_NN_Data_min.tolist(),
            'target_max': Target_NN_Data_max.tolist()
            }
    else:
        
        Normalization_Parameters=None
    
    # Convert to tensors for neural network input
    X = torch.tensor(Input_NN_Data, dtype=torch.float32)
    y = torch.tensor(Target_NN_Data, dtype=torch.float32)

    return X, y, Normalization_Parameters


def Prepare_NN_Dataset_by_Profile(Models, input_feature_names, target_feature_names, normalization="None"):
    """
    Prepares the input and target data for neural network training.
    
    Args:
        Models: A list of profile objects containing impedance and depth data.
        input_feature_names: A list of feature names to extract from each profile.
        target_feature_names: A list of target feature names to extract from each profile.
        normalization: A string indicating the type of normalization ("None", "z_Score", "Min-Max").
    
    Returns:
        X_profiles: A list of tensors, where each tensor corresponds to a profile's input features.
        y_profiles: A list of tensors, where each tensor corresponds to a profile's target values.
        Normalization_Parameters: A dictionary containing the computed normalization parameters.
    """
    Input_NN_Data = []
    Target_NN_Data = []
    
    # Step 1: Collect all data to compute global normalization parameters
    for profile in Models:
        data_length = len(getattr(profile, input_feature_names[0]))
        
        # Extract input and target features for the profile
        input_data = np.array([[getattr(profile, feature)[data_point] for feature in input_feature_names]
                               for data_point in range(data_length)])
        output_data = np.array([[getattr(profile, feature)[data_point] for feature in target_feature_names]
                                for data_point in range(data_length)])
        
        # Store profile data in a global list for computing statistics
        Input_NN_Data.append(input_data)
        Target_NN_Data.append(output_data)
    
    # Convert lists of arrays into one big NumPy array for global statistics
    Input_NN_Data_all = np.vstack(Input_NN_Data)
    Target_NN_Data_all = np.vstack(Target_NN_Data)
    
    Normalization_Parameters = None  # Default case

    # Step 2: Compute global normalization parameters
    if normalization.lower() != "none":
        
        if normalization.lower() == "z_score":
            # Compute mean and std across all profiles
            Input_mean = np.mean(Input_NN_Data_all, axis=0)
            Input_std = np.std(Input_NN_Data_all, axis=0)
            
            Target_mean = np.mean(Target_NN_Data_all, axis=0)
            Target_std = np.std(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_mean': Input_mean.tolist(),
                'input_std': Input_std.tolist(),
                'target_mean': Target_mean.tolist(),
                'target_std': Target_std.tolist()
            }

        elif normalization.lower() == "min-max":
            # Compute global min/max across all profiles
            Input_min = np.min(Input_NN_Data_all, axis=0)
            Input_max = np.max(Input_NN_Data_all, axis=0)
            
            Target_min = np.min(Target_NN_Data_all, axis=0)
            Target_max = np.max(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_min': Input_min.tolist(),
                'input_max': Input_max.tolist(),
                'target_min': Target_min.tolist(),
                'target_max': Target_max.tolist()
            }

    # Step 3: Normalize and store profiles individually
    X_profiles = []
    y_profiles = []

    for i, (input_data, target_data) in enumerate(zip(Input_NN_Data, Target_NN_Data)):
        
        if normalization.lower() == "z_score":
            input_data = (input_data - Input_mean) / Input_std
            target_data = (target_data - Target_mean) / Target_std

        elif normalization.lower() == "min-max":
            input_data = (input_data - Input_min) / (Input_max - Input_min)
            target_data = (target_data - Target_min) / (Target_max - Target_min)

        # Convert to PyTorch tensors and store as separate profiles
        X_profiles.append(torch.tensor(input_data, dtype=torch.float32))
        y_profiles.append(torch.tensor(target_data, dtype=torch.float32))

    #Sequence_Lengths = torch.tensor([len(profile) for profile in X_profiles])  # Store original lengths for padding
    max_length = max(len(profile) for profile in X_profiles)
    return X_profiles, y_profiles, Normalization_Parameters, max_length


def Prepare_NN_Dataset_by_Profile_Resistivity(Models, input_feature_names, target_feature_names, normalization="None"):
    """
    Prepares the input and target data for neural network training.
    
    Args:
        Models: A list of profile objects containing impedance and depth data.
        input_feature_names: A list of feature names to extract from each profile.
        target_feature_names: A list of target feature names to extract from each profile.
        normalization: A string indicating the type of normalization ("None", "z_Score", "Min-Max").
    
    Returns:
        X_profiles: A list of tensors, where each tensor corresponds to a profile's input features.
        y_profiles: A list of tensors, where each tensor corresponds to a profile's target values.
        Normalization_Parameters: A dictionary containing the computed normalization parameters.
    """
    Input_NN_Data = []
    Target_NN_Data = []
    
    # Step 1: Collect all data to compute global normalization parameters
    for profile in Models:
        data_length = len(getattr(profile, input_feature_names[0]))
        
        # Extract input and target features for the profile
        input_data = np.array([[getattr(profile, feature)[data_point] for feature in input_feature_names]
                               for data_point in range(data_length)])

        data_length_output = len(getattr(profile, target_feature_names[0]))  

        output_data = np.array([[getattr(profile, feature)[data_point] for feature in target_feature_names]
                                for data_point in range(data_length_output)])
        
        # Store profile data in a global list for computing statistics
        Input_NN_Data.append(input_data)
        Target_NN_Data.append(output_data)
    
    # Convert lists of arrays into one big NumPy array for global statistics
    Input_NN_Data_all = np.vstack(Input_NN_Data)
    Target_NN_Data_all = np.vstack(Target_NN_Data)
    
    Normalization_Parameters = None  # Default case

    # Step 2: Compute global normalization parameters
    if normalization.lower() != "none":
        
        if normalization.lower() == "z_score":
            # Compute mean and std across all profiles
            Input_mean = np.mean(Input_NN_Data_all, axis=0)
            Input_std = np.std(Input_NN_Data_all, axis=0)
            
            Target_mean = np.mean(Target_NN_Data_all, axis=0)
            Target_std = np.std(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_mean': Input_mean.tolist(),
                'input_std': Input_std.tolist(),
                'target_mean': Target_mean.tolist(),
                'target_std': Target_std.tolist()
            }

        elif normalization.lower() == "min-max":
            # Compute global min/max across all profiles
            Input_min = np.min(Input_NN_Data_all, axis=0)
            Input_max = np.max(Input_NN_Data_all, axis=0)
            
            Target_min = np.min(Target_NN_Data_all, axis=0)
            Target_max = np.max(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_min': Input_min.tolist(),
                'input_max': Input_max.tolist(),
                'target_min': Target_min.tolist(),
                'target_max': Target_max.tolist()
            }

    # Step 3: Normalize and store profiles individually
    X_profiles = []
    y_profiles = []

    for i, (input_data, target_data) in enumerate(zip(Input_NN_Data, Target_NN_Data)):
        
        if normalization.lower() == "z_score":
            input_data = (input_data - Input_mean) / Input_std
            target_data = (target_data - Target_mean) / Target_std

        elif normalization.lower() == "min-max":
            input_data = (input_data - Input_min) / (Input_max - Input_min)
            target_data = (target_data - Target_min) / (Target_max - Target_min)

        # Convert to PyTorch tensors and store as separate profiles
        X_profiles.append(torch.tensor(input_data, dtype=torch.float32))
        y_profiles.append(torch.tensor(target_data, dtype=torch.float32))

    #Sequence_Lengths = torch.tensor([len(profile) for profile in X_profiles])  # Store original lengths for padding
    max_length = max(len(profile) for profile in X_profiles)
    max_length_output = max(len(profile) for profile in y_profiles)
    return X_profiles, y_profiles, Normalization_Parameters, max_length, max_length_output


def Prepare_NN_Dataset_by_Profile_with_Padding(Models, input_feature_names, target_feature_names, normalization="None"):
    """
    Prepares the input and target data for neural network training.
    
    Args:
        Models: A list of profile objects containing impedance and depth data.
        input_feature_names: A list of feature names to extract from each profile.
        target_feature_names: A list of target feature names to extract from each profile.
        normalization: A string indicating the type of normalization ("None", "z_Score", "Min-Max").
    
    Returns:
        X_profiles: A list of tensors, where each tensor corresponds to a profile's input features.
        y_profiles: A list of tensors, where each tensor corresponds to a profile's target values.
        Normalization_Parameters: A dictionary containing the computed normalization parameters.
    """
    Input_NN_Data = []
    Target_NN_Data = []
    
    # Step 1: Collect all data to compute global normalization parameters
    for profile in Models:
        data_length = len(getattr(profile, input_feature_names[0]))


        
        # Extract input and target features for the profile
        input_data = np.array([[getattr(profile, feature)[data_point] for feature in input_feature_names]
                               for data_point in range(data_length)])
        output_data = np.array([[getattr(profile, feature)[data_point] for feature in target_feature_names]
                                for data_point in range(data_length)])
        
        # Store profile data in a global list for computing statistics
        Input_NN_Data.append(input_data)
        Target_NN_Data.append(output_data)
    
    # Convert lists of arrays into one big NumPy array for global statistics
    Input_NN_Data_all = np.vstack(Input_NN_Data)
    Target_NN_Data_all = np.vstack(Target_NN_Data)
    
    Normalization_Parameters = None  # Default case

    # Step 2: Compute global normalization parameters
    if normalization.lower() != "none":
        
        if normalization.lower() == "z_score":
            # Compute mean and std across all profiles
            Input_mean = np.mean(Input_NN_Data_all, axis=0)
            Input_std = np.std(Input_NN_Data_all, axis=0)
            
            Target_mean = np.mean(Target_NN_Data_all, axis=0)
            Target_std = np.std(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_mean': Input_mean.tolist(),
                'input_std': Input_std.tolist(),
                'target_mean': Target_mean.tolist(),
                'target_std': Target_std.tolist()
            }

        elif normalization.lower() == "min-max":
            # Compute global min/max across all profiles
            Input_min = np.min(Input_NN_Data_all, axis=0)
            Input_max = np.max(Input_NN_Data_all, axis=0)
            
            Target_min = np.min(Target_NN_Data_all, axis=0)
            Target_max = np.max(Target_NN_Data_all, axis=0)
            
            Normalization_Parameters = {
                'input_min': Input_min.tolist(),
                'input_max': Input_max.tolist(),
                'target_min': Target_min.tolist(),
                'target_max': Target_max.tolist()
            }

    # Step 3: Normalize and store profiles individually
    X_profiles = []
    y_profiles = []

    for i, (input_data, target_data) in enumerate(zip(Input_NN_Data, Target_NN_Data)):
        
        if normalization.lower() == "z_score":
            input_data = (input_data - Input_mean) / Input_std
            target_data = (target_data - Target_mean) / Target_std

        elif normalization.lower() == "min-max":
            input_data = (input_data - Input_min) / (Input_max - Input_min)
            target_data = (target_data - Target_min) / (Target_max - Target_min)

        # Convert to PyTorch tensors and store as separate profiles
        X_profiles.append(torch.tensor(input_data, dtype=torch.float32))
        y_profiles.append(torch.tensor(target_data, dtype=torch.float32))

    Sequence_Lengths = torch.tensor([len(profile) for profile in X_profiles])  # Store original lengths for padding
    X_profiles_Padded = pad_sequence(X_profiles, batch_first=True, padding_value=0)  # Pad with zeros
    y_Profiles_Padded = pad_sequence(y_profiles, batch_first=True, padding_value=0)  # Pad with zeros

    return X_profiles_Padded, y_Profiles_Padded, Normalization_Parameters, Sequence_Lengths


#%%


"""
Neural Network Models
"""

class Light_DNN(nn.Module):
    def __init__(self, input_feature_names, target_feature_names):
        super(Light_DNN, self).__init__()
        
        # Calculate input size dynamically based on the number of features
        input_size = len(input_feature_names)
        
        output_size = len(target_feature_names)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 24),  # Dynamically use the input size
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, output_size),
        )
    
    def forward(self, x):
        return self.network(x)


class HybridCNN(nn.Module):
    def __init__(self, input_feature_names, target_feature_names):
        super(HybridCNN, self).__init__()
        
        # Calculate input size dynamically based on the number of features
        input_size = len(input_feature_names)
        
        output_size = len(target_feature_names)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten for dense layers
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(32*input_size, 32),  # Dynamically calculate based on input size
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)  # Final output
        )

    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x


class Enhanced_HybridCNN(nn.Module):
    def __init__(self, input_feature_names, target_feature_names):
        super(Enhanced_HybridCNN, self).__init__()
        
        # Calculate input and output sizes dynamically
        input_size = len(input_feature_names)
        output_size = len(target_feature_names)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),  # Batch normalization
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),  # Larger kernel size for broader feature extraction
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling to reduce the feature map to (batch_size, 32)
            nn.Flatten()  # Flatten for dense layers
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(32, 16),  # Reduce the size to keep it lightweight
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(16, output_size)  # Final output
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x


"""
LSTM Network Models
"""

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualCNN, self).__init__()
        
        """
        #self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        #self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        
        """

        self.conv1_small = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_medium = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv1_large = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

        self.conv2 = nn.Conv1d(out_channels * 3, out_channels, kernel_size=3, padding=1)

        
        # Ensure the residual connection matches dimensions
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        residual = self.residual_conv(x)  # Transform if needed
       
        """ 
        x = F.relu(self.conv1(x))  # No inplace=True
        x = self.conv2(x)  
        x = x + residual  # No in-place modification
        
        """
        x_small = F.relu(self.conv1_small(x))
        x_medium = F.relu(self.conv1_medium(x))
        x_large = F.relu(self.conv1_large(x))

        x = torch.cat([x_small, x_medium, x_large], dim=1)  # Concatenate along channels
        x = self.conv2(x)

        return F.relu(x)   # Final activation


class Hybrid_CNN_LSTM_Residual(nn.Module):
    def __init__(self, input_feature_names, target_feature_names, cnn_channels=[32, 64], hidden_size=256, num_layers=8, dropout=0.2, bidirectional=True):
        super(Hybrid_CNN_LSTM_Residual, self).__init__()
        
        input_size = len(input_feature_names)
        output_size = len(target_feature_names)
        
        # CNN with Residual Connections
        self.res_cnn1 = ResidualCNN(input_size, cnn_channels[0])
        self.res_cnn2 = ResidualCNN(cnn_channels[0], cnn_channels[1])
        self.res_cnn3 = ResidualCNN(cnn_channels[1], cnn_channels[1]) 

        # LSTM layers
        self.lstm = nn.LSTM(cnn_channels[1], hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        
    def forward(self, x, lengths):
        x = x.transpose(1, 2)  # Convert to (batch_size, input_size, seq_length)

        # Pass through Residual CNN layers
        x = self.res_cnn1(x)
        x = self.res_cnn2(x)
        x = self.res_cnn3(x)
        
        x = x.transpose(1, 2)  # Convert back to (batch_size, seq_length, channels)
        
        lengths = lengths.to(x.device)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        
        out = self.fc(out)  # Final output
        return out


class Hybrid_CNN_LSTM_Residual_Inversion(nn.Module):
    def __init__(self, input_feature_names, target_feature_names, cnn_channels=[32, 64], hidden_size=32, num_layers=2, dropout=0.2, bidirectional=True):
        super(Hybrid_CNN_LSTM_Residual_Inversion, self).__init__()

        input_size = len(input_feature_names)
        output_size = len(target_feature_names)

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers  # Store num_layers as an instance variable

        # CNN Layers
        self.res_cnn1 = ResidualCNN(input_size, cnn_channels[0])
        self.res_cnn2 = ResidualCNN(cnn_channels[0], cnn_channels[1])
        self.res_cnn3 = ResidualCNN(cnn_channels[1], cnn_channels[1])

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # Decoder LSTM (predicts one timestep at a time)
        self.decoder_lstm = nn.LSTM(
            input_size=output_size,  # Decoder input is the output features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Decoder is not bidirectional
            dropout=dropout
        )

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, input_lengths, output_len):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)  # Convert (batch, seq_len, features) -> (batch, features, seq_len)

        # CNN Feature Extraction
        x = self.res_cnn1(x)
        x = self.res_cnn2(x)
        x = self.res_cnn3(x)

        x = x.transpose(1, 2)  # Convert back to (batch, seq_len, features)

        # Encoder LSTM
        input_lengths = input_lengths.to(x.device)
        packed_x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.encoder_lstm(packed_x)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Adjust hidden state for decoder (handle bidirectional case)
        if self.bidirectional:
            # Take only the forward hidden state (first half of the bidirectional hidden state)
            hidden = hidden[:self.num_layers]  # Use self.num_layers
            cell = cell[:self.num_layers]      # Use self.num_layers

        # Ensure output_len is properly handled (tensor or list)
        if isinstance(output_len, torch.Tensor):
            output_len_list = output_len.tolist()
        else:
            output_len_list = [output_len] * batch_size

        max_output_len = max(output_len_list)

        # Initialize decoder input (zero vector of shape (batch, 1, output_size))
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(x.device)

        # Store outputs
        outputs = []

        # Decode sequence step by step
        for t in range(max_output_len):
            decoder_out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            decoder_output = self.fc(decoder_out)  # (batch, 1, output_size)

            outputs.append(decoder_output)

            # Autoregressive step (use last prediction as next input)
            decoder_input = decoder_output

        # Stack outputs to get shape (batch, max_output_len, output_size)
        outputs = torch.cat(outputs, dim=1)

        # Mask shorter sequences in the batch
        masked_outputs = [outputs[i, :output_len_list[i], :] for i in range(batch_size)]

        # Pad back to tensor
        outputs = torch.nn.utils.rnn.pad_sequence(masked_outputs, batch_first=True)

        return outputs


class Hybrid_CNN_LSTM(nn.Module):
    def __init__(self, input_feature_names, target_feature_names, cnn_channels=[32, 64], hidden_size=256, num_layers=8, dropout=0.2, bidirectional=True):
        super(Hybrid_CNN_LSTM, self).__init__()
        
        input_size = len(input_feature_names)
        output_size = len(target_feature_names)
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, out_channels=cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, padding=1)

        # LSTM layers
        self.lstm = nn.LSTM(cnn_channels[1], hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        
    def forward(self, x, lengths):
        # Input is in shape (batch_size, seq_length, input_size), we need to transpose it

        x = x.transpose(1, 2)  # Convert to (batch_size, input_size, seq_length)

        # Pass through CNN layers
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))  # No pooling to preserve sequence length

        # Reshape to (batch_size, seq_len, channels) for LSTM
        x = x.transpose(1, 2)  # Convert back to (batch_size, seq_length, channels)
        
        lengths = lengths.to(x.device)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_out, _ = self.lstm(packed_x)
        # Unpack the sequence to pad the output to the max sequence length in the batch
        out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)

        # Final output (output_size predictions per sample)
        out = self.fc(out)

        # Return the output
        return out
 

"""
Loss Functions
"""

class Interval_Weighted_Loss(nn.Module):
    def __init__(self, base_loss="L2"):
        super().__init__()
        self.base_loss = base_loss

    def compute_interval_weights(self, y_true_log):
        # Use only the last feature as depth (log10)
        y_log_depth = y_true_log[:, :, -1]  # Shape: (B, T)
        y_linear_depth = torch.pow(10.0, y_log_depth)  # Convert to linear scale

        B, T = y_linear_depth.shape
        intervals = torch.zeros_like(y_linear_depth)

        # Compute intervals
        intervals[:, 0] = y_linear_depth[:, 1] - y_linear_depth[:, 0]
        intervals[:, 1:-1] = 0.5 * (y_linear_depth[:, 2:] - y_linear_depth[:, :-2])
        intervals[:, -1] = y_linear_depth[:, -1] - y_linear_depth[:, -2]

        # Clamp intervals to avoid divide-by-zero
        intervals = torch.clamp(intervals, min=1e-6)

        total_thickness = intervals.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        weights_depth = intervals / total_thickness  # Normalize

        # Expand weights for all features
        B, T, F = y_true_log.shape
        weights = torch.ones((B, T, F), device=y_true_log.device, dtype=y_true_log.dtype)
        weights[:, :, -1] = weights_depth  # Only depth gets interval-based weighting

        return weights

    def forward(self, y_pred, y_true):

        eps = torch.finfo(y_true.dtype).eps  # Prevent numerical instability
        abs_diff = (y_pred - y_true).abs()  # Absolute error

        if self.base_loss == "L2":
            loss_per_sample = abs_diff ** 2
        else:
            loss_per_sample = abs_diff

        weights = self.compute_interval_weights(y_true)
        weighted_loss = (loss_per_sample * weights).sum() / weights.sum()

        return torch.sqrt(weighted_loss) if self.base_loss == "L2" else weighted_loss


class Decade_Weighted_Loss(nn.Module):
    def __init__(self, base_loss="L2"):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, y_pred, y_true):
        eps = torch.finfo(y_true.dtype).eps  # Prevent numerical instability
        abs_true = y_true.abs().clamp_min(eps)  # Ensure positive values
        abs_diff = (y_pred - y_true).abs()  # Absolute error

        # Compute base loss per sample
        loss_per_sample = abs_diff if self.base_loss == "L1" else abs_diff.square()

        # Identify the shape of the batch (assuming batch_size x num_samples_per_profile)
        batch_size = y_true.shape[0]

        weights = torch.ones_like(y_true)  # Default weights = 1 for all features

        # Compute weights **per profile** in the batch
        for i in range(batch_size):
            decade_idx = torch.floor(abs_true[i]).long()  # Compute decade indices for profile i

            # Get unique decades and counts **for this profile**
            unique_decades, counts_per_decade = torch.unique(decade_idx, return_counts=True)

            # Compute weights per decade for this profile
            decade_weights = 1.0 / counts_per_decade.float()

            # Assign weights to corresponding samples
            weights[i, :, -1] = decade_weights[torch.searchsorted(unique_decades, decade_idx)].view(-1)

        # Compute weighted + unweighted loss
        weighted_loss = torch.sum(loss_per_sample * weights) / torch.sum(weights)

        return weighted_loss.sqrt() if self.base_loss == "L2" else weighted_loss


class Normalized_Losses_Optimized(nn.Module):
    def __init__(self,Loss):
        super(Normalized_Losses_Optimized, self).__init__()
        self.loss=Loss

    def forward(self, y_pred, y_true):

        eps = torch.finfo(y_true.dtype).eps  # Dynamic epsilon based on dtype
        abs_true = torch.abs(y_true).clamp_min(eps) # Avoid division by zero
        abs_diff = torch.abs(y_pred - y_true)

        # Compute losses
        losses = {
            "Local_L1": torch.mean(abs_diff / abs_true),
            "Local_L2": torch.sqrt(torch.mean((abs_diff / abs_true) ** 2)),
            "Global_L1": torch.sum(abs_diff) / torch.sum(abs_true),
            "Global_L2": torch.sqrt(torch.sum(abs_diff ** 2) / torch.sum(abs_true ** 2))
        }

        # Select the required loss
        if self.loss == "Local_L1":
            mean_abs_diff = torch.mean(abs_diff / abs_true)
            return mean_abs_diff

        elif self.loss == "Local_L2":
            mean_sq_diff = torch.mean((abs_diff / abs_true).square())
            return torch.sqrt(mean_sq_diff)

        elif self.loss == "Global_L1":
            sum_abs_diff = torch.sum(abs_diff)
            sum_abs_true = torch.sum(abs_true)
            return sum_abs_diff / sum_abs_true

        elif self.loss == "Global_L2":
            sum_sq_diff = torch.sum(abs_diff.square())
            sum_sq_true = torch.sum(abs_true.square())

            return torch.sqrt(sum_sq_diff / sum_sq_true)
        
        else:
            return None
        

class Hybrid_Gradient_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Hybrid loss function combining point-wise error and shape similarity.
        Args:
            alpha: Weight for point-wise loss (MAE/MSE).
            beta: Weight for shape loss (Gradient Loss).
        """
        super(Hybrid_Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae_loss = nn.L1Loss()

    def forward(self, y_pred, y_true, X):
        """
        Computes the loss.

        Args:
            y_pred: (B, T, F) - Predicted depth points.
            y_true: (B, T, F) - Ground truth depth points.
            X: (B, T, F) - Input tensor, where the first column is resistance values.
        """
        # 1. **Point-wise MAE Loss**
        mae_loss = self.mae_loss(y_pred, y_true)

        # 2. **Non-Uniform Gradient-Based Shape Loss**
        x_values = X[:, :, -1]  # First column is resistance (B, T)

        # Compute finite differences (gradients) using non-uniform spacing
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # (B, T-1, F)
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]  # (B, T-1, F)
        
        dx = x_values[:, 1:] - x_values[:, :-1]  # (B, T-1)

        # Avoid division by zero
        dx = torch.where(dx == 0, torch.tensor(1e-8, device=dx.device, dtype=dx.dtype), dx)
        dx = dx.unsqueeze(-1)  # Expand for broadcasting (B, T-1, 1)

        grad_pred = dy_pred / dx
        grad_true = dy_true / dx

        grad_loss = self.mae_loss(grad_pred, grad_true)  # Compare normalized gradients
        
        # Dynamic weighting of gradient loss based on gradient mismatch
        grad_diff = torch.abs(grad_pred - grad_true).mean()
        dynamic_beta = torch.clamp(grad_diff * 10, min=0.5, max=1.5)  # Adjust beta dynamically
        
        total_loss = self.alpha * mae_loss + dynamic_beta * grad_loss

        return total_loss
    


#%%
"""
Training Functions
"""

def train_model(model, X, y, train_size=0.8,  n_epochs=1000, batch_size=256, lr=0.0001, early_stop_limit=50, seed=42, loss_mode='MSE', optimizer_mode='Adam', lr_mode='None', device='cpu'):
    """
    Train a neural network model with the option of early stopping and learning rate scheduling.
    
    Args:
        model (nn.Module): The model to be trained.
        X (torch.Tensor): Input data (features).
        y (torch.Tensor): Target data (labels).
        n_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        early_stop_limit (int): Number of epochs without improvement to trigger early stopping.
        seed (int): Random seed for data shuffling and splitting.
        loss_mode (str): Loss function selection. Options: 'mse', 'mae', 'huber'.
        optimizer_mode (str): Optimizer selection. Options: 'adam', 'sgd', 'rmsprop'.
        lr_mode (str): Learning rate strategy. Options: 'onecycle', 'cosine', 'step', 'none'.
        device (str): Device for training (e.g., 'cpu' or 'cuda').
        
    
    Returns:
        model (nn.Module): The trained model.
        history (list): A list containing the MSE values for each epoch.
    """

    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X.cpu(), y.cpu(), train_size=train_size, random_state=seed, shuffle=True)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    # Model, loss function, optimizer, and scheduler
    model = model.to(device)
    
    # Select Loss Function
    loss_functions = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss()
    }
    loss_fn = loss_functions.get(loss_mode.lower(), torch.nn.MSELoss())  # Default to MSELoss

    # Select Optimizer
    optimizer_options = {
        'adam': optim.Adam(model.parameters(), lr=lr),
        'sgd': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'rmsprop': optim.RMSprop(model.parameters(), lr=lr)
    }
    optimizer = optimizer_options.get(optimizer_mode.lower(), optim.Adam(model.parameters(), lr=lr))  # Default to Adam

    # Select Learning Rate Scheduler
    scheduler_options = {
        'onecycle': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10, steps_per_epoch=len(X_train) // batch_size, epochs=n_epochs),
        'cosinewr': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2),
        'cosinelr': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6),
        'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5),
        'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
        'none': None  # No scheduler
    }
    
    scheduler = scheduler_options.get(lr_mode.lower(), None)  # Default to no scheduler

    # Training parameters
    best_eval_loss = float('inf')
    best_weights = None
    history = {'train_loss': [], 'val_loss': [], 'lr_history':[], 
               "train_MAE_error":[], "train_R2_error":[],
               "val_MAE_error":[], "val_R2_error":[]}
    early_stop_count = 0
    
    # Training loop with early stopping
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_mae = 0
        epoch_r2 = 0
        batch_start = torch.arange(0, len(X_train), batch_size)

        with tqdm.tqdm(batch_start, unit="batch") as bar:
            bar.set_description(f"Epoch {epoch+1}")
            for start in bar:
                # Batch preparation
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_losses.append(loss.item())

                # Calculate MAE and R² for training data
                mae = mean_absolute_error(y_batch.detach().cpu(), y_pred.detach().cpu())  # Detach tensors before converting to numpy
                r2 = r2_score(y_batch.detach().cpu(), y_pred.detach().cpu())  # Detach tensors before converting to numpy
            

                epoch_mae += mae
                epoch_r2 += r2
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
                optimizer.step()
            

            # Update scheduler
            #if scheduler:
             #   scheduler.step()
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        history['lr_history'].append(current_lr)
        
        #print(f"Epoch {epoch + 1}: Learning Rate = {current_lr:.6e}")
        
        # Compute average training loss for the epoch
        train_loss = sum(epoch_losses) / len(epoch_losses)

        train_mae = epoch_mae / len(batch_start)
        train_r2 = epoch_r2 / len(batch_start)
        
        history['train_loss'].append(train_loss)
        history['train_MAE_error'].append(train_mae)
        history['train_R2_error'].append(train_r2)
  

        # Evaluate on test data
        model.eval()
        eval_mae = 0
        eval_r2 = 0
        with torch.no_grad():
            y_pred = model(X_test)
            validation_loss = loss_fn(y_pred, y_test).item()

            # Calculate MAE and R² for validation data
            mae = mean_absolute_error(y_test.detach().cpu(), y_pred.detach().cpu())  # Detach tensors before converting to numpy
            r2 = r2_score(y_test.detach().cpu(), y_pred.detach().cpu())  # Detach tensors before converting to numpy

            eval_mae += mae
            eval_r2 += r2
            
            val_mae = eval_mae / len(batch_start)
            val_r2 = eval_r2 / len(batch_start)

            history['val_loss'].append(validation_loss)
            history['val_MAE_error'].append(val_mae)
            history['val_R2_error'].append(val_r2)
            #print(f"Epoch {epoch + 1}: {loss_mode} = {validation_loss:.6f}, Train MAE = {train_mae:.6f}, Eval MAE = {val_mae:.6f}, Train R² = {train_r2:.6f}, Eval R² = {val_r2:.6f}, Best {loss_mode} = {best_eval_loss:.6f}")
            # Learning Rate Scheduler Step
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(validation_loss)  # Adjust LR based on validation loss
                    print("---------------------- Learning Rate: "+ str(scheduler.get_last_lr()) +"--------------------")
                else:
                    scheduler.step()  # Standard schedulers
                    print("---------------------- Learning Rate: "+ str(scheduler.get_last_lr()) +"--------------------")
            else:
                
                print(f"---------------------- Learning Rate: {optimizer.param_groups[0]['lr']} --------------------")
            # Print epoch results
            print(f"Epoch {epoch + 1}: Training {loss_mode} Loss= {train_loss:.6f}, Validation {loss_mode} Loss= {validation_loss:.6f}, Best {loss_mode} Loss= {best_eval_loss:.6f}")
            # Save best model weights
            if validation_loss < best_eval_loss:
                best_eval_loss = validation_loss
                best_weights = copy.deepcopy(model.state_dict())
                early_stop_count = 0  # Reset early stopping counter
            else:
                early_stop_count += 1

        # Early stopping
        if early_stop_count >= early_stop_limit:
            print("Early stopping triggered.")
            break

    # Restore the best model weights
    model.load_state_dict(best_weights)
    print(f"Best {loss_mode} achieved: {best_eval_loss}")
    
    return model, history


#%%
"""
Predicting Functions: Functions to make the prediction of the cumulative model
"""

def Rescaling_Prediction_Multiparameter(model, models_test, metadata, device='cpu'):
    """
    Evaluate the model on unseen data with optional normalization/denormalization.
    
    Args:
        model (nn.Module): The trained neural network model.
        models_test (list): A list of test profile objects containing input data.
        metadata (dict): Metadata containing feature names, normalization parameters, etc.
        device (str): Device to use for evaluation ('cpu' or 'cuda').
        
    Returns:
        None: Updates each profile in models_test with evaluated results.
    """
    input_feature_names = metadata['training_input_features']
    prediction_features_names = metadata['prediction_features']
    normalization = metadata['normalization']

    # Switch model to evaluation mode
    model.eval()

    for profile in models_test:
        data_length = len(getattr(profile, input_feature_names[0]))

        # Collect and store input data for the profile
        profile_input_data = np.array([ [getattr(profile, feature)[data_point] for feature in input_feature_names] 
                                       for data_point in range(data_length)])

        profile.NN_Input_Data = profile_input_data

        # Normalize if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                profile.NN_Input_Data = (profile.NN_Input_Data - metadata['normalization_params']['input_mean']) / metadata['normalization_params']['input_std']
            elif normalization.lower() == "min_max":
                profile.NN_Input_Data = (profile.NN_Input_Data - metadata['normalization_params']['input_min']) / (metadata['normalization_params']['input_max']- metadata['normalization_params']['input_min'])

        # Move test data to the same device as the model
        profile.NN_Input_Data = torch.tensor(profile.NN_Input_Data, dtype=torch.float32).to(device)

        # Perform model prediction
        with torch.no_grad():
            profile.NN_Prediction = model(profile.NN_Input_Data)
            
        # Convert prediction back to numpy
        profile.NN_Prediction = profile.NN_Prediction.cpu().numpy()

        # Denormalize the predictions if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                profile.NN_Prediction= (profile.NN_Prediction * metadata['normalization_params']['target_std']) + metadata['normalization_params']['target_mean']
            elif normalization.lower() == "min_max":
                profile.NN_Prediction = (profile.NN_Prediction * (metadata['normalization_params']['target_max']- metadata['normalization_params']['target_min']))  + metadata['normalization_params']['target_min']
                
        profile.Predicted_Model = {prediction_features_names[i]: profile.NN_Prediction[:, i] for i in range(profile.NN_Prediction.shape[1])}
        
        if metadata['logarithmic_data'].lower() !="none":
            if metadata['logarithmic_data'].lower() =="natural":
                profile.Predicted_Model['Predicted_Depth']=np.expm1(profile.Predicted_Model['Predicted_Depth'])
                
            if metadata['logarithmic_data'].lower() =="base_10":
                profile.Predicted_Model['Predicted_Depth']=10**(profile.Predicted_Model['Predicted_Depth'])
        
        # Enforce positive thickness adjustments
        for j in range(1, data_length):
            thickness = profile.Predicted_Model['Predicted_Depth'][j] - profile.Predicted_Model['Predicted_Depth'][j - 1]
            if thickness <= 0:
                profile.Predicted_Model['Predicted_Depth'][j] += abs(thickness) + 1
                
        if len(metadata['prediction_features']) == 1 and metadata['prediction_features'][0] == "Predicted_Depth":
            profile.Predicted_Model['Predicted_Resistance'] = profile.Resistance_Data

        if len(metadata['prediction_features']) != 1 and "Predicted_Layered_Resistivity" in metadata['prediction_features']:

            profile.Predicted_Model['Predicted_Resistance'] = profile.Resistance_Data
            
        profile.Predicted_Model['Predicted_Layered_Resistivity']=em_functions.Layered_Resistivity_from_harmonic_resistance(profile.Predicted_Model['Predicted_Resistance'], profile.Predicted_Model['Predicted_Depth'])



def Rescaling_Prediction_Multiparameter_LSTM(model, models_test, metadata, device='cpu'):
    """
    Evaluate the LSTM model on unseen data with optional normalization/denormalization.
    
    Args:
        model (nn.Module): The trained LSTM neural network model.
        models_test (list): A list of test profile objects containing input data.
        metadata (dict): Metadata containing feature names, normalization parameters, max sequence length, etc.
        device (str): Device to use for evaluation ('cpu' or 'cuda').
        
    Returns:
        None: Updates each profile in models_test with evaluated results.
    """
    input_feature_names = metadata['training_input_features']
    output_features_names = metadata['training_output_features']
    prediction_features_names = metadata['prediction_features']
    normalization = metadata['normalization']
    normalization_params = metadata['normalization_params']
    max_seq_length = metadata['Max_Sequence_Length']  # Get the max sequence length from training

    # Switch model to evaluation mode
    model.eval()

    for profile in models_test:
        data_length = len(getattr(profile, input_feature_names[0]))

        # Collect and store input data for the profile
        profile_input_data = np.array([
            [getattr(profile, feature)[data_point] for feature in input_feature_names]
            for data_point in range(data_length)
        ])

        # Normalize input data if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                profile_input_data = (
                    (profile_input_data - np.array(normalization_params['input_mean'])) / 
                    np.array(normalization_params['input_std'])
                )
            elif normalization.lower() == "min_max":
                profile_input_data = (
                    (profile_input_data - np.array(normalization_params['input_min'])) / 
                    (np.array(normalization_params['input_max']) - np.array(normalization_params['input_min']))
                )

        # Convert to tensor
        
        profile_input_data = torch.tensor(profile_input_data, dtype=torch.float32)

        # Pad to max sequence length
        padded_input = torch.zeros((max_seq_length, profile_input_data.shape[1]))  # Pre-allocate with padding value (0)
        padded_input[:min(data_length, max_seq_length), :] = profile_input_data[:max_seq_length, :]
        #Move to device (add batch dimension)
        
        padded_input = padded_input.unsqueeze(0).to(device)
        
        if metadata['network_params'].get("type") is not None and metadata['network_params']['type']=="Transformer":
            
            batch_size=1
          
            with torch.no_grad():
                seq_lengths = torch.tensor([min(data_length, max_seq_length)], dtype=torch.int64).to(device)
                key_padding_mask = torch.arange(max_seq_length, device=device).expand(batch_size, max_seq_length) >= seq_lengths.unsqueeze(0)
                
                model_output = model(padded_input, key_padding_mask.to(device))      
            
        else:
        
            # Perform model prediction
            with torch.no_grad():
                seq_lengths = torch.tensor([min(data_length, max_seq_length)], dtype=torch.int64).to(device)
                model_output = model(padded_input, seq_lengths)

            # Remove batch dimension and convert to numpy
        model_output = model_output.squeeze(0).cpu().numpy()    
        
        if len(model_output.shape) == 1:
            model_output = model_output.reshape(-1, 1)
            
        
        # Denormalize the predictions if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                model_output = (
                    (model_output * np.array(normalization_params['target_std'])) +
                    np.array(normalization_params['target_mean'])
                )
            elif normalization.lower() == "min_max":
                model_output = (
                    (model_output *
                     (np.array(normalization_params['target_max']) - np.array(normalization_params['target_min']))) +
                    np.array(normalization_params['target_min'])
                )
        

        # Store predictions in the profile
        profile.NN_Prediction = model_output[:data_length]  # Only keep original sequence length
        # Map predictions to feature names
        profile.Predicted_Model = {
            prediction_features_names[j]: profile.NN_Prediction[:, j]
            for j in range(profile.NN_Prediction.shape[1])
        }
        
        if metadata['logarithmic_data'].lower() !="none":
            if metadata['logarithmic_data'].lower() =="natural":
                for key in profile.Predicted_Model:
                    profile.Predicted_Model[key] = np.expm1(profile.Predicted_Model[key])
            
                
            elif metadata['logarithmic_data'].lower() =="base_10":
                for key in profile.Predicted_Model:
                    profile.Predicted_Model[key] = 10**(profile.Predicted_Model[key])
           
        
        # Enforce positive thickness adjustments for the predicted depth
        for j in range(1, len(profile.Predicted_Model['Predicted_Depth'])):
            thickness = profile.Predicted_Model['Predicted_Depth'][j] - profile.Predicted_Model['Predicted_Depth'][j - 1]
            if thickness <= 0:
                profile.Predicted_Model['Predicted_Depth'][j] += abs(thickness) + 1
        
        
        if "Predicted_Resistance" not in metadata['prediction_features']:

            profile.Predicted_Model['Predicted_Resistance'] = profile.Resistance_Data
            
        if "Predicted_Layered_Resistivity" not in metadata['prediction_features']:
            
            profile.Predicted_Model['Predicted_Layered_Resistivity']=em_functions.Layered_Resistivity_from_harmonic_resistance(profile.Predicted_Model['Predicted_Resistance'], profile.Predicted_Model['Predicted_Depth'])



def Rescaling_Prediction_Multiparameter_LSTM_Parametric_Inversion(model, models_test, metadata, device='cpu'):
    """
    Evaluate the LSTM model on unseen data with optional normalization/denormalization.
    
    Args:
        model (nn.Module): The trained LSTM neural network model.
        models_test (list): A list of test profile objects containing input data.
        metadata (dict): Metadata containing feature names, normalization parameters, max sequence length, etc.
        device (str): Device to use for evaluation ('cpu' or 'cuda').
        
    Returns:
        None: Updates each profile in models_test with evaluated results.
    """
    input_feature_names = metadata['training_input_features']
    output_features_names = metadata['training_output_features']
    prediction_features_names = metadata['prediction_features']
    normalization = metadata['normalization']
    normalization_params = metadata['normalization_params']
    max_seq_length = metadata['Max_Sequence_Length']  # Get the max sequence length from training

    # Switch model to evaluation mode
    model.eval()

    for profile in models_test:
        data_length = len(getattr(profile, input_feature_names[0]))

        # Collect and store input data for the profile
        profile_input_data = np.array([
            [getattr(profile, feature)[data_point] for feature in input_feature_names]
            for data_point in range(data_length)
        ])

        # Normalize input data if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                profile_input_data = (
                    (profile_input_data - np.array(normalization_params['input_mean'])) / 
                    np.array(normalization_params['input_std'])
                )
            elif normalization.lower() == "min_max":
                profile_input_data = (
                    (profile_input_data - np.array(normalization_params['input_min'])) / 
                    (np.array(normalization_params['input_max']) - np.array(normalization_params['input_min']))
                )

        # Convert to tensor
        
        profile_input_data = torch.tensor(profile_input_data, dtype=torch.float32)

        # Pad to max sequence length
        padded_input = torch.zeros((max_seq_length, profile_input_data.shape[1]))  # Pre-allocate with padding value (0)
        padded_input[:min(data_length, max_seq_length), :] = profile_input_data[:max_seq_length, :]
        #Move to device (add batch dimension)
        
        padded_input = padded_input.unsqueeze(0).to(device)
        
        if metadata['network_params'].get("type") is not None and metadata['network_params']['type']=="Transformer":
            
            batch_size=1
          
            with torch.no_grad():
                seq_lengths = torch.tensor([min(data_length, max_seq_length)], dtype=torch.int64).to(device)
                key_padding_mask = torch.arange(max_seq_length, device=device).expand(batch_size, max_seq_length) >= seq_lengths.unsqueeze(0)
                
                model_output = model(padded_input, key_padding_mask.to(device))      
            
        else:
        
            # Perform model prediction
            with torch.no_grad():
                seq_lengths = torch.tensor([min(data_length, max_seq_length)], dtype=torch.int64).to(device)
                
                output_seq_lengths = torch.tensor([min(profile.Estimated_Layers, max_seq_length)], dtype=torch.int64).to(device)
                
                model_output = model(padded_input, seq_lengths,output_seq_lengths)

            # Remove batch dimension and convert to numpy
        model_output = model_output.squeeze(0).cpu().numpy()    
        
        if len(model_output.shape) == 1:
            model_output = model_output.reshape(-1, 1)
            
        
        # Denormalize the predictions if required
        if normalization.lower() != "none":
            if normalization.lower() == "z_score":
                model_output = (
                    (model_output * np.array(normalization_params['target_std'])) +
                    np.array(normalization_params['target_mean'])
                )
            elif normalization.lower() == "min_max":
                model_output = (
                    (model_output *
                     (np.array(normalization_params['target_max']) - np.array(normalization_params['target_min']))) +
                    np.array(normalization_params['target_min'])
                )
        

        # Store predictions in the profile
        profile.NN_Prediction = model_output[:output_seq_lengths]  # Only keep original sequence length
        # Map predictions to feature names
        profile.Predicted_Model = {
            prediction_features_names[j]: profile.NN_Prediction[:, j]
            for j in range(profile.NN_Prediction.shape[1])
        }
        
        if metadata['logarithmic_data'].lower() !="none":
            if metadata['logarithmic_data'].lower() =="natural":
                for key in profile.Predicted_Model:
                    profile.Predicted_Model[key] = np.expm1(profile.Predicted_Model[key])
            
                
            elif metadata['logarithmic_data'].lower() =="base_10":
                for key in profile.Predicted_Model:
                    profile.Predicted_Model[key] = 10**(profile.Predicted_Model[key])
                    
        
        profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"] = np.concatenate(([2e14], profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"])) # Insert at the beginning (index 0)
        profile.Predicted_Model["Predicted_Depth_Vector"] = np.concatenate(([0], profile.Predicted_Model["Predicted_Depth_Vector"])) 
        
        profile.Predicted_Model["Predicted_Depth_Vector"]  = profile.Predicted_Model["Predicted_Depth_Vector"].astype(int)
        #profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"]= np.concatenate(profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"], [profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"][-1]])
        
        
        profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"] = np.append( profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"],
                                                                                    profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"][-1])
        
        [profile.Predicted_Model["Predicted_Resistivity_Model"], 
         profile.Predicted_Model["Predicted_Conductivity_Model"], 
         profile.Predicted_Model["Predicted_Depth_Model"]] = em_functions.layered_models_variable_dz(profile.Predicted_Model["Predicted_Depth_Vector"], 
                                                                                        profile.Predicted_Model["Predicted_Layered_Resistivity_Vector"], 
                                                                                        profile.Predicted_Model["Predicted_Depth_Vector"][-1]*3, delta=1)  
        
        
#%%

"""
Neural Network Saving Functions
"""

def save_model_and_metadata(model_path, file_name, 
                            model, training_history,
                            normalization_params,
                            num_models_trained, training_input_features, training_output_features,prediction_features, training_params,
                            logarithmic_data="None", 
                            normalization="None"):
    """
    Save a PyTorch model along with metadata in an efficient and organized manner.

    Args:
        model (torch.nn.Module): The trained model to save.
        model_path (str): Path to save the model.
        file_name (str): Name of the file to save.
        normalization_params (dict): Dictionary containing normalization parameters:
                                     {'input_mean': ..., 'input_std': ..., 'target_mean': ..., 'target_std': ...}.
        num_models_trained (int): Number of models used to train the data.
        input_features (list): List of input feature names.
        output_features (list): List of output feature names.
        prediction_features (list): List of prediction feature names.
        training_params (dict): Dictionary of training parameters such as learning rate, batch size, epochs, optimizer, etc.
        normalized (bool): Indicates if the model was trained with normalized data.

    Returns:
        None
    """
    metadata = {
        'logarithmic_data': logarithmic_data,
        "normalization": normalization,
        'normalization_params': normalization_params,
        'num_models_trained': num_models_trained,
        'training_input_features':  training_input_features,
        'training_output_features': training_output_features,
        'prediction_features': prediction_features,
        'training_params': training_params
    }
    
    # Save the model and metadata together
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'training_history': training_history
    }
    
    torch.save(save_data, model_path + file_name)
    print(f"Model and metadata saved to {file_name}")


def load_model_and_metadata(model_class, model_path, file_name, device='cpu'):
    """
    Load a PyTorch model along with its metadata.

    Args:
        model_class (torch.nn.Module): The class of the model to load.
        model_path (str): Path where the model file is saved.
        file_name (str): Name of the file to load.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The loaded model.
        metadata (dict): The loaded metadata.
    """
    # Load the saved data
    save_data = torch.load(model_path + file_name, map_location=device, weights_only=False)
    
    # Extract the metadata
    metadata = save_data['metadata']

    # Extract the training history
    training_history = save_data['training_history']
    
    # Initialize the model using the extracted parameters
    model = model_class(input_feature_names= metadata['training_input_features'], target_feature_names= metadata['training_output_features'])
    
    # Load model weights
    model.load_state_dict(save_data['model_state_dict'])
    model.to(device)
    
    print(f"Model and metadata loaded from {file_name}")
    
    # Return the model and metadata
    return model, metadata, training_history


def save_model_and_metadata_lstm(model_path, file_name, 
                            model, training_history,
                            normalization_params,
                            num_models_trained, 
                            training_input_features, training_output_features,
                            prediction_features, 
                            max_sequence, network_params, training_params, 
                            logarithmic_data="None",
                            normalization="None"):
    """
    Save a PyTorch model along with metadata in an efficient and organized manner.

    Args:
        model (torch.nn.Module): The trained model to save.
        model_path (str): Path to save the model.
        file_name (str): Name of the file to save.
        normalization_params (dict): Dictionary containing normalization parameters:
                                     {'input_mean': ..., 'input_std': ..., 'target_mean': ..., 'target_std': ...}.
        num_models_trained (int): Number of models used to train the data.
        input_features (list): List of input feature names.
        output_features (list): List of output feature names.
        prediction_features (list): List of prediction feature names.
        training_params (dict): Dictionary of training parameters such as learning rate, batch size, epochs, optimizer, etc.
        normalized (bool): Indicates if the model was trained with normalized data.

    Returns:
        None
    """
    metadata = {
        'logarithmic_data': logarithmic_data,
        "normalization": normalization,
        'normalization_params': normalization_params,
        'num_models_trained': num_models_trained,
        'training_input_features': training_input_features,
        'training_output_features': training_output_features,
        'prediction_features': prediction_features,
        'Max_Sequence_Length': max_sequence,
        'network_params':network_params,
        'training_params': training_params
        
    }
    
    # Save the model and metadata together
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'training_history': training_history
    }
    
    torch.save(save_data, model_path + file_name)
    print(f"Model and metadata saved to {file_name}")


def save_model_and_metadata_lstm_inversion(model_path, file_name, 
                            model, training_history,
                            normalization_params,
                            num_models_trained, 
                            training_input_features, training_output_features,
                            prediction_features, 
                            max_sequence, max_output_sequence, network_params, training_params, 
                            logarithmic_data="None",
                            normalization="None"):
    """
    Save a PyTorch model along with metadata in an efficient and organized manner.

    Args:
        model (torch.nn.Module): The trained model to save.
        model_path (str): Path to save the model.
        file_name (str): Name of the file to save.
        normalization_params (dict): Dictionary containing normalization parameters:
                                     {'input_mean': ..., 'input_std': ..., 'target_mean': ..., 'target_std': ...}.
        num_models_trained (int): Number of models used to train the data.
        input_features (list): List of input feature names.
        output_features (list): List of output feature names.
        prediction_features (list): List of prediction feature names.
        training_params (dict): Dictionary of training parameters such as learning rate, batch size, epochs, optimizer, etc.
        normalized (bool): Indicates if the model was trained with normalized data.

    Returns:
        None
    """
    metadata = {
        'logarithmic_data': logarithmic_data,
        "normalization": normalization,
        'normalization_params': normalization_params,
        'num_models_trained': num_models_trained,
        'training_input_features': training_input_features,
        'training_output_features': training_output_features,
        'prediction_features': prediction_features,
        'Max_Sequence_Length': max_sequence,
        'Max_Output_Sequence_Length': max_output_sequence,
        'network_params':network_params,
        'training_params': training_params
        
    }
    
    # Save the model and metadata together
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'training_history': training_history
    }
    
    torch.save(save_data, model_path + file_name)
    print(f"Model and metadata saved to {file_name}")


def load_model_and_metadata_lstm(model_path, file_name, device='cpu'):
    """
    Load a PyTorch model along with its metadata.

    Args:
        model_path (str): Path where the model file is saved.
        file_name (str): Name of the file to load.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The loaded model.
        metadata (dict): The loaded metadata.
    """
    # Load the saved data
    save_data = torch.load(model_path + file_name, map_location=device, weights_only=False)
    
    # Extract the metadata
    metadata = save_data['metadata']
    # Extract the training history
    training_history = save_data['training_history']
    
    known_model_classes = [
    "Hybrid_CNN_LSTM_Residual",
    "Hybrid_CNN_LSTM_Residual_Inversion",]
    # Add more as needed]

    model_name = next((name for name in known_model_classes if file_name.startswith(name)), None)


    model_class = globals().get(model_name) 
    
    # Initialize the model using the extracted parameters
    
    
    if metadata['network_params'].get("type") is not None and metadata['network_params']['type']=="Transformer":
        
        model = model_class(input_feature_names= metadata['training_input_features'], 
                            target_feature_names= metadata['training_output_features'],
                            cnn_channels=metadata['network_params']['cnn_channels'], 
                            num_layers=metadata['network_params']['num_layers'],
                            hidden_dim=metadata['network_params']['hidden_dim'], 
                            nhead = metadata['network_params']['nhead'],
                            dropout = metadata['network_params']['dropout'],
                            )
        
    else:
        model = model_class(input_feature_names= metadata['training_input_features'], 
                            target_feature_names= metadata['training_output_features'],
                            cnn_channels=metadata['network_params']['cnn_channels'], 
                            hidden_size=metadata['network_params']['hidden_size'], 
                            num_layers=metadata['network_params']['num_layers'],
                            dropout = metadata['network_params']['dropout'],
                            bidirectional = metadata['network_params']['bidirectional'])
    
    # Load model weights
    model.load_state_dict(save_data['model_state_dict'])
    model.to(device)
    
    print(f"Model and metadata loaded from {file_name}")
    
    # Return the model and metadata
    return model, metadata, training_history

#%%



