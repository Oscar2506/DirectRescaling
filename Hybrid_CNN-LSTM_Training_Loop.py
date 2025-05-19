 # -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:52:37 2024

@author: Oscar Ivan Calderon Hernandez Ph.D Student Politecnico di Torino
"""

"""
Basic Libraries
"""
import math
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import shap

import re

"""
Sklearn libraries 
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

"""
Torch Libraries 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
import copy

"""
Personal Libraries
"""
import em_functions
import nn_functions

"""
Physical Constants
"""

pi=math.pi
exp=math.exp(1)
i_num=1j
mu0=mu_0=4*pi*(10**(-7))    #Magnetic Permeability of the Earth

#%%
"""
Initial Parameters for the Training 
"""
Random_Models=True

Train_path = "/samoa/data/ocalderonherna/MT_Data/Neural_Network_Data/Training_Models/Random_Models/"

Logarithmic_Data="Base_10"

inversion=False

noise=True

Normalization="z_Score"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if inversion:
    Input_Feature_Names = ['Resistance_Data', 'Resistance_Gradient_Data', 'Phase_Data', 'Frequencies_Data'] 
    Output_Feature_Names = ['Harmonic_Resistance_Model_Data','Depth_Model_Data'] 

    Prediction_Feature_Names = ['Predicted_Resistance','Predicted_Depth'] 

else:

    Input_Feature_Names = ['Resistance_Data', 'Phase_Data', 'Frequencies_Data']        
    Output_Feature_Names = ['Depth_Frequency_Points'] 

    Prediction_Feature_Names = ['Predicted_Depth']


if Logarithmic_Data.lower() !="none":

    Input_Feature_Names = [name + "_log" for name in Input_Feature_Names]  

    Output_Feature_Names = [name + "_log" for name in Output_Feature_Names] 


Training_Parameters = {
    'train_size': 0.8,
    'learning_rate':0.001,
    'batch_size': 32,
    'epochs': 1000,
    'early_stop_limit': 200,
    'loss': "Interval Weighted",  #Loss Functions [MAE, MSE, Huber, Interval Weighted, Decade_Weigthed, Hybrid_Gradient, Normalized]
    'optimizer': 'Adam',
    'scheduler': 'Cosinewr',
    'seed' : 42,
    'device' : device}
    


#%%

"""
Reading of the Information of the Models to Train
"""
start_time = time.time()

Train_paths=[]

for model_path in os.listdir(Train_path):
    # check if current path is a file
    Train_paths.append(model_path)
    
Train_paths=sorted(Train_paths, key=len)        

Models=[]

models_to_train=len(Train_paths)

print("----- Reading Training Models -----")

for i in Train_paths[0:models_to_train]:
    
    #print("----- Reading Training: "+str(i)+" -----")
    
    dir_path=Train_path+str(i)
    
    match = re.search(r'_\d+',dir_path)
    
    label="Train_Model"+ match.group()
    
    globals()[label]=em_functions.Load_npz_Model(dir_path, training=True)
    
    Models.append(globals()[label])
    


for profile in Models:    
    
    print("----- Rescaling for: "+str(profile.model)+" -----")

    if inversion:
        profile.Depth_Model_Data, profile.Harmonic_Resistance_Model_Data = em_functions.model_subsampling(profile.Depth_Model, profile.Harmonic_Resistance_Model, len(profile.Resistance_Data))


    profile.Logarithmic_Data(Logarithmic_Data, training=True, inversion=inversion)

    em_functions.Transverse_Resistance_Points(profile)

    if noise:
        noise_level = 0.05
        noise_vector=(1 + np.random.normal(0, noise_level, size=profile.Resistance_Data.shape))
        profile.Resistance_Data = profile.Resistance_Data * noise_vector
        profile.Impedance_Data_Imag=abs(profile.Impedance_Data.imag) * noise_vector
        profile.Phase_Data = profile.Phase_Data * noise_vector
    
    if Logarithmic_Data.lower() != "none":

        if Logarithmic_Data=="Base_10":
            profile.Discretized_Layered_Resistivity_log=np.log10(profile.Discretized_Layered_Resistivity)
        
        elif Logarithmic_Data=="Natural":

            profile.Discretized_Layered_Resistivity_log=np.log1p(profile.Discretized_Layered_Resistivity)


        profile.Resistance_Gradient_Data_log=np.gradient(profile.Resistance_Data_log, profile.Frequencies_Data_log)
        
    else:
        
        profile.Resistance_Gradient_Data=np.gradient(profile.Resistance_Data, profile.Frequencies_Data)

#%%
"""
Shuffling of the data
"""    
random.seed(42)  
list_of_models=list(range(1,models_to_train+1))
combined = list(zip(Models, list_of_models))

random.shuffle(combined)

# Unzip back into separate lists
Models, Models_List = zip(*combined)
#%%

 
    
def train_model_Hybrid_CNN_LSTM(model, X, y, train_size=0.8, n_epochs=1000, batch_size=256, lr=0.0001, early_stop_limit=50, seed=42, loss_mode='MSE', optimizer_mode='Adam', lr_mode='None', device='cpu'):
    """
    Train a neural network model with the option of early stopping and learning rate scheduling.
    
    Args:
        model (nn.Module): The model to be trained.
        dataset (torch.utils.data.Dataset): Dataset containing input and target sequences.
        train_size (float): Proportion of data to use for training.
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
        history (dict): A dictionary containing training and validation metrics.
    """

    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    dataset=nn_functions.Models_Dataset(X,y)

    # Split the dataset into training and testing sets
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=nn_functions.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=nn_functions.collate_fn)

    # Model, loss function, optimizer, and scheduler
    model = model.to(device)

    # Select Loss Function
    loss_functions = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss(),
        'normalized': nn_functions.Normalized_Losses_Optimized(Loss=loss_mode),  # Callable function
        'hybrid_gradient': nn_functions.Hybrid_Gradient_Loss(alpha=0.5, beta=0.5), 
        "decade_weighted": nn_functions.Decade_Weighted_Loss(),
        "interval_weighted": nn_functions.Interval_Weighted_Loss(),
    }
    
    loss_mode_lower=loss_mode.lower()

    Normalized_Losses_Names = ["local_l1", "local_l2", "global_l1", "global_l2"]

    if loss_mode_lower in Normalized_Losses_Names:
        loss_mode_lower="normalized"

    loss_fn = loss_functions.get(loss_mode_lower, torch.nn.MSELoss())  # Default to MSELoss

    # Select Optimizer
    optimizer_options = {
        'adam': optim.Adam(model.parameters(), lr=lr),#, betas=(0.9, 0.95), weight_decay=1e-4),
        'sgd': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'rmsprop': optim.RMSprop(model.parameters(), lr=lr)
    }
    optimizer = optimizer_options.get(optimizer_mode.lower(), optim.Adam(model.parameters(), lr=lr))  # Default to Adam

    # Select Learning Rate Scheduler
    scheduler_options = {
        'onecycle': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10, total_steps=n_epochs * len(train_loader),  # Total training steps
                    pct_start=0.3,            # 30% of training time spent on increasing LR
                    anneal_strategy='cos',    # Smooth cosine decay
                    div_factor=25,            # Initial LR = max_lr / div_factor
                    final_div_factor=1e4      # Final LR = max_lr / final_div_factor
                    ),
        'cosinewr': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6),
        'cosinelr': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6),
        'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5),
        'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20),
        'none': None  # No scheduler
    }
    scheduler = scheduler_options.get(lr_mode.lower(), None)  # Default to no scheduler

    # Training parameters
    best_eval_loss = float('inf')
    best_weights = None
    history = {
        'train_loss': [], 'val_loss': [], 'lr_history': [],
        'train_R2_profile': [], 'train_R2_global': [],
        'val_R2_profile': [], 'val_R2_global': [],
        'train_percentage_error': [], 'val_percentage_error': [],
        'train_NRMSE_profile': [], 'val_NRMSE_profile': [],
        'train_NRMSE_global': [], 'val_NRMSE_global': []
        }
    
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_r2 = 0
        epoch_nrmse = 0
        epoch_percentage_error = 0
        profile_nrmse_list = []
        profile_percentage_error_list = []
        profile_r2_list = []
        
        for X_batch, y_batch, seq_len in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch"):
            X_batch, y_batch, seq_len = X_batch.to(device), y_batch.to(device), seq_len.to(device)
    
            # Forward pass
            y_pred = model(X_batch, seq_len)
            
            # Compute loss
            if loss_mode_lower == "hybrid_gradient":
                loss = loss_fn(y_pred, y_batch, X_batch)  # Pass X_batch for resistance-based gradient
            else:
                loss = loss_fn(y_pred, y_batch)  # Default loss
                
            epoch_losses.append(loss.item())
            
            # Compute R² per profile
            y_batch_flat = y_batch.view(y_batch.shape[0], -1).cpu().detach().numpy()
            y_pred_flat = y_pred.view(y_pred.shape[0], -1).cpu().detach().numpy()
            
            r2_profiles = [r2_score(y_true, y_pred) for y_true, y_pred in zip(y_batch_flat, y_pred_flat)]
            profile_r2_list.extend(r2_profiles)
            epoch_r2 += np.mean(r2_profiles)
            
            # Dynamically get sequence length & output dimension
            seq_len = y_batch.shape[1]  # Number of time steps
            output_dim = y_batch.shape[-1]  # Number of output features
            
            # Compute MSE per profile (reduce over seq_len)
            mse_profile = torch.mean((y_batch - y_pred) ** 2, dim=1)  # Shape: (batch_size, output_dim)
            rmse_profile = torch.sqrt(mse_profile)  # Shape: (batch_size, output_dim)
            
            # Compute NRMSE per profile
            min_vals, _ = torch.min(y_batch, dim=1, keepdim=True)  # Keep seq_len dim
            max_vals, _ = torch.max(y_batch, dim=1, keepdim=True)  # Keep seq_len dim
            range_vals = max_vals - min_vals + 1e-8  # Prevent division by zero
            range_vals = range_vals.squeeze(1)  # Shape: (batch_size, output_dim), matches rmse_profile
            
            nrmse_profile = (rmse_profile / range_vals).detach().cpu().numpy()
            profile_nrmse_list.extend(nrmse_profile.tolist())
            epoch_nrmse += np.mean(nrmse_profile)
            
            # Compute Percentage Error per profile
            percentage_error = (torch.abs(y_batch - y_pred) / (torch.abs(y_batch) + 1e-8)) * 100
            percentage_error_profile = torch.mean(percentage_error, dim=1).detach().cpu().numpy()  # Shape: (batch_size, output_dim)
            
            profile_percentage_error_list.extend(percentage_error_profile.tolist())
            epoch_percentage_error += np.mean(percentage_error_profile)

        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
    
        # Compute epoch-wise averages
        train_loss = np.mean(epoch_losses)
        train_r2_global = epoch_r2 / len(train_loader)
        train_nrmse_global = epoch_nrmse / len(train_loader)
        train_percentage_global = epoch_percentage_error / len(train_loader)
        
        # Store in history
        history['train_loss'].append(train_loss)
        history['train_R2_profile'].append(profile_r2_list)
        history['train_R2_global'].append(train_r2_global)
        history['train_NRMSE_global'].append(train_nrmse_global)
        history['train_NRMSE_profile'].append(profile_nrmse_list)
        history['train_percentage_error'].append(train_percentage_global)



        # Evaluation phase
        model.eval()
        eval_losses = []
        epoch_r2 = 0
        epoch_nrmse = 0
        epoch_percentage_error = 0
        profile_nrmse_list = []
        profile_percentage_error_list = []
        profile_r2_list = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len in test_loader:
                X_batch, y_batch, seq_len = X_batch.to(device), y_batch.to(device), seq_len.to(device)
                
                
                y_pred = model(X_batch, seq_len)
                
                #loss = loss_fn(y_pred, y_batch) if loss_mode_lower != "weigthed" else loss_fn(y_pred, y_batch, seq_len)
                if loss_mode_lower == "hybrid_gradient":
                    loss = loss_fn(y_pred, y_batch, X_batch)  # Pass X_batch for resistance-based gradient
                else:
                    loss = loss_fn(y_pred, y_batch)  # Default loss
                eval_losses.append(loss.item())

                # Compute R² per profile
                y_batch_flat = y_batch.view(y_batch.shape[0], -1).cpu().detach().numpy()
                y_pred_flat = y_pred.view(y_pred.shape[0], -1).cpu().detach().numpy()
                
                r2_profiles = [r2_score(y_true, y_pred) for y_true, y_pred in zip(y_batch_flat, y_pred_flat)]
                profile_r2_list.extend(r2_profiles)
                epoch_r2 += np.mean(r2_profiles)
                
                # Dynamically get sequence length & output dimension
                seq_len = y_batch.shape[1]  # Number of time steps
                output_dim = y_batch.shape[-1]  # Number of output features
                
                # Compute MSE per profile (reduce over seq_len)
                mse_profile = torch.mean((y_batch - y_pred) ** 2, dim=1)  # Shape: (batch_size, output_dim)
                rmse_profile = torch.sqrt(mse_profile)  # Shape: (batch_size, output_dim)
                
                # Compute NRMSE per profile
                min_vals, _ = torch.min(y_batch, dim=1, keepdim=True)  # Keep seq_len dim
                max_vals, _ = torch.max(y_batch, dim=1, keepdim=True)  # Keep seq_len dim
                range_vals = max_vals - min_vals + 1e-8  # Prevent division by zero
                range_vals = range_vals.squeeze(1)  # Shape: (batch_size, output_dim), matches rmse_profile
                
                nrmse_profile = (rmse_profile / range_vals).detach().cpu().numpy()
                profile_nrmse_list.extend(nrmse_profile.tolist())
                epoch_nrmse += np.mean(nrmse_profile)
                
                # Compute Percentage Error per profile
                percentage_error = (torch.abs(y_batch - y_pred) / (torch.abs(y_batch) + 1e-8)) * 100
                percentage_error_profile = torch.mean(percentage_error, dim=1).detach().cpu().numpy()  # Shape: (batch_size, output_dim)
                
                profile_percentage_error_list.extend(percentage_error_profile.tolist())
                epoch_percentage_error += np.mean(percentage_error_profile)

        # Compute evaluation metrics
        val_loss = np.mean(eval_losses)
        val_r2_global = epoch_r2 / len(test_loader)
        val_nrmse_global = epoch_nrmse / len(test_loader)
        val_percentage_global = epoch_percentage_error / len(test_loader)
        
        # Store in history
        history['val_loss'].append(val_loss)
        history['val_R2_profile'].append(profile_r2_list)
        history['val_R2_global'].append(val_r2_global)
        history['val_NRMSE_global'].append(val_nrmse_global)
        history['val_NRMSE_profile'].append(profile_nrmse_list)
        history['val_percentage_error'].append(val_percentage_global)
        
        # Learning Rate Scheduler Step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # Adjust LR based on validation loss
                print("---------------------- Learning Rate: "+ str(scheduler.get_last_lr()) +"--------------------")
            else:
                scheduler.step()  # Standard schedulers
                print("---------------------- Learning Rate: "+ str(scheduler.get_last_lr()) +"--------------------")
        else:
            
            print(f"---------------------- Learning Rate: {optimizer.param_groups[0]['lr']} --------------------")
        # Print epoch results
        print(f"Epoch {epoch + 1}: Training {loss_mode} Loss= {train_loss:.6f}, Validation {loss_mode} Loss= {val_loss:.6f}, Best {loss_mode} Loss= {best_eval_loss:.6f}")

        # Save best model weights
        if val_loss < best_eval_loss:
            best_eval_loss = val_loss
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

    return model, history


#%%
"""
Data preparation for the network
"""

Input_Features=len(Input_Feature_Names)

Ouput_Features=len(Output_Feature_Names)

X, y, Normalization_Parameters, max_sequence = nn_functions.Prepare_NN_Dataset_by_Profile(Models, Input_Feature_Names, Output_Feature_Names, normalization=Normalization) 

#%%

"""
Training of the network
"""

LSTM=True
if LSTM:

    Network_Parameters = {
        'type': "LSTM",
        'cnn_channels': [32,64],
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.1,
        'bidirectional': True
    }
    
    model = nn_functions.Hybrid_CNN_LSTM_Residual(Input_Feature_Names,Output_Feature_Names,
                            cnn_channels=Network_Parameters['cnn_channels'], 
                            hidden_size=Network_Parameters['hidden_size'], 
                            num_layers=Network_Parameters['num_layers'],
                            dropout = Network_Parameters['dropout'],
                            bidirectional=Network_Parameters['bidirectional'])
    
else:

    Network_Parameters = {
        'type': "Transformer",
        'cnn_channels': 32,
        'num_layers': 4,
        'hidden_dim':128,
        'nhead':4,
        'dropout': 0.2,
    }
    
    model = Transformer_Model(Input_Feature_Names,Output_Feature_Names,
                            cnn_channels=Network_Parameters['cnn_channels'], 
                            num_layers=Network_Parameters['num_layers'],
                            hidden_dim=Network_Parameters['hidden_dim'], 
                            nhead = Network_Parameters['nhead'],
                            dropout=Network_Parameters['dropout'])


Network_Model = str(model.__class__.__name__)

start_time = time.time()

if LSTM:
    
    trained_model, training_history =  train_model_Hybrid_CNN_LSTM(model, X, y,
                                                               train_size=Training_Parameters['train_size'], 
                                                               n_epochs=Training_Parameters['epochs'], 
                                                               batch_size=Training_Parameters['batch_size'], 
                                                               lr=Training_Parameters['learning_rate'], 
                                                               early_stop_limit=Training_Parameters['early_stop_limit'], 
                                                               seed=Training_Parameters['seed'],                                                          
                                                               loss_mode=Training_Parameters['loss'], 
                                                               optimizer_mode=Training_Parameters['optimizer'],
                                                               lr_mode=Training_Parameters['scheduler'],
                                                               device=Training_Parameters['device'])
       
else:

    trained_model, training_history =  train_model_Transformer(model, X, y,
                                                               train_size=Training_Parameters['train_size'], 
                                                               n_epochs=Training_Parameters['epochs'], 
                                                               batch_size=Training_Parameters['batch_size'], 
                                                               lr=Training_Parameters['learning_rate'], 
                                                               early_stop_limit=Training_Parameters['early_stop_limit'], 
                                                               seed=Training_Parameters['seed'],                                                          
                                                               loss_mode=Training_Parameters['loss'], 
                                                               optimizer_mode=Training_Parameters['optimizer'],
                                                               lr_mode=Training_Parameters['scheduler'],
                                                               device=Training_Parameters['device'])

end_time = time.time()
total_runtime = (end_time - start_time)/60
print(f"Total runtime for the Rescaling of the Models:  {total_runtime:.2f} minutes")

train_size = int(0.8 * len(Models_List))
Training_List = Models_List[:train_size]
Validation_List = Models_List[train_size:]

#print(f"List of First 15 Training Models:  {Training_List[0:15]}")

#print(f"List of first 15 Validation Models:  {Validation_List[0:15]}")

#%%
Model_path = "/samoa/data/ocalderonherna/MT_Data/Neural_Network_Models/CNN_LSTM_Trained_Models/Final_Models/"

Random_Models=True

# Define file name based on normalization and feature count

Rand = "Random" if Random_Models else "Not_Rand"

Run=1

if inversion:
    
    file_name = f"{Network_Model}_Run_{Run}_Inversion.pth"

else:
    
    file_name = f"{Network_Model}_Run_{Run}_Rescaling_Resistance.pth"


#%%

# Save model and metadata
nn_functions.save_model_and_metadata_lstm(
    model_path=Model_path,
    file_name=file_name,
    model=trained_model,
    training_history=training_history,
    normalization_params=Normalization_Parameters,
    num_models_trained=models_to_train,
    training_input_features=Input_Feature_Names,
    training_output_features=Output_Feature_Names,
    prediction_features=Prediction_Feature_Names,
    max_sequence=max_sequence,
    training_params=Training_Parameters,
    network_params=Network_Parameters,
    logarithmic_data=Logarithmic_Data,
    normalization=Normalization
)


