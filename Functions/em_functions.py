# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:40:59 2023

@author: Oscar Ivan Calderon Hernandez Ph.D Student Politecnico di Torino
"""
"""------------------------------ Function and Class Main File for the Simulation of the Impedance / Resistance Data and Models ------------------------------"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sklearn

import math
import cmath
import random
import h5py
import empymod
import time


from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from scipy import interpolate



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


#Functions to Simulate MT, FDEM ot TDEM Data 


"""
MT 2D EM fields simulation
This Simulation Considers Offset 0 and a far a way source to compute the MT response
"""

def mt_simulation_fields(res,depth,frequencies, offset=0, receiver=0.1, Case_2D=False):
        model = {
            'src': [-100_000_000, -100_000_000, -100_000_000],     # Source - x, y,z
            'rec': [offset, offset, receiver],        # Receiver Coordinates - x, y, z, 
            'depth': depth,  # Layer boundaries
            'res':   res,    # Air, subsurface resistivities
            'freqtime': frequencies,                   # Frequencies
            'verb': 0,                                 # Amount of Information Displayed in the Simulation
            'htarg': {'pts_per_dec': -1},              # (For faster computation)
        }

        ex_field = empymod.dipole(ab=11, **model)  # Increase the verbosity to see some more output
        hy_field = empymod.dipole(ab=51, **model)

        ey_field=None
        hx_field=None

        if Case_2D:
            ey_field= empymod.dipole(ab=21, **model) 
            hx_field = empymod.dipole(ab=41, **model)
        
        return ex_field, hy_field, ey_field, hx_field 



"""
This Simulation computes the electric and magnetic field lowering the source up to a given point in depth for a given spacing (delta).
The 2D Case is pending
"""

def EM_Field_propagation(model, res, depth, frequencies, write_output, file_path, delta=10, depth_cutoff=1.5):

    print(f"----- Simulation of EM Fields for {model} Starting -----")
    
    limit = depth[-1] * depth_cutoff
    field_depth = np.arange(0, limit, delta)
    
    electric_field = []
    magnetic_field = []
    
    for j in field_depth:
        if np.isclose(j % 100, 0, atol=delta / 2):  
            print(f"Current Depth of Source = {j:.2f} [m] for {model}")
        
        ex_field, hy_field, ey_field, hx_field= mt_simulation_fields(res, depth, frequencies, receiver=j)
        electric_field.append(ex_field)
        magnetic_field.append(hy_field)
    
    print(f"----- Simulation of EM Fields for {model} Finished -----")
    
    # Convert to NumPy arrays for efficient computation
    electric_field = np.abs(np.array(electric_field))
    magnetic_field = np.abs(np.array(magnetic_field))
    
    # Compute impedance
    impedance = (electric_field / magnetic_field) ** 2
    
    # Normalize fields and impedance
    per_elec_field = electric_field / electric_field[0, :]  # Broadcasting for normalization
    per_mag_field = magnetic_field / magnetic_field[0, :]
    per_impedance = impedance / impedance[0, :]
    
    # Save output if requested
    if write_output:
        output_dir = file_path+ str(model)+"\\EM_Field_Simulation\\"
        os.makedirs(output_dir, exist_ok=True)
        
        EM_Fields_save_matrix(output_dir, "Electric_Field_Matrix.txt", electric_field)
        EM_Fields_save_matrix(output_dir, "Magnetic_Field_Matrix.txt", magnetic_field)
        EM_Fields_save_matrix(output_dir, "Electric_Impedance_Matrix.txt", impedance)
        
        EM_Fields_save_matrix(output_dir, "Electric_Field_Amplitude_Matrix.txt", per_elec_field)
        EM_Fields_save_matrix(output_dir, "Magnetic_Field_Amplitude_Matrix.txt", per_mag_field)
        EM_Fields_save_matrix(output_dir, "Electric_Impedance_Amplitude_Matrix.txt", per_impedance)
    
    return electric_field, magnetic_field, impedance, per_elec_field, per_mag_field, per_impedance, field_depth



"""
Impedance and Phase Computation for a 1D Model
"""

def impedance_1D(ex_field,hy_field):

  Z = ex_field/hy_field
  
  phase_deg = (180/pi)*np.arctan2(Z.imag, Z.real)

  return Z, phase_deg


"""
------------------------------- Airborne FDEM Data Simulation ------------------------------
In this version the parameters of the source and receiver are fixed
"""


def FDEM_simulation(con, thicknesses, frequencies, layers):

    from SimPEG import maps
    from SimPEG.electromagnetics import frequency_domain as fdem

    receiver_location = np.array([10.0, 0.0, 30.0])
    receiver_orientation = "z"  # "x", "y" or "z"
    data_type = "ppm"  # "secondary", "total" or "ppm"

    receiver_list = []
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location,
            orientation=receiver_orientation,
            data_type=data_type,
            component="real",
        )
    )
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location,
            orientation=receiver_orientation,
            data_type=data_type,
            component="imag",
        )
    )

    # Define the source list. A source must be defined for each frequency.
    source_location = np.array([0.0, 0.0, 30.0])
    source_orientation = "z"  # "x", "y" or "z"
    moment = 1.0  # dipole moment

    source_list = []
    for freq in frequencies:
        source_list.append(
            fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=freq,
                location=source_location,
                orientation=source_orientation,
                moment=moment,
            )
        )

    # Define a 1D FDEM survey
    survey = fdem.survey.Survey(source_list)

    model_mapping = maps.IdentityMap(nP=layers+1)
            
    simulation = fdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=model_mapping,
    )
    con= con[1::]
    Magnetic_Field = simulation.dpred(con)/10000

    return Magnetic_Field


"""
------------------------------- Airborne TDEM Data Simulation ------------------------------
In this version the parameters of the source and receiver are fixed
"""


def TDEM_simulation(con, thicknesses, periods, layers):

    from SimPEG import maps
    import SimPEG.electromagnetics.time_domain as tdem

    # Source properties
    source_location = np.array([0.0, 0.0, 20.0])
    source_orientation = "z"  # "x", "y" or "z"
    source_current = 1.0  # maximum on-time current
    source_radius = 6.0  # source loop radius

    # Receiver properties
    receiver_location = np.array([0.0, 0.0, 20.0])
    receiver_orientation = "z"  # "x", "y" or "z"

    # Define receiver list. In our case, we have only a single receiver for each source.
    # When simulating the response for multiple component and/or field orientations,
    # the list consists of multiple receiver objects.
    receiver_list = []
    receiver_list.append(
        tdem.receivers.PointMagneticFluxDensity(
        receiver_location, periods, orientation=receiver_orientation
        )
    )

    # Define the source waveform. Here we define a unit step-off. The definition of
    # other waveform types is covered in a separate tutorial.
    waveform = tdem.sources.StepOffWaveform()

    # Define source list. In our case, we have only a single source.
    source_list = [
        tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=waveform,
        current=source_current,
        radius=source_radius,
        )
    ]
    # Define a 1D TDEM survey
    survey = tdem.Survey(source_list)


    model_mapping = maps.IdentityMap(nP=layers+1)

    # Define the simulation
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=model_mapping,
    )

    # Predict data for a given model
    con= con[1::]

    Magnetic_Flux = simulation.dpred(con)

    return Magnetic_Flux


#Functions to Compute Resistivity, Equivalent Resistivity and Different Cumulative Models eg. Transverse Resistance, Longitudinal Conductance, etc.


"""
Resistivity and Conductivity Models with a variable dz interval
"""

def layered_models_variable_dz(depth, res, max_depth, delta=1):
    
    depth_model = np.arange(0, max_depth + delta, delta)
    res_model_segments = []  

    # Calculate index positions in depth_model corresponding to depth entries
    depth_indices = [int(d // delta) for d in depth]

    # Loop through the depth intervals and populate resistivity model
    for i in range(len(depth) - 1):
        start_idx = depth_indices[i]
        end_idx = depth_indices[i + 1]

        # Fill resistivity values for each segment
        res_segment = [res[i + 1]] * (end_idx - start_idx)
        res_model_segments.extend(res_segment)
    
    # Add the final layer resistivity beyond the last depth point
    res_model_segments.extend([res[-1]] * (len(depth_model) - len(res_model_segments)))

    # Convert accumulated segments to numpy array
    res_model = np.array(res_model_segments)

    con_model = 1 / res_model

    # Return depth_model, res_model, and con_model with first elements removed
    return res_model[1:], con_model[1:], depth_model[1:]


"""
 Function to create subsampled models while maintaining equal samples per decade (when possible).
"""

def model_subsampling(depth, resistance, num_samples):
    """

    Parameters:
    - depth (np.array): Array of depth values.
    - parameter (np.array): Corresponding parameter values.
    - num_samples (int): Total number of samples to return.

    Returns:
    - sub_depth (np.array): Subsampled depth values.
    - sub_parameter (np.array): Subsampled parameter values.
    """

    # Convert depth to log10 space
    log_depth = np.log10(depth)

    # Identify unique decades
    min_decade, max_decade = int(np.floor(log_depth.min())), int(np.ceil(log_depth.max()))
    decades = np.arange(min_decade, max_decade + 1)

    # Group indices by decade
    decade_indices = {dec: [] for dec in decades}
    for i, log_d in enumerate(log_depth):
        decade = int(np.floor(log_d))
        decade_indices[decade].append(i)

    # Compute initial allocation (equal division)
    samples_per_decade = {dec: max(1, num_samples // len(decades)) for dec in decades}

    # Adjust allocation to ensure no decade is allocated more samples than it has
    for dec in decades:
        samples_per_decade[dec] = min(samples_per_decade[dec], len(decade_indices[dec]))

    # Calculate total allocated samples
    total_allocated = sum(samples_per_decade.values())

    # Distribute remaining samples
    remaining_samples = num_samples - total_allocated
    if remaining_samples > 0:
        # Sort decades by the number of available samples (descending)
        sorted_decades = sorted(decades, key=lambda d: len(decade_indices[d]), reverse=True)
        for dec in sorted_decades:
            if remaining_samples <= 0:
                break
            # Calculate how many additional samples can be added to this decade
            available_samples = len(decade_indices[dec]) - samples_per_decade[dec]
            if available_samples > 0:
                add_samples = min(available_samples, remaining_samples)
                samples_per_decade[dec] += add_samples
                remaining_samples -= add_samples

    # Remove excess samples if any
    while total_allocated > num_samples:
        # Sort decades by the number of allocated samples (ascending)
        sorted_decades = sorted(decades, key=lambda d: samples_per_decade[d])
        for dec in sorted_decades:
            if total_allocated <= num_samples:
                break
            if samples_per_decade[dec] > 1:  # Ensure at least one sample per decade
                samples_per_decade[dec] -= 1
                total_allocated -= 1
                
    # Print the number of samples allocated per decade
    #print("Samples per decade:")
    #for dec in sorted(decades):
     #   print(f"Decade {dec}: {samples_per_decade[dec]} samples")

    # Select samples
    sub_indices = []
    for dec in decades:
        indices = decade_indices[dec]
        num_to_sample = samples_per_decade[dec]
        if len(indices) <= num_to_sample:
            sub_indices.extend(indices)  # Take all if there are few
        else:
            sub_indices.extend(np.random.choice(indices, num_to_sample, replace=False))  # Random selection

    # Ensure sorted order
    sub_indices = sorted(sub_indices)

    return depth[sub_indices], resistance[sub_indices]


"""
Function to create a smoother version of a given resistivity and depth vector used to simulate MT data
"""

def Model_Vector_Smoother(res, depth, sub_layers=10, intervals=5):
    smooth_res = res[:2]  # Keep first two values
    smooth_depth = depth[:2]  # Keep first depth point
    
    for i in range(2, len(depth)):
        
        min_thickness = (depth[i] - depth[i - 1]) // intervals
        thickness = min_thickness * (intervals // 2)  # Ensure correct truncation
        
        first_point=depth[i-1]+thickness       
        second_point=depth[i]-thickness
        
        # Define resistivity transition (smooth gradient from res[i] to res[i+1])
        
        resistivity_transition = np.round(np.linspace(res[i-1], res[i], sub_layers+3 ),1)
        
        if i<len(depth)-1:
            
            sub_depths =np.sort(np.linspace(smooth_depth[-1], first_point, sub_layers)).astype(int)
            smooth_depth.extend(sub_depths)
            smooth_res.extend(resistivity_transition[1:-1])
                        
            smooth_depth.extend([first_point, second_point])
            smooth_res.append(res[i])
                   
        else:
            
            sub_depths =np.sort(np.linspace(smooth_depth[-1], depth[i-1],  sub_layers)).astype(int)
            smooth_depth.extend(sub_depths)
            smooth_res.extend(resistivity_transition[1:-1])
            smooth_depth.append(depth[-2])
            smooth_res.append(res[-1])
        
    smooth_res.append(res[-1])
    smooth_depth.append(depth[-1])

    return smooth_res, smooth_depth



"""
Transverse Resistance and Longitudinal Conductance computation
"""

def transverse_resistance_longitudinal_conductance(Resistivity_Model, Depth_Model):

    # Initialize first values outside loop for clarity
    resistance = [Resistivity_Model[0] * Depth_Model[0]]
    conductance = [Depth_Model[0] / Resistivity_Model[0]]
    
    # Calculate transverse resistance and longitudinal conductance for each layer
    for i in range(1, len(Resistivity_Model)):
        thickness = Depth_Model[i] - Depth_Model[i - 1]
        
        # Add transverse resistance and longitudinal conductance cumulatively
        resistance.append(resistance[-1] + (Resistivity_Model[i] * thickness))
        conductance.append(conductance[-1] + (thickness / Resistivity_Model[i]))
    
    # Convert lists to 1D numpy arrays
    resistance = np.array(resistance)
    conductance = np.array(conductance)
    
    return resistance, conductance



"""
Calculation of the Equivalent Resistivity (TR-LC) by Using the Transverse Resistance and Longitudinal Conductance
"""

def Equivalent_resistivity_TR_LC(Transverse_Resistance, Longitudinal_Conductance):

    """
    Compute the equivalent resistivity and conductivity based on 
    transverse resistance and longitudinal conductance.

    Parameters:
    - resistance: 1D numpy array of transverse resistance values.
    - conductance: 1D numpy array of longitudinal conductance values.

    Returns:
    - equivalent_res: 1D numpy array of equivalent resistivity values.
    - equivalent_con: 1D numpy array of equivalent conductivity values.
    """
    
    # Validate input types and shapes
    if not (isinstance(Transverse_Resistance, np.ndarray) and isinstance(Longitudinal_Conductance, np.ndarray)):
        raise TypeError("Both resistance and conductance must be numpy arrays.")
    if Transverse_Resistance.ndim != 1 or Longitudinal_Conductance.ndim != 1:
        raise ValueError("Both resistance and conductance must be 1D arrays.")
    if Transverse_Resistance.shape != Longitudinal_Conductance.shape:
        raise ValueError("Resistance and conductance arrays must have the same shape.")
    
    # Calculate equivalent resistivity and conductivity
    equivalent_res = np.sqrt(Transverse_Resistance / Longitudinal_Conductance)
    
    return equivalent_res



"""
Calculation of the Equivalent Resistivity by Using the Transverse Resistance
"""

def Equivalent_resistivity_TR(Transverse_Resistance,Depth):

    Equivalent_resistivity_TR = Transverse_Resistance / Depth
            
    return Equivalent_resistivity_TR



"""
Calculation of the Equivalent Resistivity by Using the Longitudinal Conductance
"""

def Equivalent_resistivity_LC(Longitudinal_Conductance,Depth):

    Equivalent_resistivity_LC = Depth / Longitudinal_Conductance
            
    return Equivalent_resistivity_LC




#Functions to compute different resistance models


"""
Calculation of the Resistance by using the Equivalent Resistivity computed only from the Longitudinal Conductance
"""

def Resistance_From_Equivalent_Resistivity(equivalent_resistivity,depth):
    
    Resistance= equivalent_resistivity / depth

    return Resistance




#Functions to build the rescaling function by using a cubic spline


"""
Retrieval of the depth points when R(z) ~ Z(f)
"""
def Model_Data_matching(Resistance_Data, Resistance_Model, Depth_Model):
    idx_array = np.abs(Resistance_Model[:, None] - Resistance_Data).argmin(axis=0)
    
    # Map indices to corresponding depths
    Frequency_Depth_Points = Depth_Model[idx_array]


    return np.array(Frequency_Depth_Points)



#Functions to discretize the layered resistivity starting from a given resistance.

"""
Discretization of the Layered Resistivity starting from a Resistance model obtained by using the longitudinal conductance only.
"""

def Layered_Resistivity_from_harmonic_resistance_old(Harmonic_Resistance, Depth):
    """
    Compute layered resistivity from harmonic resistance and depth.

    Parameters:
    - Harmonic_Resistance (array-like): Array of harmonic resistance values.
    - Depth (array-like): Array of depth values.

    Returns:
    - numpy.ndarray: Array of layered resistivity values.
    """
    Harmonic_Resistance = np.asarray(Harmonic_Resistance)
    Depth = np.asarray(Depth)
    
    data = len(Harmonic_Resistance)
    Layered_Resistivity = np.zeros(data)
    Longitudinal_Conductance = np.zeros(data)
    
    for i in range(data):
        if i == 0:
            Layered_Resistivity[i] = Harmonic_Resistance[i] * Depth[i]
            Longitudinal_Conductance[i] = 1 / Harmonic_Resistance[i]
        else:
            Thickness = Depth[i] - Depth[i - 1]
            Factor = Thickness
            Factor1 = (1 / Harmonic_Resistance[i]) - Longitudinal_Conductance[i - 1]
            
            Resistivity = Factor / Factor1 if Factor1 > 0 else Layered_Resistivity[i - 1]
            
            Layered_Resistivity[i] = max(Resistivity, 1e-5)  # Ensure resistivity doesn't fall below threshold
            Longitudinal_Conductance[i] = (
                Thickness / Layered_Resistivity[i] + Longitudinal_Conductance[i - 1]
            )
    
    return Layered_Resistivity


def Layered_Resistivity_from_harmonic_resistance(Harmonic_Resistance, Depth):
    """
    Compute layered resistivity from harmonic resistance and depth.

    Parameters:
    - Harmonic_Resistance (array-like): Array of harmonic resistance values.
    - Depth (array-like): Array of depth values.

    Returns:
    - numpy.ndarray: Array of layered resistivity values.
    """
    Harmonic_Resistance = np.asarray(Harmonic_Resistance)
    Depth = np.asarray(Depth)
    
    data = len(Harmonic_Resistance)
    
    # Precompute depths differences and harmonic resistances inverse
    Thickness = np.diff(Depth, prepend=0)  # Calculate Depth[i] - Depth[i-1] with prepended 0 for i=0
    Inverse_Harmonic_Resistance = 1 / Harmonic_Resistance
    
    # Initialize arrays
    Layered_Resistivity = np.zeros(data)
    Longitudinal_Conductance = np.zeros(data)
    
    # First element
    Layered_Resistivity[0] = Harmonic_Resistance[0] * Depth[0]
    Longitudinal_Conductance[0] = Inverse_Harmonic_Resistance[0]
    
    # Vectorized operations for the rest
    for i in range(1, data):
        Factor1 = Inverse_Harmonic_Resistance[i] - Longitudinal_Conductance[i - 1]

        Resistivity = np.where(Factor1 > 0, Thickness[i] / Factor1, Layered_Resistivity[i - 1])
        
        Layered_Resistivity[i] = np.maximum(Resistivity, 1e-5)  # Apply the threshold
        Longitudinal_Conductance[i] = Thickness[i] / Layered_Resistivity[i] + Longitudinal_Conductance[i - 1]


    return Layered_Resistivity


def count_slope_segments(data, threshold=0.1):
    
    # Initialize variables
    num_segments = 0
    last_slope = None

    for i in range(1, len(data)):
        # Check if the slope has changed significantly
        if last_slope is None or np.abs(data[i] - last_slope) > threshold:
            num_segments += 1
            last_slope = data[i]  # Update the last slope to the new segment's slope

    return num_segments




#Functions to create different plots


def Plot_spline_frequency_depth(X, Y, params, x_label='Frequency [Hz]', y_label='Depth [m]', log_scale=True, save_path=None):
    """
    Cubic Spline Interpolation and Plotting for Frequency-Depth Data.

    Parameters:
    - X: Array-like, independent variable (e.g., frequency)
    - Y: Array-like, dependent variable (e.g., depth)
    - params: Parameters for spline interpolation (scipy splev)
    - x_label: Label for the x-axis
    - y_label: Label for the y-axis
    - log_scale: Whether to use logarithmic scaling for the x-axis
    - save_path: Path to save the plot (optional)
    """
    # Perform spline interpolation
    Y_pred = interpolate.splev(X, params, der=0)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.scatter(1 / X, Y, color='red', label='Data Points', zorder=3)
    plt.plot(1 / X, Y_pred, color='blue', label='Cubic Spline', zorder=2)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title("Frequency-Depth Cubic Spline Interpolation", fontsize=14)
    plt.ylim(0, max(max(Y), max(Y_pred)) + 500)  # Adjust y-limit dynamically
    plt.gca().invert_yaxis()  # Invert depth axis
    if log_scale:
        plt.xscale('log')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


#%% Additional functions 

"""
Function to find the closest value in an array
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


"""
Calculation of the Skin Depth
"""
def skin_depth(Z,frequencies):
    
    Skin_Depth=abs(Z)/(pi*frequencies*mu0)*(1/np.sqrt(2)) #Skin-Depth
                          
    return Skin_Depth



"""
Apparent Resistivity Calculation (Vozof)
"""
def apparent_resistivity(Z,frequencies):
    ang_freq=2*pi*(frequencies)
    App_Resistivity=abs(Z)**2 / (ang_freq * mu0)
    
    return App_Resistivity





# Function to create truncated copies of attributes
def Truncate_Data(Model, attributes, cutoff_index, suffix="_Data"):
    """
    Create truncated copies of attributes in an instance where the sufix "Data" refers to the truncated ones.

    Parameters:
    - instance: Object containing attributes to copy and truncate.
    - attributes (list of str): List of attribute names to truncate and copy.
    - cutoff_index (int): Index to truncate arrays at.
    - suffix (str): Suffix to add to the truncated attribute names.
    """
    for attr in attributes:
        original = getattr(Model, attr)
        truncated = original[0:cutoff_index]
        setattr(Model, f"{attr}{suffix}", truncated)
    



"""
Function to Save a family of models with a given number of attributes in a H5 file (memory efficient).
"""
def save_stations_to_hdf5(path, name, stations):
    filename=path+name
    
    with h5py.File(filename, "w") as f:
        for station in stations:
            # Create a group for each station
            group = f.create_group(station.model)
            
            # Save the station's properties as datasets
            group.create_dataset("Resistivity_Vector", data=station.res, compression="gzip")
            group.create_dataset("Depth_Vector", data=station.depth, compression="gzip")
            
              
            group.create_dataset("Resistivity_Model", data=station.Model_Res, compression="gzip")
            group.create_dataset("Depth_Model", data=station.Model_Depth, compression="gzip")
            
            group.create_dataset("Harmonic_Resistivity_Model", data=station.Harmonic_Resistivity_Model, compression="gzip")
            group.create_dataset("Harmonic_Resistance_Model", data=station.Harmonic_Resistance_Model, compression="gzip")
            
            group.create_dataset("Model_Frequencies", data=station.Frequencies, compression="gzip")
            group.create_dataset("Model_Periods", data=station.Periods, compression="gzip")
                        
            group.create_dataset("Survey_Frequencies", data=station.Frequencies_Data, compression="gzip")
            group.create_dataset("Survey_Periods", data=station.Periods_Data, compression="gzip")
            
            group.create_dataset("Impedance", data=station.Impedance, compression="gzip")
            group.create_dataset("Phase", data=station.Phase, compression="gzip")
            
            
            group.create_dataset("Impedance_Data", data=station.Impedance_Data, compression="gzip")            
            group.create_dataset("Phase_Data", data=station.Phase_Data, compression="gzip")
            group.create_dataset("Resistance_Data", data=station.Resistance_Data, compression="gzip")
            group.create_dataset("Skin_Depth_Data", data=station.Skin_Depth_Data, compression="gzip")


"""
Function to Save the EM field propapation computed by lowering a given source (memory efficient).
"""
def EM_Fields_save_matrix(directory, filename, matrix):
    """
    Save a matrix to a text file.

    Parameters:
        directory (str): Directory to save the file.
        filename (str): Name of the file.
        matrix (ndarray): Matrix to save.
    """
    filepath = os.path.join(directory, filename)
    np.savetxt(filepath, matrix)
    print(f"Saved {filename} to {directory}")


"""
Function to save a model saved as a npz file.
"""


def Save_npz_data(data, path, training=False):
    save_dict = {
        "model": data.model,
        "samples":data.samples,
        "res": data.res,
        "depth": data.depth,
        "Frequency_Vector": data.Log_Frequencies,
        "Max_Depth_Factor": data.Max_Depth_Factor,
        
        "Frequencies": data.Frequencies,
        "Periods": data.Periods,
        
        "Resistivity_Model": data.Resistivity_Model,
        "Depth_Model": data.Depth_Model,
        
        "Harmonic_Resistance_Model": data.Harmonic_Resistance_Model,
        "Harmonic_Resistivity_Model": data.Harmonic_Resistivity_Model,
        
        "Frequencies_Data": data.Frequencies_Data,
        "Periods_Data": 1 / data.Frequencies_Data,
        "Resistance_Data": data.Resistance_Data,
        "Impedance_Data": data.Impedance_Data,
        "Impedance_Data_Imag": data.Impedance_Imag_Data,
        "Phase_Data": data.Phase_Data,
        "Skin_Depth": data.Skin_Depth
    }

    # Add training-specific keys if training is True
    if training:
        save_dict.update({ 
            "Depth_Frequency_Points": data.Depth_Frequency_Points,
            #"Retrieved_Resistance_Points": data.Retrieved_Resistance_Points,
            "Discretized_Layered_Resistivity": data.Discretized_Layered_Resistivity,
        })

    # Save the dictionary
    np.savez_compressed(path + data.model, **save_dict)



def Save_npz_Model(data, path):
    np.savez_compressed(path + data.model,
                        
                        model=data.model,
                        samples=data.samples,
                        res=data.res,
                        depth=data.depth,
                        Frequency_Vector=data.Log_Frequencies,
                        Max_Depth_Factor=data.Max_Depth_Factor,
                        
                        Frequencies=data.Frequencies,
                        Periods=data.Periods,
                        
                        Resistivity_Model=data.Resistivity_Model,
                        Depth_Model=data.Depth_Model,
                        
                        Harmonic_Resistance_Model=data.Harmonic_Resistance_Model,
                        Harmonic_Resistivity_Model=data.Harmonic_Resistivity_Model,
                        
                        Frequencies_Data= data.Frequencies_Data,
                        Periods_Data= 1/data.Frequencies_Data,
                        Resistance_Data=data.Resistance_Data,
                        Impedance_Data=data.Impedance_Data,
                        Phase_Data=data.Phase_Data,
                        Skin_Depth=data.Skin_Depth)

    #print(f"Model information saved to {path}{data.model}.npz")
    

"""
Function to read a model saved as a npz file.
"""

def Load_npz_Model(filepath,training=False):
    
    data = np.load(filepath)
    model_number = str(data['model'])
    samples=np.array(data['samples'])
    res = [float(round(val, 1)) for val in data['res']]
    depth = data['depth'].astype(np.int64).tolist()

    #frequencies=np.logspace(data['Frequency_Vector'][0], data['Frequency_Vector'][1], samples)
    
    max_depth = int(depth[-1] * data['Max_Depth_Factor'])
    
    model_instance = Models_MT(model_number, samples, res, depth, data['Frequencies'], max_depth)
    
    
    # Assign data to the model
    model_instance.Resistivity_Model = data['Resistivity_Model']
    model_instance.Depth_Model = data['Depth_Model']
    model_instance.Harmonic_Resistance_Model = data['Harmonic_Resistance_Model']
    model_instance.Harmonic_Resistivity_Model = data['Harmonic_Resistivity_Model']
    model_instance.Frequencies_Data = data['Frequencies_Data']
    model_instance.Periods_Data = data['Periods_Data']
    model_instance.Resistance_Data = data['Resistance_Data']
    model_instance.Impedance_Data = data['Impedance_Data']
    #model_instance.Impedance_Imag_Data = data['Impedance_Imag_Data']
    model_instance.Phase_Data = data['Phase_Data']
    model_instance.Skin_Depth = data['Skin_Depth']

    if training:
        model_instance.Depth_Frequency_Points = data['Depth_Frequency_Points']
        #model_instance.Retrieved_Resistance_Points = data['Retrieved_Resistance_Points']
        model_instance.Discretized_Layered_Resistivity = data['Discretized_Layered_Resistivity']


    print(f"----- {model_number} loaded successfully -----")
    return model_instance


#%%
"""
Classes
"""

class Models_MT(object):

    def __init__(self, model, samples, res, depth, frequencies, max_depth, offset=0):
        self.model = model
        self.res = res
        self.samples=samples
        #self.con= [1/x for x in res]
        self.depth = depth
        self.Frequencies=frequencies
        self.Periods=1/frequencies
        self.offsets=offset
        self.ang_freq=2*pi*(frequencies)
        self.max_depth=max_depth

    def __str__(self):
        return ("MT Model Parameters:\n"
                f"Model: {self.model}\n"
                f"Number of Layers in the Model: {len(self.res[1:-1])}\n"
                f"Layer Resistivities: {self.res[1:-1]} [\u03A9m]\n"
                f"Layer Depths: {self.depth[1:-1]} [m]\n"
        )


    def Compute_Models(self, delta=1):

        """
        Generation of the Resistivity and Conductivity Models
        """
        [self.Resistivity_Model, self.Conductivity_Model, self.Depth_Model] = layered_models_variable_dz(self.depth, self.res, self.max_depth, delta=delta)  

        [self.Transverse_Resistance_Model, self.Longitudinal_Conductance_Model]=transverse_resistance_longitudinal_conductance(self.Resistivity_Model, self.Depth_Model)

        #self.TR_LC_Equivalent_Resistivity_Model=  Equivalent_resistivity_TR_LC(self.Transverse_Resistance_Model, self.Longitudinal_Conductance_Model)

        #self.TR_Equivalent_Resistivity_Model=  Equivalent_resistivity_TR(self.Transverse_Resistance_Model,self.Depth_Model)

        self.Harmonic_Resistivity_Model=  Equivalent_resistivity_LC(self.Longitudinal_Conductance_Model,self.Depth_Model)


        #self.TR_LC_Resistance_Model=  Resistance_From_Equivalent_Resistivity(self.TR_LC_Equivalent_Resistivity_Model, self.Depth_Model)

        #self.TR_Resistance_Model=  Resistance_From_Equivalent_Resistivity(self.TR_Equivalent_Resistivity_Model,self.Depth_Model)

        self.Harmonic_Resistance_Model = Resistance_From_Equivalent_Resistivity(self.Harmonic_Resistivity_Model, self.Depth_Model)



    def Simulate_MT_data(self,Case_2D=False):

        """
        Simulation of the Electric and Magnetic Fields
        """
        [self.ex_field, self.hy_field, self.ey_field, self.hx_field]=mt_simulation_fields(self.res, self.depth,self.Frequencies,offset=self.offsets, Case_2D=Case_2D)      
        
        """
        Simulation of the Impedance and Phase
        """
        [self.Impedance, self.Phase] = impedance_1D(self.ex_field,self.hy_field)

        """
        Computation of the Skin Depth for the simulated MT Data
        """
        self.Skin_Depth=skin_depth(self.Impedance,self.Frequencies)

        

    def Simulate_EM_Fields(self, file_path=None, delta=10, write_output=False):

        [self.Simulated_Elec_Field, self.Simulated_Mag_Field, self.Simulated_Imp_Field, 
         self.Simulated_Amp_Elec_Field, self.Simulated_Amp_Mag_Field, self.Simulated_Amp_Imp_Field, 
         self.Simulated_Depth_Field] = EM_Field_propagation(self.model, self.res, self.depth, self.Frequencies,write_output, file_path, delta)
        

    
    def Truncate_MT_data(self, attributes, truncate=True, depth_cutoff=1.5):

        """
        This function aims to truncate the simulated data up to a given physical limit (when the impedance has reached the halfspace), the function leaves 
        the simulated data intact and retuns a copy of the desired attributes, with the sufix "Data"
        """
        if truncate:

            Depth=self.depth[-1]*depth_cutoff
            index = int(np.where(self.Depth_Model == Depth)[0])

            self.Depth_Cutoff_Index = len(self.Impedance.real[self.Impedance.real > self.Harmonic_Resistance_Model[index]])

        else: 
            
            self.Depth_Cutoff_Index=len(self.Frequencies)

        Truncate_Data(self, attributes, self.Depth_Cutoff_Index, suffix="_Data")

        self.Resistance_Data=self.Impedance_Data.real

        self.Impedance_Imag_Data=abs(self.Impedance_Data.imag)



    def Compute_Apparent_resistivity(self): #(To Check)

        """
        Simulation of the Apparent Resistivity
        """        
        self.Apparent_Resistivity = apparent_resistivity(self.Impedance,self.Frequencies)

        """
        Calculation of the Niblet-Bostick Depth (To Check)
        """
        #[self.App_Resistivity, self.App_Conductivity, self.Estimated_Depth] = treviño_depth(self.Apparent_Resistivity, self.Frequencies, self.max_depth)  



    def Simulate_FDEM_data(self):
        """
        Simulation of the Secondary Magnetic Field in Percentage
        """
        self.Secondary_Magnetic_Field=FDEM_simulation(self.con, self.thickness, self.frequencies, len(self.thickness))



    def Simulate_TDEM_data(self):
        """
        Simulation of the Magnetic Flux
        """
        self.Magnetic_Flux=TDEM_simulation(self.con, self.thickness, self.periods, len(self.thickness))



    def Rescale_Data(self):

        """
        Calculation of the Depth points to do the Frequency-Depth Pairs
        """
        self.Depth_Frequency_Points=Model_Data_matching(self.Resistance_Data, self.Harmonic_Resistance_Model, self.Depth_Model)

        """Depth-Frequency Cubic Spline Modeling """

        self.Spline_Factors= interpolate.splrep(1/self.Frequencies_Data, self.Depth_Frequency_Points, k=3, s=0)
 
    
        """Depth-Frequency Cubic Spline Rescaling """
    
        self.Rescaled_Depth=interpolate.splev(1/self.Frequencies_Data, self.Spline_Factors, der=0)    

    
        """Harmonic Resistance Rescaling""" 

        self.Discretized_Layered_Resistivity=Layered_Resistivity_from_harmonic_resistance(self.Resistance_Data, self.Rescaled_Depth)



    def Layered_Resistivity_Retrieval(self):
        from scipy.interpolate import PchipInterpolator

        """
        Calculation of the Depth points to do the Resistivity Retrieval
        """

        #self.Depth_Frequency_Points=Model_Data_matching(self.Resistance_Data, self.Harmonic_Resistance_Model, self.Depth_Model)

        """Resistance Model Cubic Spline Modeling """

        #self.Spline_Factors= interpolate.splrep(self.Harmonic_Resistance_Model[::-1], self.Depth_Model[::-1] ,k=3, s=0)
        
        pchip_interp = PchipInterpolator(np.log10(self.Harmonic_Resistance_Model[::-1]), np.log10(self.Depth_Model[::-1]))

        #pchip_interp2 = PchipInterpolator(self.Harmonic_Resistance_Model[::-1], self.Depth_Model[::-1])
        
        self.Depth_Frequency_Points = pchip_interp(np.log10(self.Resistance_Data[::-1]))

        #self.Depth_Frequency_Points2 = pchip_interp2(self.Resistance_Data[::-1])

        #self.Depth_Frequency_Points2= self.Depth_Frequency_Points2[::-1]
        #self.Depth_Frequency_Points=interpolate.splev(self.Resistance_Data[::-1], self.Spline_Factors, der=0)  

        self.Depth_Frequency_Points= 10**(self.Depth_Frequency_Points[::-1])


        """Resistivity Retrieval """

        self.Discretized_Layered_Resistivity=Layered_Resistivity_from_harmonic_resistance(self.Resistance_Data, self.Depth_Frequency_Points)       
        

    def Logarithmic_Data(self, log="natural", training=False, inversion=False):

        """
        Calculation of the Natural Logarithm for the Data
        """

        if log.lower() !="none":

            if log.lower() =="natural":
                self.Resistance_Data_log=np.log1p(self.Resistance_Data)

                self.Impedance_Imag_Data_log=np.log1p(self.Impedance_Imag_Data)

                self.Frequencies_Data_log=np.log1p(self.Frequencies_Data)

                self.Phase_Data_log=np.log1p(self.Phase_Data)

                self.Skin_Depth_log=np.log1p(self.Skin_Depth)

                if training:

                    #self.Retrieved_Resistance_Points_log=np.log1p(self.Retrieved_Resistance_Points)

                    self.Depth_Frequency_Points_log=np.log1p(self.Depth_Frequency_Points)

                    self.Discretized_Layered_Resistivity_log=np.log1p(self.Discretized_Layered_Resistivity)

                    if inversion:

                        self.Depth_Model_Data_log=np.log1p(self.Depth_Model_Data)
                        self.Harmonic_Resistance_Model_Data_log=np.log1p(self.Harmonic_Resistance_Model_Data)

            elif log.lower() =="base_10":

                self.Resistance_Data_log=np.log10(self.Resistance_Data)

                self.Impedance_Imag_Data_log=np.log10(self.Impedance_Imag_Data)

                self.Frequencies_Data_log=np.log10(self.Frequencies_Data)

                self.Phase_Data_log=np.log10(self.Phase_Data)

                self.Skin_Depth_log=np.log10(self.Skin_Depth)

                if training:

                    #self.Retrieved_Resistance_Points_log=np.log10(self.Retrieved_Resistance_Points)

                    self.Depth_Frequency_Points_log=np.log10(self.Depth_Frequency_Points)

                    self.Discretized_Layered_Resistivity_log=np.log10(self.Discretized_Layered_Resistivity)

                    if inversion:

                        self.Depth_Model_Data_log=np.log10(self.Depth_Model_Data)
                        self.Harmonic_Resistance_Model_Data_log=np.log10(self.Harmonic_Resistance_Model_Data)

        




class Reconstructed_Models_MT(object):

    def __init__(self, model, res, depth, frequencies, max_depth, Impedance, Phase, Skin_Depth):
        self.model = model
        self.res = res
        self.con= [1/x for x in res]
        self.depth = depth
        self.Frequencies=frequencies
        self.Periods=1/frequencies
        self.max_depth=max_depth

        self.Impedance=Impedance

        self.Phase=Phase

        self.Skin_Depth=Skin_Depth

        
    def Compute_Models(self, delta=1):

        """
        Generation of the Resistivity and Conductivity Models
        """
        [self.Resistivity_Model, self.Conductivity_Model, self.Depth_Model] = layered_models_variable_dz(self.depth,self.res,self.max_depth, delta=delta)  

        [self.Transverse_Resistance_Model, self.Longitudinal_Conductance_Model]=transverse_resistance_longitudinal_conductance(self.Resistivity_Model, self.Depth_Model)

        self.Harmonic_Resistivity_Model=  Equivalent_resistivity_LC(self.Longitudinal_Conductance_Model,self.Depth_Model)

        self.Harmonic_Resistance_Model = Resistance_From_Equivalent_Resistivity(self.Harmonic_Resistivity_Model, self.Depth_Model)


    def Truncate_MT_data(self, attributes, depth_cuttof=1.5):

        """
        This function aims to truncate the simulated data up to a given physical limit (a maximum penetration based on the skin depth), the function leaves 
        the simulated data intact and retuns a copy of the desired attributes, with the sufix "Data"
        """

        self.Depth_Cutoff_Index=find_nearest(self.Skin_Depth,self.depth[-1]*depth_cuttof)

        Truncate_MT_Data(self, attributes, self.Depth_Cutoff_Index, suffix="_Data")

        self.Resistance_Data=self.Impedance_Data.real

        self.Phase_Data_Real=self.Phase_Data.real



    def Rescale_Data(self):

        """
        Calculation of the Depth points to do the Frequency-Depth Pairs
        """
        self.Depth_Frequency_Points=Model_Data_matching(self.Resistance_Data, self.Harmonic_Resistance_Model, self.Depth_Model)

        """Depth-Frequency Cubic Spline Modeling """

        self.Spline_Factors= interpolate.splrep(1/self.Frequencies_Data, self.Depth_Frequency_Points, k=3, s=0)
 
    
        """Depth-Frequency Cubic Spline Rescaling """
    
        self.Rescaled_Depth=interpolate.splev(1/self.Frequencies_Data, self.Spline_Factors, der=0)    

    
        """Harmonic Resistance Rescaling"""

        self.Discretized_Layered_Resistivity=Layered_Resistivity_from_harmonic_resistance(self.Resistance_Data, self.Rescaled_Depth)



        

#Checked 
#%%
