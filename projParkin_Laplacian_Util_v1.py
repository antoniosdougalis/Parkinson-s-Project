# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 02:30:38 2026

written by Antonios Dougalis, Feb 2026, Kozani Greece
contact: antoniosdougalis (at) gmail.com; antoniosdougalis (at) med.uoc.gr 
"""

import numpy as np
import mne
from scipy.special import legendre

# definitions of functions

def create_HeadMontage_RawInfo(fs=256):

    # load BioSemi32 montage
    montage = mne.channels.make_standard_montage('biosemi32')

    # channel names
    ch_names = montage.ch_names

    # create info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types='eeg'
    )

    # attach montage
    info.set_montage(montage)

    # electrode coordinates dictionary
    pos = montage.get_positions()['ch_pos']
    electrode_coordinates = {ch: pos[ch] for ch in ch_names}

    return info, ch_names, electrode_coordinates


def generate_Head_CartCoord(electrode_coordinates):
        
    x,y,z = [ np.zeros(len(electrode_coordinates)) for _ in range(3)]
    sensor_labels = list(electrode_coordinates.keys())
    
    for idx, (electrode, coordinates) in enumerate(electrode_coordinates.items()):
        x[idx] = coordinates[0]
        y[idx] = coordinates[1]
        z[idx] = coordinates[2]
        
    return x,y,z, sensor_labels   


def laplacian_spatialFiltering(data, x, y, z, leg_order=10, smoothing=1e-5):
    
    """
    Compute surface Laplacian of EEG data using the Perrin method.
    'modification on original description by Mike X Cohen in Matlab,
    see Analysing Neural Time Series. Theory and Practise, the MIT Press'
    
    Parameters:
    data : numpy.ndarray
        EEG data (electrodes X time/trials).
    x, y, z : numpy.ndarray
        x, y, z coordinates of electrode positions.
    leg_order : int, optional
        Order of the Legendre polynomial (default is 10, or 12 for >100 electrodes).
    smoothing : float, optional
        Smoothing parameter (default is 1e-5).
    
    Returns:
    EEGlapl : numpy.ndarray
        Surface Laplacian of EEG data.
    G, H : numpy.ndarray
        G and H weight matrices.
           
    """
    
    # initialise some parameters
    numelectrodes = data.shape[0]
    m = 4;
    leg_order = leg_order
    smoothing = smoothing
    
    # convert Cartesian to Polar coordinates and use radius for each electrode
    def cart2sph(x, y, z):
        """
        Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, r).
        
        Parameters:
            x, y, z: Cartesian coordinates.
        
        Returns:
            azimuth, elevation, r: Spherical coordinates (in radians for azimuth and elevation, meters for r).
        """
        spherical_radii = np.sqrt(x**2 + y**2 + z**2) # spherical radii, distance form center of head [0,0,0]
        azimuth = np.arctan2(y, x)  # Azimuth phi angle (in radians), XY angle
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation theta angle (in radians), XZ angle
        
        return azimuth, elevation, spherical_radii
    

    # extract polar coordinates and scale XYZ coordinates to unit sphere
    azimuth, elevation, spherical_radii = cart2sph(x, y, z) 
    maxrad = np.max(spherical_radii)
    x, y, z = x / maxrad, y / maxrad, z / maxrad
    
    # Compute cosine distance matrix of all electrode pairs
    cosdist = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(numelectrodes):
            temp =  (x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2 
            cosdist[i, j] = 1 - ( temp / 2)
   
    cosdist = cosdist + cosdist.T + np.eye(numelectrodes)
    
    # Initialize the Legendre polynomial array
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))

    # Compute Legendre polynomials
    for ni in range(1, leg_order+1):
        # Get the Legendre polynomial of order 'ni'
        poly = legendre(ni)  # scipy returns a polynomial object
        legpoly[ni-1, :, :] = poly(cosdist)  # Evaluate the polynomial at the cosine distances
        
        
    # Precompute electrode-independent variables
    twoN1 = 2 * np.arange(1, leg_order+1) + 1
    gdenom = ( np.arange(1, leg_order+1) * ( np.arange(1, leg_order+1) + 1) )**m
    hdenom = ( np.arange(1, leg_order+1) * ( np.arange(1, leg_order+1) + 1) )**(m-1)
    
    # Compute G and H weight matrices
    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(numelectrodes):
            g = 0
            h = 0
            for ni in range(leg_order):
                g = g + (twoN1[ni] * legpoly[ni, i, j]) / gdenom[ni]
                h = h - (twoN1[ni] * legpoly[ni, i, j]) / hdenom[ni]
            G[i, j] =  g / (4 * np.pi)
            H[i, j] = -h / (4 * np.pi)
    
    # mirror matrix (run if j index above starts run from i index: No need to use if the j run from 0 as above)
    # G = G + G.T
    # H = H + H.T
        
    # Correct for diagonal-double
    G = G - ( np.eye(numelectrodes) * ( G[0,0] / 2) )
    H = H - ( np.eye(numelectrodes) * ( H[0,0] / 2) )
    
    ## compute laplacian
    # Reshape data to electrodes X time X epochs/trials
    original_data_shape = np.shape(data) # electrodes X time X epoch/trials
    
    if len(original_data_shape)>=2:
        data = np.reshape(data, (original_data_shape[0], np.prod(original_data_shape[1:])))
        print(data.shape)
    elif len(original_data_shape) == 1:
        data = data[:, None]
        
    # Add smoothing constant to diagonal
    # (change G so output is unadulterated)
    Gs = G + np.eye(numelectrodes) * smoothing
    
    # Compute C matrix
    GsinvS = np.sum(np.linalg.inv(Gs), axis =1) # return row vector (equivalent to Matlab sum)
    print(f'shape of the GsinvS matrix is {GsinvS.shape}')
    dataGs = np.dot(data.T, np.linalg.inv(Gs)) # equivalent to MatLab data'/Gs
    print(f'shape of the dataGs matrix is {dataGs.shape}')
    
    scaling_factor = np.sum(dataGs, axis=1) / np.sum(GsinvS)
    scaling_factor = scaling_factor[:, None] # unsqueeze the second axis
    print(f'shape of the scaling_factor matrix is {scaling_factor.shape}')
    C = dataGs - scaling_factor*GsinvS
    print(f'shape of the C matrix is {C.shape}')
    
    # Compute surface Laplacian (reshape to original data size)
    EEGlapl = np.dot(C, H.T).T
    EEGlapl = EEGlapl.reshape(original_data_shape)
    
    return EEGlapl, G, H
