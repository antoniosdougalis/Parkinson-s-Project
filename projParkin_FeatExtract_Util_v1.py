# import general libraries for all functions

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4',
 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

#%% Compute time domain statistical moments

def compute_epoch_stats(epoched_EEG):
    """
    Compute mean, variance, and interquartile range (IQR) per epoch of EEG data.
    INPUTS:
    epoched_data : 3D array of shape (n_epochs, n_channels, n_timepoints)

    OUTPUTS:
    mean_vals : np.ndarray       
    var_vals : np.ndarray       
    iqr_vals : np.ndarray
       
    """
    
    # Mean across epochs (axis 0)
    mean_vals = np.mean(epoched_EEG, axis=-1)

    # Variance across epochs (axis 0)
    var_vals = np.var(epoched_EEG, axis=-1)

    # IQR across epochs (axis 0)
    q75 = np.percentile(epoched_EEG, 75, axis=-1)
    q25 = np.percentile(epoched_EEG, 25, axis=-1)
    iqr_vals = q75 - q25

    return mean_vals, var_vals, iqr_vals

#%% Compute Welch PSD for all epochs of all channels for all data

from scipy.signal import welch

def compute_psd_welch(epoched_EEG, fs, perOverlap=None, freq_res=None):
    """
    Compute the Welch PSD for each epoch and channel in EEG data.

    Parameters:
    -----------
    epoched_EEG : np.ndarray
        3D array with shape (epochs, channels, samples)
    fs : float
        Sampling frequency in Hz
    perOverlap : float
        Overlap percentage between windows (e.g., 0.5 for 50%)
    freq_res : float
        Desired frequency resolution in Hz

    Returns:
    --------
    psd : np.ndarray
        3D array with shape (epochs, channels, frequency_bins)
    f_vec : np.ndarray
        Frequency vector corresponding to the PSD values
    """
    if perOverlap is None or freq_res is None:
        perOverlap = 0.5
        freq_res   = 1
        
        
    n_pnts = epoched_EEG.shape[-1] # signal epoch sample points
    nperseg = min(int(fs / freq_res), n_pnts) # segment length in sample points AND true resolution
    noverlap = int(nperseg * perOverlap) # overlap in sample points
    
    # actual windonws calculationss
    step = nperseg - noverlap
    num_windows = 1 + (n_pnts - nperseg) // step  # actual computed windows that will be used
    
    nfft = 2*nperseg # (nfft must be >= nperseg) # define frequency points and resolution
    f_len = nfft // 2 + 1  # Length of PSD output
    
    trueRes = fs/nperseg # TRUE Resolution will dpendit on the actuall points in the segment and fs
    gridRes = fs/nfft # this is the intrpolated resoltion die to nfft zero padding
    
    print(f'Performing Welch PSD analysis for every EEG data epoch with:\n'
          f'- Number of windows per epoch: {num_windows}\n'
          f'- Points per window: {nperseg}\n'
          f'- Overlap: {perOverlap * 100}%\n'
          f'- True frequency resolution: {trueRes:.3f} Hz\n'
          f'- Grid frequency resolution: {gridRes:.3f} Hz\n'
          f'- Output frequency bins: {f_len}\n')

    out_psd = np.zeros((epoched_EEG.shape[0], epoched_EEG.shape[1], f_len))

    for ep in range(epoched_EEG.shape[0]):
        for chan in range(epoched_EEG.shape[1]):
            f_vec, out_psd[ep, chan, :] = welch(
                epoched_EEG[ep, chan, :],
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend='constant',
                return_onesided=True,
                scaling='density',
                axis=-1,
                average='mean'
            )

    print('PSD computation complete.')
    
    return out_psd, f_vec

# out_psd, f_vec = compute_psd_welch(epoched_EEG, fs=fs, perOverlap=0.5, freq_res=0.5)


#%% LOOP TO CALCULATE THE EXPONENTial FOR ALL ELECTRODES AND ALL EPOCHS

from fooof import FOOOF

def aperiodic_periodic_Spectral(out_psd, f_vec, peak_threshold= 2.0, min_peak_height= 1.5, aperiodic_mode='fixed', freq_range = None ):
    """
    Compute the aperiod and perdioc compmeonet from Welch PSD for each epoch and channel in EEG data.

    Parameters:
    -----------
    out_psd : np.ndarray
        3D array with shape (epochs, channels, samples)
    f_vec : np.ndarray
        1D array with shape (freq,)
    peak_threshold : float
        stanfard deviation for Gausain fits (default)
    min_peak_height : float
        Desired power in units of PSD for periodic peak detection (above the aperioidc component) 
    aperiodic_search mode: str
        'fixed or 'knee' for the shape of the PSD, default at fixed

    Returns:
    --------
    ap_params : np.ndarray
        3D array with shape for the aperiodic analysis (epochs, channels, 2): 0 is the offset and 1 is the aperiodic exponent
    peak_params : np.ndarray
        4D array with shape for the periodic analysis (epochs, channels, num_peaks, 3): 0 is the peak Central Frequency, 1 peak Power, 2 peak bandwidth
    r_squared: np.ndarray
       2D array with shape for the periodic analysis (epochs, channels): fit r_squared
    fit_error : np.ndarray
        2D array with shape for the periodic analysis (epochs, channels): fit error
    """
    
    ap_params   =  np.full( (out_psd.shape[0], out_psd.shape[1], 2), np.nan  ) 
    peak_params =  np.full( (out_psd.shape[0], out_psd.shape[1], 3), np.nan ) 
    r_squared, fit_error = [ np.full( (out_psd.shape[0], out_psd.shape[1]), np.nan ) for _ in range(2) ] 
    
    if freq_range is None:
        freq_range = [ 1, 45 ]
    else:
        freq_range = freq_range
    
    for ep in range(out_psd.shape[0]):
           
        for channi in range(out_psd.shape[1]):
            
            fm = FOOOF(peak_width_limits = [ np.round(3*(f_vec[1]-f_vec[0]),2), 12], 
                       peak_threshold  = peak_threshold, 
                       min_peak_height = min_peak_height, 
                       aperiodic_mode  = aperiodic_mode, 
                       verbose=False)
            
            # to avoid crushed of algorithm due to zeros (replace zeros with 0.01 time the minimum (excluding zeros)
            psd = out_psd[ep, channi, :]
            floor = 1e-15 if not np.any(psd > 0) else 0.01 * np.min(psd[psd > 0])
            psd = np.maximum(psd, floor)
            fm.fit(f_vec, psd, freq_range)

            ap_par, peak_par, r_sq, fit_er, gauss_params = fm.get_results()
            
            # get all data
            ap_params[ep, channi, :2] =  ap_par
            r_squared[ep, channi] = r_sq
            fit_error[ep, channi] = fit_er
            
            if len(peak_par)!=0:
                peak_params[ep, channi, :] = peak_par[0, :]
    
    return ap_params, peak_params, r_squared, fit_error
        
# ap_params, peak_params, r_squared, fit_error = aperiodic_periodic_Spectral(out_psd, f_vec, peak_threshold= 2.0, min_peak_height= 1.0, aperiodic_mode='fixed' )


# Import IRASA related functions
from neurodsp.aperiodic import compute_irasa, fit_irasa

def compute_irasa_aperiodic(epoched_EEG, fs, perOverlap=None, freq_res=None, freq_range = None):
    '''
    #IRASA algorithm for separationof periodic and aperiodic components using the neurDSP library
    NB the perOverlap=None, freq_res=None are used to match the spectral welch ppoperties and have analogues irasa determinations as the 
    determinations of FOOOF which are based on the welch. The irasa algorithm recompoutes the Spectrum using similar manners
    
    INPUTS: 
        epoched_EEG is of size epochs x channels x frequencies
        fs is the sampling rate of the file
        perOverla p: is the fracvtional overlap for spectral decomposition
        freq_res is the desired frequency resolution of the spectral decomposition
        freq_range is a list holding the low-high frequencies for fitting the aperiodic via irasa
        
    OUTPUT:
    irasa aperiodic exponent
    '''
    
    if perOverlap is None or freq_res is None or freq_range is None:
        perOverlap= 0.5
        freq_res = 0.5
        freq_range = [1, 45]
    else:
        perOverlap=perOverlap
        freq_res=freq_res
        freq_range = freq_range
    
            
    n_epochs, n_chans, n_pnts = epoched_EEG.shape # signal epoch sample points
    
    nperseg = min(int(fs / freq_res), n_pnts) # segment length in sample points AND true resolution
    noverlap = int(nperseg * perOverlap) # overlap in sample points   
    nfft = 2*nperseg # (nfft must be >= nperseg) # define frequency points and resolution
    f_len = nfft // 2 + 1  # Length of PSD output
    
    # actual windonws calculationss
    # step = nperseg - noverlap
    # num_windows = 1 + (n_pnts - nperseg) // step  # actual computed windows that will be used
    # trueRes = fs/nperseg # TRUE Resolution will dpendit on the actuall points in the segment and fs
    # gridRes = fs/nfft # this is the intrpolated resoltion die to nfft zero padding
    
    # construct the frequency vector
    niquist = fs/2
    irasa_f_vec = np.linspace(0, niquist, f_len)
    
    # print(f'Performing Irasa  PSD analysis for every EEG data epoch with:\n'
    #       f'- Number of windows per epoch: {num_windows}\n'
    #       f'- Points per window: {nperseg}\n'
    #       f'- Overlap: {perOverlap * 100}%\n'
    #       f'- True frequency resolution: {trueRes:.3f} Hz\n'
    #       f'- Grid frequency resolution: {gridRes:.3f} Hz\n'
    #       f'- Output frequency bins: {f_len}\n')
    
    
    # initialise containers
    # get the frequency vectro length for freq from freq:range[0] till freq_range[1] for itialising irasa output
    len_hz_freqRange = np.where(irasa_f_vec==freq_range[1])[0][0] - np.where(irasa_f_vec==freq_range[0])[0][0] + 1
    
    psd_Aperiodic, psd_Periodic = [ np.zeros( (n_epochs, n_chans, len_hz_freqRange) ) for _ in range(2) ];
    irasa_exp = np.zeros( (n_epochs, n_chans) ) ;

    for ep in range(n_epochs):
        for channi in range(n_chans):
            
                # Compute the IRASA decomposition of the data (Note that welch decomposition is done with the same parameters as the Previous step)
                freqs_irasa, psd_Aperiodic[ep, channi, :], psd_Periodic[ep, channi, :] = compute_irasa( epoched_EEG[ep, channi, :], fs, 
                                                            f_range = freq_range, hset = [1.1, 1.2, 1.3], nperseg=nperseg, noverlap=noverlap, nfft=nfft, thresh=1.0)
                
                # Fit the aperiodic component of the IRASA results
                # avoid crushes of Aperioidc fill of zeros
                if np.all(psd_Aperiodic[ep, channi, :] == 0):
                    irasa_exp[ep, channi] = np.nan
                    
                else:                   
                    intercept, fit_sl = fit_irasa(freqs_irasa, psd_Aperiodic[ep, channi, :])
                    irasa_exp[ep, channi] = fit_sl
    
    return irasa_exp    
            
#%% Connectivity analysis: Prepare the parameters
                
from mne_connectivity import spectral_connectivity_time

Freq_Bands = {"delta": [1.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0], "gamma": [30.0, 45.0]}
connectivity_methods = ["plv", 'pli']

def compute_mne_connectivity(epoched_EEG, fs, Freq_Bands = None, connectivity_methods = None):
    '''
    Parameters
    ----------
    epoched_EEG : nparray of dimensions epochs, channesl time points
        epoched data to be analysed.
    fs : int
        sampling rate of data.
    Freq_Bands : dictionary, band: list of min and high frequency 
         The default is {"delta": [1.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0], "gamma": [30.0, 45.0]}.
    connectivity_methods : list of connectivity meathdios suuports bu mne
        The default is ["plv", 'pli']. also coh or wpli can be used

    Returns
    -------
    diag_outCon_array : 5 D np array n_con_methods, n_epochs, n_channels, n_channels, n_freq_bands 
        Connectivity results per method choosen per epoch perc channel parir per freqency band .

    '''
    
    # get some parametrs out
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    
    # define analysis parametrs
    if Freq_Bands is None or connectivity_methods is None:
        Freq_Bands = {"delta": [1.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0], "gamma": [30.0, 45.0]}
        connectivity_methods = ["plv", 'pli']
    else: 
        Freq_Bands = Freq_Bands
        connectivity_methods = connectivity_methods
        
    # Freq bands of interest
    n_freq_bands = len(Freq_Bands)
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))
    
    # Provide the freq points
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))
    
    # The dictionary with frequencies are converted to tuples for the function
    fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
    fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])
    
    # We will try two different connectivity measurements as an example
    n_con_methods = len(connectivity_methods)

    #% Procedure for connectivity on the the epoch level
    # Pre-allocatate memory for the connectivity matrices
    n_cycles    = np.linspace(3, 9, len(freqs)) # for the wavelet cycles per frequency (alternative to full width a half max)
    
    diag_outCon_array = np.zeros( (n_con_methods, n_epochs, n_chans, n_chans, n_freq_bands ) )
    
    # concatenate the data together for the loop
    diag_outCon = spectral_connectivity_time(epoched_EEG,
                                 freqs    = freqs,
                                 method   = connectivity_methods,
                                 sfreq    = fs,
                                 mode     = 'cwt_morlet',
                                 n_cycles = n_cycles,
                                 fmin     = fmin, 
                                 fmax     = fmax,
                                 average  = False,
                                 faverage = True,
                                 verbose  = False)
    diag_outCon[0].shape  
      
    # Get data as connectivity matrices
    for c in range(n_con_methods):
        diag_outCon_array[c, :, :, :, :] = diag_outCon[c].get_data(output="dense")
    
    # check shape for sanity
    diag_outCon_array.shape
    
    return diag_outCon_array

#-----------------------------------------------------------------------------

# modified function to pot them all in the same plot   
def testplot_diagCond_matrix_ax( diag_con_data, ch_names,
    Freq_Bands, connectivity_methods, freqi, conni,
    ax=None):
    
    """
    Visualize the connectivity matrix on a provided axis, or create a new figure if none given.
    
    INPUTS:
        diag_con_data (np.ndarray): Phase time series (e.g., delta phase).
        ch_names (list of str): Channel names of the EEG montage.
        Freq_Bands (dict): Frequency bands (e.g., {'delta': [1, 4]}).
        connectivity_methods (list of str): Connectivity methods applied to the data.
        freqi (str): Frequency band key to plot.
        conni (str): Connectivity method to plot.
        ax (matplotlib.axes.Axes): Axis to plot into. If None, a new figure is created.
    
    OUTPUTS:
        Optional: Created figure if ax is None.
    """
    
    fi = list(Freq_Bands.keys()).index(freqi)
    ci = connectivity_methods.index(conni)

    # Create new figure and axis if none provided
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6 * len(Freq_Bands), 6))
        own_fig = True

    # Plot with imshow
    con_plot = ax.imshow(
        np.mean(diag_con_data[ci, :, :, :, fi], axis=0),
        cmap="jet", vmin=0, vmax=1
    )
    ax.set_title(f"{freqi}, {conni}")
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names)

    # Add colorbar only if creating own figure
    if own_fig:
        fig.colorbar(con_plot, ax=ax, shrink=0.7, label="Connectivity")
        return fig
    else:
        # Optional: you could return con_plot or nothing
        return None

            
#%% Phase clustering and Modulation Index

#% Create a complex Wavelet
def createComplexWaveletFamily(timeL, freq, fwhm, fs, PrintToggle):
    
    """
    Create a family of complex Morlet Wavelets centrered at specified Frequencies
    
    INPUTS:
        timeL (int): Time ins seconds used to create the wavelet time vector centrered to zero 
        (based on the integer provided and the sampling rate,  np.arange(-timeL*fs,timeL*fs+1)/fs ).
        freq (list): A list of central frequencies for the Wavelet.
        fwhm (list): A list of full width a half maximal in seconds.
        fs (int): The sampling rate required, based on the data t be analysed
        PrintToggle(Boolean): Controls a Figure of Wavelets in Time and Frequency domain for Validation and inspection
        
    OUTPUTS:
        sineFamily: The sine wave component of the Wavelet
        gausFamily: The dampening Gaussian wave for the Wavelet
        waveletFamily: The unnormalised wavelets
        wavtime: The wavelet tme vector (to be used for the convolution
        Figure: for inspection of wavelets at time and Frequency domain
        
    """
    # cretae wavelet vector
    wavtime = np.arange(-timeL*fs,timeL*fs+1)/fs
    npnts = len(wavtime)
    
    waveletFamily = np.zeros( ( len(freq), npnts ), dtype=np.complex64)
    sineFamily    = np.zeros( ( len(freq), npnts ), dtype=np.complex64)
    gausFamily    = np.zeros( ( len(freq), npnts ), dtype=np.float64)
    
    # frequency axis for the power spectrum of wavelet
    hz_axis = np.linspace(0, fs/2, int( npnts/2)+1)
    
    for fi in range(len(freq)):
        sinepart = np.exp( 1j*2*np.pi*freq[fi]*wavtime )
        gauspart = np.exp( (-4*np.log(2)*wavtime**2)/(fwhm[fi]**2) )
        wavelet = sinepart * gauspart
  
        waveletFamily[ fi, : ] =  wavelet
        sineFamily[ fi, : ]    =  sinepart
        gausFamily[ fi, : ]    =  gauspart
        
    # for plotting the time and frequency domain representation of the wavelets
    if PrintToggle == True:
    
        # setup the figure
        fig,ax = plt.subplots(len(freq),2,figsize=(15,10))
    
        for j in range(len(freq)):
            
            # power spectrum of wavelet
            waveletX = ( 2*np.abs(scipy.fftpack.fft(waveletFamily[ j, : ])/npnts) )**2
            
            # time-domain version
            ax[j,0].plot(wavtime,np.real(waveletFamily[ j, : ]),'k', label = 'real')
            ax[j,0].plot(wavtime,gausFamily[ j, : ],'r', label = 'gaussian')
            ax[j,0].plot(wavtime,np.imag(waveletFamily[ j, : ]),'g', label = 'imag')
            ax[j,0].plot([-2, 2],[0.5, 0.5], 'b--', label = 'fwhm')
            ax[j,0].set_xlim([-0.5, 0.5])
            # ax[j,0].set_ylim([0 , 1])
            ax[j,0].legend()
            
            ax[j,0].set_xlabel('Time (s)')
            ax[j,0].set_ylabel('Normalised Amplitude (a.u.)')
            ax[j,0].set_title('Time domain')
            
            # frequency-domain version
            # ax[j,1].stem(hz,waveletX[:len(hz)],'k')#,use_line_collection=True)
            ax[j,1].plot(hz_axis,waveletX[:len(hz_axis)],'m')
            ax[j,1].set_xlim([0,65])
            # ax[j,1].set_ylim([-0.01, 0.01])
            ax[j,1].set_xlabel('Frequency(s)')
            ax[j,1].set_ylabel('Wavelet Power')
            ax[j,1].set_title(f'Frequency domain: Wavelet Central frequency: {freq[j]} Hz')
    
        plt.tight_layout()
        plt.show()
     
    return wavtime, sineFamily, gausFamily,  waveletFamily 

# create the normalised fft of complex wavelet family ready for convolution procedures
def fftWavelet(epoched_EEG, timeL, freq , fwhm, fs, PrintToggle = False):
    
    '''
    Create the normalised fft of A complex wavelet family ready for convolution procedures
    Uses the function createComplexWaveletFamily to create the family of complex Morlet wavelets first
    
    INPUTS:
        epoched_EEG: numpy array, epoched_EEG for the analysis (used only for reference to shape the convolution)
        timeL : (float), total wavelet length in time (seconds)
        freq: list of ints representing the central frequency of the wavelets
        fwhm: list of floats represenmting the time (seconds) full width at half maximum
        fs = (int), sampling frequncy of data
        
    OUTPUTS:
        wvX: the normalised Complex Morlet wavelet 
        n_wavelet: the length of the wavelet (time in samples)
        n_conv: the length of the convolution vector in data point that will be used for ffts during convolution of wavelet with the data     
        
    '''
    
    if timeL is None or freq is None or fwhm is None:
        # raise ValueError("PLease provide input arguments as 1.data 2. timeL, freq, fwhm, fs Default setting are loaded.")
        timeL = 2
        freq = [2, 6, 10, 20, 38]
        fwhm = [0.45, 0.35, 0.25, 0.25, 0.1]
        
    
    if fs is None:
        raise ValueError("PLease provide input arguments fs in Hz. Unable to start analysis.")
        
    wavtime, sineFamily, gausFamily, waveletFamily = createComplexWaveletFamily(timeL, freq, fwhm, fs, PrintToggle)    
    
    # loop to obtain the fft of every eps_dataoch for every wavelet frequency
    n_wavelet = len(wavtime);
    n_data    = epoched_EEG.shape[-1]
    n_conv    = n_wavelet+n_data-1;
    
    # FFT of wavelet and its normalisation
    waveletX = scipy.fftpack.fft(waveletFamily, n_conv, axis=1) ;
    wvX = (waveletX.T/ np.abs( np.max(waveletX, axis=1 )).T).T; # normalised to max
    
    return wvX, n_wavelet, n_conv

# wvX, n_wavelet, n_conv = fftWavelet(epoched_EEG, timeL = 2, freq = [6, 45], fwhm = [0.4, 0.1], fs = 500, PrintToggle = True)
    
#% compute power spectrum via Morlet wavelet decomposition
def compute_Morlet_Spectrum(epoched_EEG, fs, max_freq = None, num_freq = None):
    
    if num_freq is None or max_freq is None:
        max_freq = 45
        num_freq = 90
    else:
        max_freq = max_freq
        num_freq = num_freq
    
    freq_res = max_freq/num_freq
    freq = np.arange(freq_res, max_freq+freq_res, freq_res)
    fwhm = np.linspace(1, 0.3, num_freq)
    
    wvX, n_wavelet, n_conv = fftWavelet(epoched_EEG, timeL = 2, freq = freq, fwhm = fwhm, fs = fs, PrintToggle = False)
    

    # helper to get rid of zero padding elements during fft multiplication
    half_wavN = (n_wavelet-1)/2; 
    n_data    = epoched_EEG.shape[-1]
    
    # initialize output time-frequency data
    coef_as = np.zeros( (len(freq), n_conv), dtype=np.complex64  ); # complex fft coeff
    mod_as = np.zeros( (len(freq), n_data), dtype=np.complex64  )
    filt_sig, rawPSD, dbPSD = [ np.zeros( (len(freq), n_data) ) for _ in range(3) ];

    # intialise final holding array for power time series
    mean_PSD = np.zeros( (epoched_EEG.shape[0], epoched_EEG.shape[1], len(freq)) );
    
    for ep in range(epoched_EEG.shape[0]):
        msg = f'  Working on epoch {ep+1}/{epoched_EEG.shape[0]}'
        sys.stdout.write('\r' + msg)
        
        temp_data =  (epoched_EEG[ ep, :, :] ).copy()
            
        # take the fft of the_data (no normalisation applied)
        dataX = scipy.fftpack.fft(temp_data , n_conv, axis = -1)
        
        # loop to get data
        for chans in range(epoched_EEG.shape[1]): #¤ for every channel
            for fi in range(len(freq)): # for every frequency of the wavelet family
            # % frequency domain convulution  followed by inverse fft of the product
            
                prod_tensor = ( wvX[fi,:] *dataX[chans,:] )
                coef_as[fi, :] =  scipy.fftpack.ifft( prod_tensor, n_conv) ; 
                mod_as[fi, :] = coef_as[fi, int(half_wavN+1):int(n_conv-half_wavN)+1 ] ; # % remove zero padding
                
                # Outputs
                filt_sig[fi, :] = np.real(mod_as[fi, :]);# filtered signal
                rawPSD[ fi, :] = (2*np.abs(filt_sig[fi, :] ))**2 # % Raw spectra in voltage units (this is tf)
                dbPSD[fi, :] = 10*np.log10(rawPSD[ fi, :]) # % Power spectra in dB units
                
                # if inf values exist replace them with nans
                rawPSD[ fi, :] = np.where(np.isfinite(rawPSD[ fi, :]), rawPSD[ fi, :], np.nan)
                rawPSD[ fi, ~np.isfinite(rawPSD[ fi, :])]= np.nan
                
                mean_PSD[ep, chans, fi] = np.nanmean(rawPSD[ fi, :])

    
    return mean_PSD, freq, fwhm 

# run function                    
# mean_PSD, freq, fwhm = compute_Morlet_Spectrum(epoched_EEG, epochs_PerSubj, fs, num_freq = 45)

#%% Compute modulation index for phase coupling

def compute_pac(phas, ampl, n_bins, PrintToggle):
    """
    Compute Phase-Amplitude Coupling (PAC) using Modulation Index (MI)
    
    Parameters:
        phas (np.ndarray): Phase time series (e.g., delta phase).
        ampl (np.ndarray): Amplitude time series (e.g., gamma envelope).
        n_bins (int): Number of phase bins (default=18).
        PrintToggle(Boolean): Controls a Figure of Results for Validation
        
    Returns:
        MI (float): Modulation Index.
        distKL (float): KL divergence.
        amplP (np.ndarray): Normalized amplitude per bin.
        bin_centers (np.ndarray): Phase bin centers (in degrees).
    """

    # Input validation
    if phas is None or ampl is None:
        raise ValueError("Phase and amplitude data must be provided.")

    if not isinstance(phas, np.ndarray) or not isinstance(ampl, np.ndarray):
        raise TypeError("Phase and amplitude inputs must be numpy arrays.")

    if phas.shape != ampl.shape:
        raise ValueError("Phase and amplitude must be the same shape.")

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("Number of bins must be a positive integer.")

    # Bin setup
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # center of bins

    # Bin the phase values
    bin_idx = np.digitize(phas, bin_edges) - 1  # subtract 1 for 0-based indexing

    # Initialize amplitude bins
    ampl_bin = np.zeros(n_bins)

    for bin in range(n_bins):
        indices = np.where(bin_idx == bin)[0]
        if indices.size > 0:
            ampl_bin[bin] = np.mean(ampl[indices])

    # Normalize to get a probability distribution
    ampl_p = ampl_bin / np.sum(ampl_bin)

    # Avoid log(0) by replacing zeros with eps
    ampl_p = np.where(ampl_p == 0, np.finfo(float).eps, ampl_p)

    # Compute entropy and KL divergence
    # shannon_entropy = -np.sum(ampl_p * np.log(ampl_p))
    # dist_entropy = np.log(n_bins) - shannon_entropy

    uniform_p = np.ones(n_bins) / n_bins
    dist_kl = np.sum(ampl_p * np.log(ampl_p / uniform_p))

    # Compute Modulation Index
    mi = dist_kl / np.log(n_bins)

    if PrintToggle == True:
        # Plotting (optional, for debugging or visualization)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        axs[0, 0].bar( np.concatenate([np.degrees(bin_centers), np.degrees(bin_centers) + 360]), np.concatenate([ampl_p, ampl_p]), color='blue', width=360/n_bins, edgecolor='black' )
        axs[0, 0].set(title = "Frequency-Amplitude Coupling", xlabel = "Phase Angle Bins (degree)", ylabel = "Normalized Gamma Amplitude" )
        
        axs[0, 1].plot(phas, label="Delta Phase")
        axs[0, 1].set( ylabel= "Delta Phase (rad)", xlabel = 'Time', title = 'Delta Phase- Gamma Amplitude')
        ax2 = axs[0, 1].twinx()
        ax2.plot(ampl, color='orange', label="Gamma Envelope", alpha=0.6)
        ax2.set(ylabel ="Gamma Amplitude")
        
        axs[1, 0].plot(np.concatenate([np.degrees(bin_centers), np.degrees(bin_centers) + 360]), np.concatenate([ampl_p, ampl_p]))
        axs[1, 0].set(title ="Normalized Gamma Amplitude", xlabel = "Phase Angle Bins (degrees)", ylabel = "Amplitude")
        
        axs[1, 1].remove()  # Remove boxed version
        
        ax_polar = fig.add_subplot(2, 2, 4, polar=True)
        ax_polar.bar(bin_centers, ampl_p, color='blue', width=2*np.pi/n_bins, edgecolor='k')
        ax_polar.set_title(f"Polar Histogram with MI of {np.round(mi, 4)}")
        ax_polar.spines['polar'].set_visible(False)
        
        ax_polar.set_rlabel_position(0)              # Move radial labels to a fixed angle
        ax_polar.tick_params(labelsize=10)           # Increase label font size
        ax_polar.grid(True)                          # Ensure gridlines are visible
        ax_polar.yaxis.label.set_color('black')      # Set radial label color
        ax_polar.set_ylim(0, np.max(ampl_p) * 1.1)

        
        plt.tight_layout()
        plt.show()

    return mi, dist_kl, ampl_p, np.degrees(bin_centers)


def pacMI(epoched_EEG, fs, timeL = None, freq= None, fwhm= None):
    '''
    Computes the Modulation Index (MI) to quantify the strength of phase-amplitude studies as descrived by Tort et al.,
    Function calls the function 
    1. fftWavele, to create convolution ready complex Morlet wavelt families at desired frequencies
    and 
    2.compute_pac  to compute MI and dist_KL according to Tort et al., 2012
    
    INPUTS:
        epoched_EEG: numpy array, epoched_EEG for the analysis are in shape epochs x chans x time points
        Calculations are conducted internally per Epoch and per channel
        
        fs = (int), sampling frequncy of data in Hz
        
        The remaining paramters define the Morlet wavelet used for the phase and amplitude calculations.
        Note that the first Frequency input acts as the Phase calculating and the Second is the Amplitude calcluating Frequency
        
        timeL : (float), total wavelet length in time (seconds)
        freq: list of ints (minimum 2) representing the central frequency of the wavelets
        fwhm: list of floats (minimum 2) represenmting the time (seconds) full width at half maximum
        
        if None is specified then the following parameters are enforced and algorith calculates Theta Phase vs Gamma Amplitude MI
        
        if  timeL is None or freq is None or fwhm is None:
            # raise ValueError("PLease provide input arguments as 1.data 2. timeL, freq, fwhm, fs Default setting are loaded.")
            timeL = 2
            freqList = [ 6, 35]
            fwhmList = [0.4, 0.3]
        
    OUTPUTS
        MI,
        dist_KL,
        testData, dictionary of important significant results for Figure reconstruction 
           storing (epochs, channels, Phase Data, Amplitude, dist_KL and MI values)
    
    '''
    
    if  timeL is None or freq is None or fwhm is None:
        # print('timeL (in seconds for wavelet) and the TWO frequncies for Phase and Ampltide coupling have not been specified, default settings are engaged')
        timeL = 2
        freqList = [ 6, 35]
        fwhmList = [0.4, 0.3]
        # print(f'length of wavelet is {timeL} secs, \n phase calculating Frequency {freqList[0]} Hz, \n amplitude calculating Frequency {freqList[1]} Hz')
    
    if fs is None:
        raise ValueError("PLease provide input arguments fs in Hz. Unable to start analysis.")
        
    wvX, n_wavelet, n_conv = fftWavelet(epoched_EEG, timeL = timeL, freq = freqList, fwhm = fwhmList, fs = fs, PrintToggle = False)
    
    num_freq = len(freqList)
    
    n_bins = 20 # bins for angles and amplitudes
    MI, dist_KL =[ np.zeros( (epoched_EEG.shape[1], epoched_EEG.shape[0] ) ) for _ in range(2) ]
    testData = {} # dictionasry of important significant results
    
    # helper to get rid of zero padding elements during fft mutliplication
    half_wavN = (n_wavelet-1)/2; 
    n_data    = epoched_EEG.shape[-1]
    n_conv    = n_wavelet+n_data-1;
    
    # initialize output time-frequency data
    coef_as = np.zeros( (num_freq, n_conv), dtype=np.complex64  ); # complex fft coeff
    mod_as  = np.zeros( (num_freq, n_data), dtype=np.complex64  )
    filt_sig, Env, phase =  [ np.zeros( (num_freq, n_data) ) for _ in range(3) ] 
    

    for chans in range(epoched_EEG.shape[1]):
        
        # msg = f'processing channel {ch_names[chans]}'
        # sys.stdout.write('\r' + msg)
        
        temp_data =  ( epoched_EEG[:, chans, :] ).copy()
                
        # take the fft of the s_data (no normalisation applied)
        dataX = scipy.fftpack.fft(temp_data , n_conv, axis = -1)
        
        # loop to get filtering of data per frequency
        for ep in range( epoched_EEG.shape[0] ): # for every epoch
        
            for fi in range(num_freq): # for every frequency of the wavelet family
            
                # print(fi)
                prod_tensor = ( wvX[fi,:] *dataX[ep, :])
                coef_as[fi, :] =  scipy.fftpack.ifft( prod_tensor, n_conv) ; 
                mod_as[fi, :] = coef_as[fi, int(half_wavN+1):int(n_conv-half_wavN)+1 ] ; # % remove zero padding
                
                # Outputs
                filt_sig[fi, :] = 2*np.real(mod_as[fi, :]);# filtered signal
                phase[ fi, :]   = np.angle(mod_as[fi, :] ) # extract phase of signal
                Env[ fi, :]     = np.abs(mod_as[fi, :] )# % Amplitude envelope in voltage units (this is tf)
                
                
            MI[chans, ep], dist_KL[chans, ep], _ , _ = compute_pac(phase[0, :], Env[1, :], n_bins, False)
        
            if MI[chans, ep] > 0.001:
                # print(f' :large MI found {np.round(MI[chans, ep], 3)}, Channel {ch_names[chans]} epoch {ep}')
                # compute_pac(phase[0, :], Env[1, :], n_bins, True)
            
                if ch_names[chans] not in testData:
                    testData[ch_names[chans]] = []
                    
                # Create a dictionary for this specific epoch
                epoch_data = {                
                    'Channel': ch_names[chans],
                    'Epoch': ep,
                    'Phase Data': phase[0, :].copy(),
                    'Ampl Data': Env[1, :].copy(),
                    'MI': MI[chans, ep].copy(), 
                    'dist_KL':dist_KL[chans, ep].copy()
                }
            
                testData[ch_names[chans]].append( epoch_data )
                        
        
    return MI, dist_KL, testData

# MI, dist_KL, testData = pacMI(epoched_EEG)

#%% Compute phase-based connectivity

def compute_phase_timeSeries(epoched_EEG, fs, timeL = None, freqList=None, fwhmList=None):
    
    '''
    
    compute_phase_timeSeries of the data for given frequencies. 
    Function calls the function 
    fftWavelet, to create convolution ready complex Morlet wavelt families at desired frequencies
    
    INPUTS:
        epoched_EEG: numpy array, epoched_EEG for the analysis are in shape epochs x chans x time points
        Calculations are conducted internally per Epoch and per channel
        
        fs = (int), sampling frequncy of data in Hz
        
        The remaining paramters define the Morlet wavelet used for the phase calculations.
        
        timeL : (float), total wavelet length in time (seconds)
        freq: list of ints (minimum 1) representing the central frequency of the wavelets
        fwhm: list of floats (minimum 1) represenmting the time (seconds) full width at half maximum
        
        if None is specified then the following parameters are enforced 
        timeL = 2
        freqList = [2, 6, 10, 20, 38]
        fwhmList = [0.45, 0.35, 0.25, 0.25, 0.1]
                

    OUTPUTS:
        tph : phase time series of data 4D numpy as epochs x chans x freq bands x sample points 
            
    '''
    
    if timeL is None or freqList is None or fwhmList is None:
        timeL = 2
        freqList = [2, 6, 10, 20, 38]
        fwhmList = [0.45, 0.35, 0.25, 0.25, 0.1]
         
    wvX, n_wavelet, n_conv = fftWavelet(epoched_EEG, timeL = timeL, freq = freqList, fwhm = fwhmList, fs = fs, PrintToggle = False)
    
    num_freq = len(freqList)

    # helper to get rid of zero padding elements during fft mutliplication
    half_wavN = (n_wavelet-1)/2; 
    n_data    = epoched_EEG.shape[-1]
    n_conv    = n_wavelet+n_data-1;
    
    # initialize output time-frequency data
    tph = np.zeros( (epoched_EEG.shape[0], epoched_EEG.shape[1], num_freq, n_data) );
    
    coef_as = np.zeros( (num_freq, n_conv), dtype=np.complex64  ); # complex fft coeff
    mod_as  = np.zeros( (num_freq, n_data), dtype=np.complex64  )
    phase   = np.zeros( (num_freq, n_data) ) 
    

    for chans in range(epoched_EEG.shape[1]):
        msg = f'processing channel {ch_names[chans]}'
        sys.stdout.write('\r' + msg)
                
        temp_data =  ( epoched_EEG[:, chans, :] ).copy()
                
        # take the fft of the s_data (no normalisation applied)
        dataX = scipy.fftpack.fft(temp_data , n_conv, axis = -1)
        
        # loop to get filtering of data per frequency
        for ep in range(epoched_EEG.shape[0]): # for every epoch
        
            for fi in range(num_freq): # for every frequency of the wavelet family
            
                # print(ep)
                prod_tensor = ( wvX[fi,:] *dataX[ep, :])
                coef_as[fi, :] =  scipy.fftpack.ifft( prod_tensor, n_conv) ; 
                mod_as[fi, :] = coef_as[fi, int(half_wavN+1):int(n_conv-half_wavN)+1 ] ; # % remove zero padding
                
                # Outputs
                phase[ fi, :]   = np.angle(mod_as[fi, :] ) # extract phase of signal
                tph[ep, chans, fi, :] = phase[ fi, :].copy() # store the phase times series of all epochs, for all chans, all freq bands
                
    return  tph  
        
# tph  = compute_phase_timeSeries(epoched_EEG)

#------------------------------------------------------------------------------
def compute_plv_pli(epoched_EEG, tph):
    
    '''
    Computes PLV and PLI connectivity metrics
    INPUTS:
        epoched:EEG: data as epochs x channels x samples
        tph: as epochs x channels x num_freq_bands x samples
        
    OUTPUTS:
        PLV AND PLI square conncectivity matrixes as 4D numpy arrays of dims:  num_freq_bands x epochs x channnels x channels
        
    '''

    num_freq = tph.shape[2]
    
    # Define the shape of the 4D tensor 
    sz = ( num_freq, epoched_EEG.shape[0], epoched_EEG.shape[1], epoched_EEG.shape[1])
    
    plv = np.zeros( (sz )) ;
    pli = np.zeros( (sz )) ;
    
    AngleDifference = np.zeros( ( epoched_EEG.shape[0], epoched_EEG.shape[1]**2, epoched_EEG.shape[-1]) ) ;
    
    # now for connectivity calculations
    for ep in range(epoched_EEG.shape[0]): # for every epoch
        msg = f'processing epoch {ep}'
        sys.stdout.write('\r' + msg)
        for j in range(epoched_EEG.shape[1]): # for the first Reference electrode till the one before the last: 
            for k in range(epoched_EEG.shape[1]): # for the number of target electrodes              
                for fi in range(num_freq):
                     
                    # take the phase angle difference
                    AngleDifference = tph[ep,j,fi,:] - tph[ep,k,fi,:];
                    
                    # finalstep calculation of PLV for the epochs for each file
                    plv[fi,ep,j,k] = np.abs(np.mean(np.exp(1j*AngleDifference) ));  
                     
                    # compute PLI:eurelise phase angle difference, take the mean of the sing of the imaginary element
                    pli[fi,ep,j,k] = np.abs(np.mean(np.sign(np.imag(np.exp(1j*AngleDifference))) ));  
                            
    return plv, pli

# plv, pli = compute_plv_pli(epoched_EEG, tph)     

#------------------------------------------------------------------------------
def compute_plv_pli_wpli(epoched_EEG, tph):
    
    '''
    Computes PLV, PLI and wPLI connectivity metrics
    
    INPUTS:
        epoched_EEG: data as epochs x channels x samples
        tph: phases as epochs x channels x num_freq_bands x samples
        
    OUTPUTS:
        PLV, PLI, wPLI square connectivity matrices as 4D numpy arrays
        dims: num_freq_bands x epochs x channels x channels
    '''

    num_freq = tph.shape[2]
    
    sz = ( num_freq, epoched_EEG.shape[0], epoched_EEG.shape[1], epoched_EEG.shape[1])
    
    plv  = np.zeros(sz)
    pli  = np.zeros(sz)
    wpli = np.zeros(sz)
    
    for ep in range(epoched_EEG.shape[0]):
        msg = f'processing epoch {ep}'
        sys.stdout.write('\r' + msg)
        
        for j in range(epoched_EEG.shape[1]):
            for k in range(epoched_EEG.shape[1]):
                for fi in range(num_freq):
                     
                    # phase difference
                    AngleDifference = tph[ep,j,fi,:] - tph[ep,k,fi,:]
                    
                    # PLV
                    plv[fi,ep,j,k] = np.abs(np.mean(np.exp(1j*AngleDifference)))
                     
                    # PLI
                    pli[fi,ep,j,k] = np.abs(np.mean(np.sign(np.imag(np.exp(1j*AngleDifference)))))
                    
                    # wPLI
                    im = np.imag(np.exp(1j * AngleDifference))
                    numerator   = np.abs(np.mean(np.abs(im) * np.sign(im)))
                    denominator = np.mean(np.abs(im))
                    wpli[fi,ep,j,k] = numerator / denominator
                            
    return plv, pli, wpli

#%% PyBispectral PAC and PPC via Bicoherence

import pybispectra; print(pybispectra.__version__)
from pybispectra import compute_fft, PAC, PPC, TDE

def compute_Bicoherence_PAC(epoched_EEG, fs, maxFreq = None):
    '''
    Computing BiCoherence antisymmetrised and Normalsied using PyBispectra library v1.3.0
    
    INPUTS:
    epoched_EEG:data as numpy arrays 3D epoch x channels x sample points
    fs: sampling frequency in Hz
    maxFreq: the maxFreq for which the pac will be computed (Hz)    
    
    OUTPUTS:
    pac_res: This Bocoherence computed for all channel paris for all frequencies as (n_chans*n_chans) x num_Freq x num_Freq

    '''
    
    if maxFreq is None:
        maxFreq = 45
    
    # intialise some parameters
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    maxFilePairCount = n_chans*n_chans
    pac_res =  np.zeros(( maxFilePairCount, maxFreq, maxFreq  ))
        
    # compute Fourier coeffs for every subject.(fft_ceifs are epochs x channels x frequencies) 
    fft_coeffs, freqs = compute_fft( data=epoched_EEG, sampling_freq=fs, n_points =fs, verbose=False )

    # initialise object PAC
    pac = PAC( data=fft_coeffs[:, :, :maxFreq], freqs=freqs[:maxFreq], sampling_freq=fs, verbose=False )

    # Note that the channel combinations to check are set in parameter indices
    pac.compute( indices= None, f1s= None, f2s= None, antisym= True, norm = True )
                             
    # collect data
    pac_res = pac.results.get_results(copy=False)
    
    print(f'\n completed Analysis for {pac.results}')
    
    return pac_res


def compute_Bicoherence_PPC(epoched_EEG, fs, maxFreq = None):
    '''
    Computing BiCoherence antisymmetrised and Normalsied using PyBispectra library v1.3.0
    
    INPUTS:
    epoched_EEG:data as numpy arrays 3D epoch x channels x sample points
    fs: sampling frequency in Hz
    maxFreq: the maxFreq for which the pac will be computed (Hz)    
    
    OUTPUTS:
    pac_res: This Bocoherence computed for all channel paris for all frequencies as (n_chans*n_chans) x num_Freq x num_Freq

    '''
    
    if maxFreq is None:
        maxFreq = 45
    
    # intialise some parameters
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    maxFilePairCount = n_chans*n_chans
    ppc_res =  np.zeros(( maxFilePairCount, maxFreq, maxFreq  ))
        
    # compute Fourier coeffs for every subject.(fft_ceifs are epochs x channels x frequencies) 
    fft_coeffs, freqs = compute_fft( data=epoched_EEG, sampling_freq=fs, n_points =fs, verbose=False )

    # initialise object PPC
    ppc = PPC( data=fft_coeffs[:, :, :maxFreq], freqs=freqs[:maxFreq], sampling_freq=fs, verbose=False )

    # Note that the channel combinations to check are set in parameter indices
    ppc.compute( indices= None, f1s= None, f2s= None)        
                             
    # collect data
    ppc_res = ppc.results.get_results(copy=False)
    
    print(f'\n completed Analysis for {ppc.results}')
    
    return ppc_res


def compute_timeDelayEstimates(epoched_EEG, fs, maxFreq = None):
    
    if maxFreq is None:
        maxFreq = 45
        
    # intialise some parameters
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    uniqPairCount = int( n_chans*(n_chans-1)/2 ) 
    tde_antisym_strength =  np.zeros(( uniqPairCount, 1 )) 

       
    # compute Fourier coeffs for every subject. 
    fft_coeffs, freqs = compute_fft(
        data=epoched_EEG, sampling_freq=fs, n_points = 2*fs, window="hamming", verbose=False )

    # initialise object TDE
    tde = TDE( data=fft_coeffs[:, :, :np.where(freqs== maxFreq)[0][0]], freqs=freqs[:np.where(freqs== maxFreq)[0][0]], sampling_freq=fs, verbose=False) 

    # compute TDE for a number of band for all electrodes!
    # tde.compute(indices= None, fmin = (1,4,8,12,25), fmax = (4,8,12,25,45), antisym= (True, False), method=1)    
    
    tde.compute(indices= None,  antisym= (True, False), method=3)    # compute TDE with Amplitude weighting
    
    # get the delay time vector
    tde_times = tde.results[0].times
    
    # get the estimated Strength
    _, tde_antisym = tde.results
 
    # collect data
    tde_antisym_res      = np.squeeze(tde_antisym.get_results(copy=False))  # return results as array (strength x times)
    tde_antisym_tau      = tde_antisym.tau
    
    # tde_pair_indices = tde_antisym.indices
        
    #get the strength of the delay    
    for uniqi in range(uniqPairCount):  
        tempA = tde_antisym_res[uniqi, :]
        timeInd = np.where( tde_times == tde_antisym_tau[uniqi])[0]
        # ind2time = tde_times[timeInd] # confirm the index is correct for the tau peak
        tde_antisym_strength[uniqi]  = float(tempA[timeInd][0])

    
    return tde_times, tde_antisym_tau, tde_antisym_strength


#%% compute Entropy and  mutual Information connectivity

def compute_entropy(epoched_EEG, nbins=25 ):

    edges = np.zeros( (epoched_EEG.shape[0], epoched_EEG.shape[1], nbins+1) )
    eps = np.finfo(np.float64).eps
    Num_el = epoched_EEG.shape[1]
    R = epoched_EEG.shape[0]
    # num_bin_edges = nbins+1
    # jointprobs = np.zeros( (num_bin_edges, num_bin_edges, Num_el, Num_el, R) ); # 5D matrix of probability matrixes per pair 
    
    countPerBin = np.zeros( (R, Num_el, nbins ) );
    edges_idx = np.zeros( (R, Num_el, epoched_EEG.shape[-1] ) ) ;
    entropy = np.zeros( (R, Num_el) ); 
    bin_idx = np.zeros( (R, Num_el, epoched_EEG.shape[-1] ) ).astype(int)
    
    for jj in range(R):
        for j in range(Num_el):
            
            # find boundaries edges for each epoch
            edges[jj, j, :] = np.linspace( np.min(epoched_EEG[jj, j,:] ), np.max( epoched_EEG[jj,j,:] ), nbins+1);
            countPerBin[ jj, j, :], bin_edges = np.histogram( epoched_EEG[jj, j, :], bins = edges[jj,j,:]); # calculate histograms on fixes edges
            countPerBin[ jj, j, :] = countPerBin[ jj, j, :] /np.sum(countPerBin[ jj, j, :] ); # normalise to probability (sum of counts of all bins)
            bin_idx[jj, j, :] = np.digitize(epoched_EEG[jj, j, :], bin_edges) - 1  # Get bin index for each element
            edges_idx[ jj, j, :] = np.searchsorted(bin_edges, epoched_EEG[jj, j, :], side='right') - 1
            entropy[ jj, j] = -np.sum( countPerBin[ jj, j, :] * np.log2(countPerBin[ jj, j, :] + eps) ); # % compute entropy
    
    return entropy, bin_idx

# entropy, bin_idx = compute_entropy(epoched_EEG, nbins=25 )

#------------------------------------------------------------------------------

def compute_mutInfo(epoched_EEG, entropy, bin_idx, nbins=25 ): 

    eps = np.finfo(np.float64).eps
    Num_el = epoched_EEG.shape[1]
    R = epoched_EEG.shape[0]   
    
    # Calculate Joint Entropy then Mutual Info 
    Joint_entropy = np.zeros( (R, Num_el, Num_el) ); # % 3D matrix
    mutInfo = np.zeros( (R, Num_el, Num_el) );
    
    for jj in range(R):
        
        msg = f'computing pairwise Mutual Information of epoch {jj} of total {R} epochs'
        sys.stdout.write('\r' + msg)
            
        for j in range(Num_el):
            for k in range(Num_el):
                temp = np.zeros( (nbins, nbins) );
                bin1 =  bin_idx[jj,j, :]  ;
                bin2 =  bin_idx[jj,k, :]  ;
                
                for i1 in range(nbins):
                    for i2 in range(nbins):
                        temp[i1, i2] = np.sum( (bin1==i1) & (bin2==i2) );
               
                temp = temp/ np.sum( temp ); # % Normalise it to probability
                Joint_entropy[jj, j, k]= -np.sum( temp.flatten()*np.log2(temp.flatten()+eps) ); # % calculate joint entropy of co distributions
                mutInfo[jj, j, k] = entropy[jj, j] + entropy[jj, k] - Joint_entropy[jj, j, k] ; # % compute MI
    
    print('JointEntropy has been computed')
    print('Mutual information per pair of electrodes has been calculated')
 
    return mutInfo            
 
# mutualInfo = compute_mutInfo(epoched_EEG, entropy, bin_idx, nbins=25  )

#%% Compute Lempel-Ziv temporal complexity score

from scipy.signal import hilbert

def compute_Lempel_Ziv_Score(epoched_EEG):
    '''
    Compute Lempel-Ziv temporal complexity score
    # Algorithm developed based on the descriptions of 
    # Aamodt et al.,2023 EEG Lempel-Ziv complexity varies with sleep stage, but does not seem to track dream experience Front in Human Neuroscience

    Method description
    # 1. extract the hilbert analytical from channel signal epoch, names H(t) 
    # 2. then take the envelope absolute absolute value of the H(t)
    # 3. calculate the mean of the absolute value of the envelope
    # 4. binarise the signal to 0 or 1 based on whetherr it is < or > than the mean signal 
    # 5. create a dictionary for the unique subsequences in the binarised epoch string
    # 6. normalise the value to the length of distinct patterns by either those from a randomly perumte (t1_LZC): according to Schwartzman et al., 2019 
    # or the max theoretical patterns (len(out)/ (len (sz_data[-1])/( np.log2(sz_data[-1])) ) accordig to Chan et al., 2025 J Aff Dis

    INPUT: data as 3d numpy array of size epochs x channels x sample points
    OUTPUT: the temporal domain tLZC score
   '''
   
    def bin_Lempel_Ziv(bin_string):
        '''
        INPUT:
        bin_string : binarised signal to 0 or 1 based on whetherr it is < or > than the mean signal 
           
        OUTPUT:
        out_dict : dictionaty of unique subsequences in the binarised epoch string
            
        '''
        
        bin_string = bin_string.astype(str)
        out_dict = {}
        w = ''
        for c in bin_string:
            wc = w + c
            if wc in out_dict:
                w = wc
            else:
                out_dict[wc] = wc
                w = c
        return out_dict
    
    # intialise some parameters
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    # tLZC_data  = np.zeros((n_epochs, n_chans, n_pnts)) # not required but good for diagnostics
    t1_LZC_score = np.zeros( (n_epochs, n_chans) ) 
        
    # temporal LSC for each channel
    for epi in range(n_epochs):        
        for chani in range(n_chans):
            
            temp = abs(hilbert( epoched_EEG[epi,chani, :]))
            median_temp = np.median(temp)
            temp[temp>median_temp]=1
            temp[temp!=1]=0
            temp = temp.astype(int)
            
            s_temp = np.random.permutation(temp.copy())
            out    = bin_Lempel_Ziv(temp)
            s_out  = bin_Lempel_Ziv(s_temp)
            
            # tLZC_data [epi, chani, :] = temp.copy() # not required but good for diagnostics
            t1_LZC_score[epi, chani] = len(out)/len(s_out)
            # t2_LZC_score[epi, chani] = len(out)/( sz_data[-1]/np.log2(sz_data[-1]) ) # Alternatively nomrlaise to time points
        
    return t1_LZC_score

#%% Frequency Sliding procedure

# Filter building in the frequency domain and its normalization 

# 1: ------------------------------------------------------------------------------
from scipy.fft import fft, ifft, fftfreq

def make_bandPass_filter( n_pnts, lowP, hiP, fs):
    
    # define the filter kernel: frequency and Gain response
    myFreq = np.array([0, lowP-0.5, lowP, hiP, hiP+0.5, 50])/[int(fs*0.5)] # nomrlaise to niquist filter range
    myGain = np.array([0,     0,    1,   1,     0,  0]) # filter gain
    
    N = n_pnts  # length of data signal
    
    # Create two sided frequency bins for the Filter kernel
    freqs = fftfreq(N, d=1/fs)  # frequencies in Hz
    
    # Normalize the freqs to Nyquist for matching
    normalized_freqs = np.abs(freqs) / (fs * 0.5)
    
    # Interpolate filter values at those normalized frequencies
    H = np.interp(normalized_freqs, myFreq, myGain)
    
    return H

# test function
# H = make_bandPass_filter(n_pnts, lowP=1, hiP=4, fs=fs) 

# 2: ------------------------------------------------------------------------------
from scipy.stats import spearmanr

def compute_FreqSliding(epoched_EEG, fs, oscBands = None, n_order= None, num_timeSteps=None ):
    '''
    Frequency sliding computations according to the method of Cohen 2014 J Neurosci
    Two versions of computation calculated: a regular and an amplitude-corrected 
    
    INPUT
    ----------
    epoched_EEG : 3D numpy array of dims epochs x channels x sample points
    fs : sampling frequency in Hz
    oscBands : dictionary: key is the freq band name and value is a tuple with bandlow, bandhigh frequencies
    n_order (int): the number of mulitscale median windonws
    num_timeSteps (int): the number of time point to evalaute the median: Must be less than the n_pnts of the epoch 
                         (1/5th of sample points is a good computational compormise)

    OUTPUT
    -------
    if_tvec : frequency slide time vector (= num_timeSteps)
    amp_freqSlide: amplitude corrected frequency sliding (Instantaneous Frequency in time): 4D array len(oscBands) x epochs x channels x num_tmeSteps
    freqSlide_corr: suqare maticex of electrode to electrode Spearman correlations: 
    freqSlide_pval: 
    

    '''
    # get the data sizes
    n_epochs, n_chans, n_pnts = epoched_EEG.shape
    
    # check same inputs
    if oscBands is None:
        # oscBands = { 'delta': (1,4), 'theta': (4,8), 'alpha': (8,12), 'beta':  (12,25), 'gamma': (25,45)  }
        oscBands = { 'theta': (4,8), 'alpha': (8,12), 'gamma': (25,45) }
    
    if n_order is None:
        n_order = 10; # 'n_order' is the number of times to compute the median and 'orders' is the size of the windows.
        # orders are cut in half to get n/2 datapoints before and n/2 datapoints after each center time point.
        orders = 1+np.linspace(10, 100, n_order).astype(int); # recommended: 10 steps between 20 and 400 ms (here specified in sample points)
    else:
        n_order = n_order
        orders = 1+np.linspace(10, 100, n_order).astype(int); 
        
    if num_timeSteps is None:
        num_timeSteps =  125 # time points to apply median (speeds analysis by reducing the number of medians computed)
        times2apply = np.linspace(1, n_pnts-2, num_timeSteps).astype(int);
    elif num_timeSteps>n_pnts:
        raise ValueError("num_timeSteps can not be leager than epoch sample points")
    else:
        num_timeSteps =  num_timeSteps 
        times2apply = np.linspace(1, n_pnts-2, num_timeSteps).astype(int);
        
        
    # define the tim evectros for oringal and Freq sliding data
    tvec = np.linspace(0, n_pnts-1, n_pnts)/fs
    if_tvec = tvec[times2apply]

    # initliase output for frequency sliding
    freqSlide = np.zeros(( len(oscBands), n_epochs, n_chans, len(times2apply) ))
    amp_freqSlide = np.zeros(( len(oscBands), n_epochs, n_chans, len(times2apply) ))

    #initliase output for correlations betweeen frequency sliding of channels
    freqSlide_corr, freqSlide_pval = [ np.full(( len(oscBands.keys()), n_epochs, n_chans, n_chans), np.nan) for _ in range(2) ]

    # Start main compuational loop
    for freqi, (fname, fRange) in enumerate( oscBands.items() ):
        # print(f' processing frequency sliding calculations for {fname} in range of {fRange[0]} and {fRange[1]} Hz \n')
        
        # create the fft of a band pass filter 
        H = make_bandPass_filter(n_pnts, lowP=fRange[0], hiP=fRange[1], fs=fs) 
        
        for ep in range(n_epochs): 
            msg = f'Frequency Sliding Computations: Working on epoch {ep+1}/{n_epochs}'
            sys.stdout.write('\r' + msg)
            
            for chani in range(n_chans):
                            
                # FFT of the signal
                X = fft(epoched_EEG[ep, chani, :])

                # Apply the filter: frequency domain multiplication followed by Inverse FFT to get filtered signal
                Y = X * H
                y = ifft(Y).real
                
                #Get the Instantneous frequency of the phase angles in Units of Hz
                temp = fs* (np.diff( np.unwrap( np.angle( hilbert(y) ) ))/(2*np.pi))
                
                # get the amplitude of the epoch at the frequency
                amp =  np.abs(hilbert(y))
               
                # At selected times ------------------------------------------------------
                # initialise output for mulitr scaled median compuatation at selected times
                phasedmed = np.zeros((n_order, len(times2apply)))
                amp_phasedmed = np.zeros((n_order, len(times2apply)))
                
                for oi in range(len(orders)): # for each order of the filter
                    halfwin = orders[oi]
                    
                    for ti in range(len(times2apply)): # for each time point (use selected time point to aply median filter not all!)
                        
                        center = times2apply[ti]
                        start = max(center - halfwin, 0)
                        stop  = min(center + halfwin+1, len(temp))
                        
                        window = temp[start:stop]
                        phasedmed[oi, ti] = np.median(window)
                        amp_phasedmed[oi, ti] = np.sum(window * amp[start:stop]) / (np.sum(amp[start:stop]) + 1e-12)
               
                smoothed_if = np.median(phasedmed, axis=0)
                amp_smoothed_if = np.median(amp_phasedmed, axis=0)

                freqSlide[freqi, ep, chani, :] = smoothed_if.copy()
                amp_freqSlide[freqi, ep, chani, :] = amp_smoothed_if.copy()
         
        # STEP TWO: calculate correlation and pval from freq sliding results   
        tempdata = amp_freqSlide[freqi, ep, :, :].copy()  # shape: (channels, len(times2apply))       
        corr, pval = spearmanr(tempdata, axis=1, nan_policy='omit')
        freqSlide_corr[freqi, ep, :, :] = corr
        freqSlide_pval[freqi, ep, :, :] = pval                 
        
    return if_tvec, amp_freqSlide, freqSlide_corr, freqSlide_pval

# 3: For Harmonic Lock of Frequency Sliding ---------------------------------------------------------------------------
from scipy.ndimage import label

def compute_masked_regions(mask, fs):
    
    # initlaise
    chan_labels = np.zeros_like(mask, dtype=int)
    chan_feat_count = np.zeros( (mask.shape[0],), dtype = int)
    
    # collect total event duration per epoch per channel
    chan_feat_dur = np.zeros( ( mask.shape[0] ) ) 
    
    for chani in range(mask.shape[0]):
        # Get the per channel contigious array and num features
        labeled_array, num_features = label(mask[chani, :], structure=np.array([1, 1, 1]))
        
        # store them
        chan_labels[chani, :] = labeled_array
        chan_feat_count[chani] = num_features               
    
        # for every contiguous region of each channel compute the duration of the event
        temp = np.zeros( (num_features) )
        
        if num_features == 0:
            continue
        else: 
            for i in range(1, num_features + 1):
                # get the the duration of each harmonic lock event (num_features) per channel: The duration is in timepoints
                temp[i-1] = len( np.where(labeled_array == i)[0] )
        
        chan_feat_dur[chani] = (np.nansum(temp)/fs) # total event duration of all locked events time in seconds!
            
    return chan_feat_count, chan_feat_dur  

# 4: For Harmonic Lock of Frequency Sliding ---------------------------------------------------------------------------

def compute_FreqSlide_HarmonicRatioLock(amp_freqSlide):
    '''
    #% Compute percentage of time in Harmonic ratio for (alpha/theta and gamma/theta)
    '''
    # dictionary ot tuple for ratios (highFreg_idx, LowFreq_idx, accelaration factor)
    ratioBands = {'alpha/theta': (1, 0, 2.0), 'gamma/theta': (2, 0, 5.0) }
    
    # get the data sizes
    _, n_epochs, n_chans, num_timeSteps = amp_freqSlide.shape
    
    # sampling rate of the freq slide is different fro original fs and deppends on its points (freqSlide is effectively 'downsampled') 
    # num_timeSteps is still 5 seconds (length of epochs) so is 125 sample points then fs = 25 Hz
    
    # redefine fs from the 'dowsampled' points in amp_freqSlide
    epoch_time = 5
    fs = num_timeSteps/epoch_time 
    
    # initialse outputs
    ratioHarm = np.zeros((n_chans, num_timeSteps)) 
    harmLock_count   = np.zeros( (len(list(ratioBands.keys())), n_epochs, n_chans ) )
    harmLock_dur     = np.zeros ( (len(list(ratioBands.keys())), n_epochs, n_chans ) )
    perTime_harmLock = np.zeros(( len(list(ratioBands.keys())), n_epochs, n_chans ) ) 
    
    
    for freqi, (fname, (hiFreg_idx, LoFreq_idx, acc_f) ) in enumerate( ratioBands.items() ):
        print(f' processing frequency sliding for {fname} with acceleration of {acc_f} \n')
        
        # get the Harmonic Lock ratio and its percentage 
        for ep in range(n_epochs): 
            
            msg = f'  Working on epoch {ep+1}/{n_epochs}'
            sys.stdout.write('\r' + msg)
            
            # Step 1: get the ratio or alpha to theata frewSlide per epoch per channel
            ratioHarm= np.round( (amp_freqSlide[hiFreg_idx, ep, :, :]/amp_freqSlide[LoFreq_idx, ep, :, :]), 1).astype(float)
            
            # find the ratio at a given acceralation
            mask = ratioHarm == acc_f
            
            # get the number of locked events and their total duration per epoch (in milliseconds)
            harmLock_count[freqi, ep, :], harmLock_dur[freqi, ep, :] = compute_masked_regions(mask, fs)
            
            # get the precentage of time in the harm Lock per epoch per channel
            perTime_harmLock[freqi, ep, :] = np.round( 100*np.nansum(mask, axis=1) / ( num_timeSteps-1), 1) # percentage in epoch where 2: 1 alpha to theta ratio persists

    return ratioHarm, harmLock_count, harmLock_dur, perTime_harmLock 

    
#%% post process the Featured Data

# import torch
       
# def feat_post_process(MI, out_psd, ap_params, peak_params, r_squared, diag_outCon_array, plv, pli, entropy, mutualInfo, epoched_EEG ):
    
#     # # 1: reshape to epochs X channels (from chans x epochs) 
#     modInd  = np.transpose(MI, (1,0)).copy()
       
#     # 2: get the statistical moments of the Welcxh poweer spectra
#     psd_mean = np.mean(out_psd, axis=-1)        
#     psd_std = np.std(out_psd, axis=-1)          
#     psd_skewness = skew(out_psd, axis=-1)      
#     psd_kurt = kurtosis(out_psd, axis=-1, fisher=False)  
#     psd_rms = np.sqrt(np.mean(out_psd**2, axis=-1))  
    
#     # 3: exptract the aperiodic parameters
#     ap_exp             = ap_params[:,:,1].copy() # exponent of aperiodic
#     ap_off             = ap_params[:,:,0].copy() # offeset  of aperiodic
    
#     # # 4: extract the periodic parameters
#     f1_peak_CF = peak_params[:, :, 0, 0 ].copy() # first periodic peak Central frequency
#     f1_peak_BP = peak_params[:, :, 0, 1 ].copy() # first periodic peak Band peak power
#     f1_peak_BW = peak_params[:, :, 0, 2 ].copy() # first periodic peak Band peak width 
    
#     # # correct for the nan missing values
#     f1_peak_CF[np.isnan(f1_peak_CF)] = 0  
#     f1_peak_BP[np.isnan(f1_peak_BP)] = 0   
#     f1_peak_BW[np.isnan(f1_peak_BW)] = 0  
    
#     # # 5: extract connectivity data required
    
#     # # add the transpose because the conncetnvity matrixes are only for the lower traingle (otherwise this will cause nan values LATER IN THE ZSCORE)
#     for ci in range(diag_outCon_array.shape[0]):
#         for epi in range(diag_outCon_array.shape[1]):
#             for fi in range(diag_outCon_array.shape[-1]):
#                 diag_outCon_array[ci,epi,:,:,fi] = diag_outCon_array[ci,epi,:,:,fi] + diag_outCon_array[ci,epi,:,:,fi].T
        
#     conn_out = np.nanmean(diag_outCon_array, axis = 2).copy() # take the mean of the first channel dimension: 
    
#     # # now pick up any band that you think is more important
#     delta_plv, theta_plv, alpha_plv, beta_plv, gamma_plv = conn_out[0, :, :, 0], conn_out[0, :, :, 1], conn_out[0, :, :, 2], conn_out[0, :, :, 3], conn_out[0, :, :, 4]
    
#     # 6 my plv and pli: 
#     my_plv, my_pli = [ np.zeros( (plv.shape[1], plv.shape[-1], plv.shape[0]) ) for _ in range(2) ]
    
#     for fi in range(plv.shape[0]):
#         for epi in range(plv.shape[1]):
#             my_plv[epi,:, fi], my_pli[epi,:, fi] = np.nanmean(plv[fi,epi,:,:], axis =-1), np.nanmean(pli[fi,epi,:,:], axis =-1) 
     
#     my_delta_plv, my_theta_plv, my_alpha_plv, my_beta_plv, my_gamma_plv = my_plv[:, :, 0], my_plv[:, :, 1], my_plv[:, :, 2], my_plv[:, :, 3], my_plv[:, :, 4]
    
#     # 7: modify Mut Info accordingly
#     mutInfo = np.nanmean(mutualInfo.copy(), axis = -1) # take the mean of electrode j, make it 2D
    
#     print('all data transformed')
    
#     # %% Combine all the data that can exist for the deep Learning
    
#     arrays = [modInd, 
#               psd_mean, psd_std, psd_skewness, psd_kurt, psd_rms,
#               ap_exp, ap_off, r_squared, f1_peak_CF, f1_peak_BP, f1_peak_BW,
#               delta_plv, theta_plv, alpha_plv, beta_plv, gamma_plv, 
#               my_delta_plv, my_theta_plv, my_alpha_plv, my_beta_plv, my_gamma_plv, 
#               entropy, mutInfo]
    
#     array_names = ["modInd", 
#                    "psd_mean", "psd_std", "psd_skewness", "psd_kurt", "psd_rms",
#                    "ap_exp", "ap_off", "r_squared", "f1_peak_CF", "f1_peak_BP", "f1_peak_BW",
#                    "delta_plv", "theta_plv", "alpha_plv", "beta_plv", "gamma_plv", 
#                    "my_delta_plv", "my_theta_plv", "my_alpha_plv", "my_beta_plv", "my_gamma_plv", 
#                    "entropy", "mutInfo"]
    
#     # test is inf or nan values are present in the features data!!
#     nanDict = {}
#     infDict = {}
    
#     for a, name in zip(arrays, array_names):
#         trial0 = np.sum(np.isnan(a))
#         trial1 = np.sum(np.isinf(a))
#         nanDict[name] = trial0 
#         infDict[name] = trial1 
        
#         if trial0 >0 or trial1>0 :
#             print('some inf or nan values are present in the feature data')
        
#         else:
#             print('feature data are clean and ready for deep learing classification')
    
#     # Expand each array along a new 3rd dimension (axis=2)
#     arrays_expanded = [a[:, :, np.newaxis] for a in arrays]
        
#     # Concatenate along the 3rd axis
#     CombData_Out = np.concatenate(arrays_expanded.copy(), axis=2)
#     CombData_Out.shape
    
#     # zscore the numpy data
#     temp_Data = zscore(CombData_Out)
    
#     # again zero the NaN values introduced by zscore of NaNs
#     temp_Data[np.isnan(temp_Data)]=0
    
#     # transfomr the data into a tensor
#     CombData = torch.from_numpy( temp_Data ).float()
    
    
#     # CHECK THAT VARIANCE IS ONE AFTER XZSCORE FOR EVERY FEATURE
#     featDict = {}
#     for feati, name in zip( range(CombData.shape[-1]),array_names ):
#         trial = torch.var(CombData[:, :, feati])
#         featDict[name] = trial
    
#     nan_score = torch.sum(torch.isnan(CombData))
    
#     # Final downsample, zscore and pack in tensor epoched_EEG
#     epoched_EEG = epoched_EEG[:,:, ::2]
#     epoched_EEG = zscore(epoched_EEG, axis = -1)
#     epoched_EEG = torch.from_numpy( epoched_EEG.copy() ).float()
        
#     print(f'EEG feature data: the zscored tensor of dims {CombData.shape} has {nan_score} nan values')    
#     print(f'featured data {CombData.shape} are a zcored tensor {type(CombData)}:\ n epoched data {epoched_EEG.shape} are a zscored tensor {type(epoched_EEG)} \n All data ready for deep learning procedures')

#     return epoched_EEG, CombData, array_names, nan_score

