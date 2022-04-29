"""
WARP-Q: Quality Prediction For Generative Neural Speech Codecs

This is the WARP-Q version used in the ICASSP 2021 Paper:

W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction
for generative neural speech codecs,” paper accepted for presentation at the 2021 IEEE 
International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021). Date of acceptance: 30 Jan 2021.

Run using python 3.x and include these package dependencies in your virtual environment:
    - pandas 
    - librosa
    - seaborn 
    - numpy 
    - scipy
    - pyvad
    - skimage
    - speechpy
    - soundfile 
 
Input: 
    - The main_test function calls a csv file that contains paths of audio files.  
    - The csv file cosists of four columns: 
        - Ref_Wave: reference speech
        - Test_Wave: test speech
        - MOS: subjective score (optinal, for plotting only)
        - Codec: type of speech codec for the test speech (optinal, for plotting only)
    
Output: 
    - Code will compute the WARP-Q quality scores between Ref_Wave and Test_Wave,
    and will store the obrained results in a new column in the same csv file.  
    
        
Releases: 
    
Warning: While this code has been tested and commented giving invalid input 
files may cause unexpected results and will not be caught by robust exception
handling or validation checking. It will just fail or give you the wrong answer.

    
(c) Dr Wissam Jassim
    University College Dublin
    wissam.a.jassim@gmail.com
    wissam.jassim@ucd.ie
    November 28, 2020

"""

# Load libraries
from msilib.schema import File
import pandas as pd
import librosa, librosa.core, librosa.display
import seaborn as sns
import numpy as np
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pyvad import vad, trim, split
from skimage.util.shape import view_as_windows
import speechpy
import soundfile as sf
import tensorflow as tf
import tensorflow_lattice as tfl
import pandas as pd
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow import keras
from absl import app
from absl import flags
from tensorflow import feature_column as fc

### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 48, 'Number of training epoch.')

###############################################################################
############################# #  WARP - Q #####################################
###############################################################################
def compute_WAPRQ(ref_path,test_path,sr=16000,n_mfcc=12,fmax=5000,patch_size=0.4,sigma=np.array([[1,1],[3,2],[1,3]])):

    # Inputs:
    # refPath: path of reference speech
    # disPath: path pf degraded speech
    # sr: sampling frequency, Hz
    # n_mfcc: number of MFCCs
    # fmax: cutoff frequency
    # patch_size: size of each patch in s
    # sigma: step size conditon for DTW

    # Output:
    # WARP-Q quality score between refPath and disPath


    ####################### Load speech files #################################
    # Load Ref Speech
    if ref_path[-4:] == '.wav':
        speech_Ref, sr_Ref = librosa.load(ref_path,sr=sr) 
    else:
        if ref_path[-4:] == '.SRC': #For ITUT database if applicable
            speech_Ref, sr_Ref  = sf.read(ref_path, format='RAW', channels=1, samplerate=16000,
                           subtype='PCM_16', endian='LITTLE')
            if sr_Ref != sr:
                speech_Ref = librosa.resample(speech_Ref, sr_Ref, sr)
                sr_Ref = sr
        
    # Load Coded Speech
    if test_path[-4:] == '.wav':
        speech_Coded, sr_Coded = librosa.load(test_path,sr=sr)
    else: 
        if test_path[-4:] == '.OUT': #For ITUT database if applicable
            speech_Coded, sr_Coded  = sf.read(test_path, format='RAW', channels=1, samplerate=16000,
                           subtype='PCM_16', endian='LITTLE')
            if sr_Coded != sr:
                speech_Coded = librosa.resample(speech_Coded, sr_Coded, sr)
                sr_Coded = sr
    
    if sr_Ref != sr_Coded:
        raise ValueError("Reference and degraded signals should have same sampling rate!")
    
    # Make sure amplitudes are in the range of [-1, 1] otherwise clipping to -1 to 1 
    # after resampling (if applicable). We experienced this issue for TCD-VOIP database only
    speech_Ref[speech_Ref>1]=1.0
    speech_Ref[speech_Ref<-1]=-1.0
    
    speech_Coded[speech_Coded>1]=1.0
    speech_Coded[speech_Coded<-1]=-1.0
    
    ###########################################################################
   
    win_length = int(0.032*sr) #32 ms frame
    hop_length = int(0.004*sr) #4 ms overlap
    #hop_length = int(0.016*sr)
    
    n_fft = 2*win_length
    lifter = 3 
    
    # DTW Parameters
    Metric = 'euclidean'

    # VAD Parameters
    hop_size_vad = 30
    sr_vad = sr
    aggresive = 0
    
    # VAD for Ref speech
    vact1 = vad(speech_Ref, sr, fs_vad = sr_vad, hop_length = hop_size_vad, vad_mode=aggresive)
    speech_Ref_vad = speech_Ref[vact1==1]
    
    # VAD for Coded speech
    vact2 = vad(speech_Coded, sr, fs_vad = sr_vad, hop_length = hop_size_vad, vad_mode=aggresive)
    speech_Coded_vad = speech_Coded[vact2==1]
   
    # Compute MFCC features for the two signals
    
    mfcc_Ref = librosa.feature.mfcc(speech_Ref_vad,sr=sr,n_mfcc=n_mfcc,fmax=fmax,
                                    n_fft=n_fft,win_length=win_length,hop_length=hop_length,lifter=lifter)
    mfcc_Coded = librosa.feature.mfcc(speech_Coded_vad,sr=sr,n_mfcc=n_mfcc,fmax=fmax,
                                    n_fft=n_fft,win_length=win_length,hop_length=hop_length,lifter=lifter)
    
    # Feature Normalisation using CMVNW method 
    mfcc_Ref = speechpy.processing.cmvnw(mfcc_Ref.T,win_size=201,variance_normalization=True).T
    mfcc_Coded = speechpy.processing.cmvnw(mfcc_Coded.T,win_size=201,variance_normalization=True).T
    
    # Divid MFCC features of Coded speech into patches
    cols = int(patch_size/(hop_length/sr))
    window_shape = (np.size(mfcc_Ref,0), cols)
    step  = int(cols/2)
    
    mfcc_Coded_patch = view_as_windows(mfcc_Coded, window_shape, step)

    Acc =[]
    band_rad = 0.25
    weights_mul=np.array([1, 1, 1])
    
    # Compute alignment cose between each patch and Ref MFCC
    for i in range(mfcc_Coded_patch.shape[1]):    
        
        patch = mfcc_Coded_patch[0][i]
        
        D, P = librosa.sequence.dtw(X=patch, Y=mfcc_Ref, metric=Metric, 
                                    step_sizes_sigma=sigma, weights_mul=weights_mul, 
                                    band_rad=band_rad, subseq=True, backtrack=True)
  
        P_librosa = P[::-1, :]
        b_ast = P_librosa[-1, 1]
        
        Acc.append(D[-1, b_ast] / D.shape[0])
        
    # Final score
    #return np.median(Acc)              # Old return to ignore DLN
    return Acc                         # Return for Feature Data

###############################################################################
################## #  Deep Lattice Network ####################################
###############################################################################
def deep_lattice_network(Acc):
    print()

###############################################################################
############### #  Main Demo Test Function ####################################
###############################################################################
def main_test():

    # Load path of speech files stored in a csv file 
    # The csv file cosists of four columns: 
    # Ref_Wave	Test_Wave 	MOS 	Codec  
    
    print(flags)

    All_Data = pd.read_csv("audio_paths.csv",index_col=None)
    
    WARP_Q = [] # List to add WARP-Q scores
    # Run WARP-Q for each row
    for index, row in All_Data.iterrows():
        score = compute_WAPRQ(ref_path=row['Ref_Wave'],test_path=row['Test_Wave'])
        WARP_Q.append(score)
        print(row['Test_Wave'])
    
    
    # Add computed score to the dataframe
    All_Data['WARP-Q'] = WARP_Q
    warp_q_df = pd.DataFrame(WARP_Q)
    # Save WARP-Q to csv
    warp_q_df.to_csv('Results.csv', index = False)
    print(warp_q_df)
    #deep_lattice_network()

if __name__ == '__main__':
    main_test()
