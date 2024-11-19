import numpy as np
import librosa
import scipy 
from scipy.signal import find_peaks, spectrogram
from scipy.ndimage import median_filter
from scipy.ndimage import binary_opening, binary_closing

def debuffer(xwin,  frame_length, overlap_length):
    L = xwin.shape[0]
    hop_length = frame_length - overlap_length
    start_frame = (np.arange(L)*hop_length).astype(np.int64)
    end_frame = (start_frame+frame_length).astype(np.int64)

    
    x = np.zeros(end_frame[-1])
    
    for k, f in enumerate(xwin):
        if f > 0:
            x[start_frame[k]:end_frame[k]] = 1
    return x
            


def nextpow2(n):
    return int(np.ceil(np.log2(n)))


def validate_required_inputs(x, fs):
    assert x.ndim == 1 and len(x) > 0 and np.isrealobj(x) and np.isfinite(x).all(), "audioIn must be a non-empty column vector of real finite values"
    assert np.isscalar(fs) and fs > 0 and np.isreal(fs) and np.isfinite(fs), "fs must be a positive scalar finite value"
    assert fs >= 20, "fs must be at least 20 Hz"

def get_thresholds_from_feature(feature, bins):
    hist_bins = max(10, round(len(feature) / bins))
    m_feature = np.mean(feature)
    n_feature, edges_feature = np.histogram(feature, hist_bins)
    
    if edges_feature[0] == 0:
        n_feature = n_feature[1:]
        edges_feature = edges_feature[1:]
    
    peaks_idx, _ = find_peaks(n_feature)
    
    if len(peaks_idx) == 0:
        M1 = m_feature / 2
        M2 = min(feature)
    elif len(peaks_idx) == 1:
        M1 = (0.5 * (np.concatenate(([0], edges_feature)) + np.concatenate((edges_feature, [0])))[peaks_idx[0] + 1])
        M2 = min(feature)
    else:
        aa = (0.5 * (np.concatenate(([0], edges_feature)) + np.concatenate((edges_feature, [0])))[peaks_idx + 1])
        M1, M2 = aa[1], aa[0]

    return M1, M2


def detect_speech(audio_in, fs, win_length=None, hop_length=None, merge_distance=None, thresholds=None):
    """ Parameters: 
        audio_in: audio samples
        fs: sample rate
        win_length: size of window for spectrogam in samples, if None adjusts the win_length to 30ms
        hop_length: window advance, if none -> win_length //2
        merge_distance: used to merge close segments (unit is samples)
    """
    
    # Internal Parameters
    W = 5                               # Weight for finding local maxima
    bins = 15                           # Number of bins for histograms
    spectral_spread_threshold = 0.05    # Threshold used for not counting spectral spread when under this energy
    lower_spread_threshold_factor = 0.8 # Factor to lower the spectral spread threshold
    smoothing_filter_length = 5         # After getting spectral spread and energy data, these features are smoothed by filters of this length

    if win_length is None:
        win_length = round(0.03 * fs)
    n_fft = 512
    
    if hop_length is None:
        hop_length = win_length // 2 # 50%
    overlap_length = win_length - hop_length
    
    window = scipy.signal.windows.hamming(win_length)
    # Normalize audio signal
    sig_max = np.max(np.abs(audio_in))
    if sig_max > 0:
        audio_in /= sig_max

    # Determine frame parameters
    frame_length = win_length

    frames = librosa.util.frame(audio_in, frame_length=frame_length, hop_length=hop_length)

    # Determine short term energy
    energy = np.sum((window ** 2).reshape(-1,1) * frames ** 2, axis=0)
   
    # Filter the short term energy twice
    f_energy = median_filter(median_filter(energy, size=smoothing_filter_length), size=smoothing_filter_length)

    # Compute spectrogram
    #f, t, Sxx = spectrogram(audio_in, fs, window="hamming", nperseg=frame_length, noverlap=overlap_length, nfft = 512,  mode='magnitude')

    Sxx = librosa.stft(audio_in, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming")
    Sxx = np.abs(Sxx)
    f = librosa.core.fft_frequencies(n_fft=n_fft, sr = fs)
  

    # Compute spectral spread
    spec_spread = np.sum(f[:,None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)

    # Normalize the feature
    spec_spread /= (fs / 2)


    if len(energy) < len(spec_spread):
        spec_spread = spec_spread[:len(energy)]

    if len(spec_spread) < len(energy):
        energy = energy[:len(spec_spread)]
    
    # Set spectral spread value to 0 for frames with low energy
    spec_spread[energy < spectral_spread_threshold] = 0

    # Filter spectral spread twice
    f_spec_spread = median_filter(median_filter(spec_spread, size=smoothing_filter_length), size=smoothing_filter_length)

    # Determine thresholds
    if thresholds is None:
        ss_m1, ss_m2 = get_thresholds_from_feature(f_spec_spread, bins)
        e1_m1, e1_m2 = get_thresholds_from_feature(f_energy, bins) 
        ww = 1 / (W + 1)
        sspread_thresh = ww * (W * ss_m2 + ss_m1) * lower_spread_threshold_factor
        energy_thresh = ww * (W * e1_m2 + e1_m1)
    else:
        energy_thresh, sspread_thresh = thresholds

    

    
    # Apply threshold criterion
    #print(sspread_thresh)
    #print(energy_thresh)
    #import matplotlib.pyplot as plt
    #plt.plot(f_spec_spread*10)
    #plt.plot(f_energy)
    speech_mask = (f_spec_spread > sspread_thresh/30) & (f_energy > energy_thresh/10)
    #plt.plot(speech_mask)
    
    if merge_distance is None:
        merge_distance = 5  # in number of frames

    #Merge frames if they are close as described by merge distance

    structuring_element = np.ones(merge_distance)

    # Perform binary closing and opening
    result = binary_closing(speech_mask, structure=structuring_element)
    result = binary_opening(result, structure=structuring_element)
    # De-buffer for frame overlap
    result = debuffer(result, frame_length, hop_length)

    return result


