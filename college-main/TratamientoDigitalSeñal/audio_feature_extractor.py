"""
This file contains functions to extract specific audio features from an audio signal.
"""

import librosa
import numpy as np

from matplotlib import pyplot as plt


def _spectrogram(audio):
    # Calculate _spectrogram
    spec = np.abs(librosa.stft(audio))
    return spec


def mfcc(audio, sample_rate, n_mfcc: float, hop_length: int, plot=True):
    """
    Compute the Mel-frequency cepstral coefficients (MFCCs) of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the MFCCs.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the MFCCs, by default True.

    Returns
    -------
    np.ndarray
        The MFCCs of the audio signal.
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length
    )
    d_mfccs = librosa.feature.delta(mfccs, order=1)

    if plot:
        # Plot MFCCs
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(mfccs, x_axis="time", sr=sample_rate, y_axis="mel")
        plt.colorbar()
        plt.title("MFCCs")

        # Plot delta MFCCs
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(d_mfccs, x_axis="time", sr=sample_rate, y_axis="mel")
        plt.colorbar()
        plt.title("Delta MFCCs")
        plt.show()

    return mfccs, d_mfccs


def spectral_flux(audio, sample_rate, plot=True):
    """
    Compute the spectral flux of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral flux.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral flux, by default True.

    Returns
    -------
    np.ndarray
        The spectral flux of the audio signal.
    """
    # Calculate _spectrogram
    spec = _spectrogram(audio)

    # Compute spectral flux
    flux = np.sum(np.square(np.diff(spec, axis=1)), axis=0)
    if plot:
        # Plot spectral flux
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram and Spectral flux")

        # Create a twin axis
        ax2 = plt.twinx()

        # Plot spectral flux on the twin axis
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, len(flux)),
            20 * np.log10(flux),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral flux (dB)", color="r")
        ax2.tick_params("y", colors="r")

    return flux


def spectral_centroid(audio, sample_rate, plot=True):
    """
    Compute the spectral centroid of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral centroid.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral centroid, by default True.

    Returns
    -------
    np.ndarray
        The spectral centroid of the audio signal.
    """
    # Calculate _spectrogram
    spec = _spectrogram(audio)

    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(S=spec)
    centroid_data = centroid[0]
    if plot:
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram and Spectral Centroid")

        # Create a twin axis
        ax2 = plt.twinx()

        # Plot spectral centroid on the twin axis
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, len(centroid_data)),
            20 * np.log10(centroid_data),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral Centroid (dB)", color="r")
        ax2.tick_params("y", colors="r")

        plt.show()
    return centroid_data


def spectral_kurtosis(audio, sample_rate, plot=True):
    """
    Compute the spectral kurtosis of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral kurtosis.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral kurtosis, by default True.

    Returns
    -------
    np.ndarray
        The spectral kurtosis of the audio signal.
    """
    # Calculate _spectrogram
    spec = _spectrogram(audio)

    centroid_data = spectral_centroid(audio, sample_rate, plot=False)

    # Compute spectral kurtosis
    kurtosis = np.sum(
        ((np.arange(0, spec.shape[0])[:, np.newaxis] - centroid_data) ** 4) * spec,
        axis=0,
    ) / (np.sum(spec, axis=0) * np.var(spec, axis=0) ** 2)
    # Normalize kurtosis to the range of the _spectrogram
    kurtosis_norm = (kurtosis - np.min(kurtosis)) / (
        np.max(kurtosis) - np.min(kurtosis)
    )
    if plot:
        # Plot _spectrogram
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram and Spectral Kurtosis")

        # Create a twin axis
        ax2 = plt.twinx()
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, len(kurtosis_norm)),
            20 * np.log10(kurtosis),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral Kurtosis (dB)", color="r")
        ax2.tick_params("y", colors="r")

        plt.show()
    return kurtosis


def spectral_rolloff(audio, sample_rate, plot=True):
    """
    Compute the spectral rolloff of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral rolloff.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral rolloff, by default True.

    Returns
    -------
    np.ndarray
        The spectral rolloff of the audio signal.
    """
    # Calculate _spectrogram
    spec = _spectrogram(audio)

    rolloff = librosa.feature.spectral_rolloff(
        S=spec, sr=sample_rate, roll_percent=0.85
    )
    if plot:
        # Plot spectral rolloff point
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectral Rolloff Point")
        # Create a twin axis

        ax2 = plt.twinx()
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, rolloff.shape[1]),
            20 * np.log10(rolloff[0]),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral Rolloff Point (dB)", color="r")
        ax2.tick_params("y", colors="r")

        plt.show()
    return rolloff[0]


def spectral_skewness(audio, sample_rate, plot=True):
    """
    Compute the spectral skewness of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral skewness.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral skewness, by default True.

    Returns
    -------
    np.ndarray
        The spectral skewness of the audio signal.
    """
    # Calculate _spectrogram
    spec = _spectrogram(audio)

    centroid_data = spectral_centroid(audio, sample_rate, plot=False)
    skewness = np.sum(((np.arange(0, spec.shape[1]) - centroid_data) ** 3) * spec) / (
        np.sum(spec, axis=0) * np.var(spec, axis=0) ** (3 / 2)
    )

    normalized_skewness = (skewness - np.min(skewness)) / (
        np.max(skewness) - np.min(skewness)
    )
    if plot:
        # Plot spectral skewness
        plt.figure(figsize=(20, 10))

        librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectral Skewness")

        # Create a twin axis
        ax2 = plt.twinx()
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, skewness.shape[0]),
            20 * np.log10(normalized_skewness),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral Skewness Point (dB)", color="r")
        ax2.tick_params("y", colors="r")

        plt.show()
    return skewness


def spectral_spread(audio, sample_rate, plot=True):
    """
    Compute the spectral spread of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the spectral spread.
    sample_rate : int
        The sample rate of the audio signal.
    plot : bool, optional
        Whether to plot the spectrogram and spectral spread, by default True.

    Returns
    -------
    np.ndarray
        The spectral spread of the audio signal.
    """
    soec = _spectrogram(audio)
    spread = librosa.feature.spectral_bandwidth(S=soec)

    if plot:
        # Plot spectral spread
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(
            librosa.amplitude_to_db(soec, ref=np.max),
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectral Spread")

        # Create a twin axis
        ax2 = plt.twinx()
        ax2.plot(
            np.linspace(0, len(audio) / sample_rate, spread.shape[1]),
            20 * np.log10(spread[0]),
            color="r",
            linewidth=2,
        )
        ax2.set_ylabel("Spectral Spread (dB)", color="r")
        ax2.tick_params("y", colors="r")

        plt.show()
    return spread[0]


def calculate_audio_features(audio, sample_rate, hop_length: int):
    """
    This function calculates all the audio features to later analyze by a neural network.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal from which to compute the audio features. Must be in mono and normalized to matlab format.
    sample_rate : int
        The sample rate of the audio signal.

    Returns
    -------
    np.ndarray
        The audio features of the audio signal.
    """

    # Calculate the features
    mfccs, delta_mfccs = mfcc(
        audio, sample_rate, n_mfcc=13, hop_length=hop_length, plot=False
    )
    centroid = spectral_centroid(audio, sample_rate, plot=False)
    spread = spectral_spread(audio, sample_rate, plot=False)
    rollof = spectral_rolloff(audio, sample_rate, plot=False)
    flux = spectral_flux(audio, sample_rate, plot=False)
    skewness = spectral_skewness(audio, sample_rate, plot=False)
    kurtosis = spectral_kurtosis(audio, sample_rate, plot=False)

    # Concatenate all features into a single array
    features = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.var(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1),
            np.var(delta_mfccs, axis=1),
            np.mean(flux),
            np.var(flux),
            np.mean(centroid),
            np.mean(spread),
            np.mean(skewness),
            np.mean(kurtosis),
            np.mean(rollof),
        ]
    )

    return features
