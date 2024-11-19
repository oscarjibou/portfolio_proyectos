from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from typing import Sequence, Optional
from scipy.signal import get_window
import scipy.signal as signal
from scipy.io.wavfile import write, read
from scipy.io import wavfile
from pydub.effects import compress_dynamic_range


import numpy as np
import pydub


def continuous_time_plot(
    *args: ArrayLike,
    variable_name: str,
    xlabel="Time (s)",
    ylim=None,
    save_path=None,
    line_style="-",  # Default line style
    linewidth=2,  # Adjust the width of the line
    principal_lines=None,
    secondary_lines=None,
):
    """
    This function plots a time series plot with the following characteristics:
    - Line plot with points
    - Horizontal line at y=0
    - Title with the name of the variable
    - X and Y axis labels
    - Grid
    - Y axis starts at 0
    - X axis labels rotated 45 degrees

    Args:
    x (pd.Series): X axis values.
    y (pd.Series): Y axis values.
    variable_name (str, optional): Name of the variable to be displayed in the title. Defaults to str.

    xlabel (str, optional): X axis label. Defaults to "Date".
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        *args,
        linestyle=line_style,  # Specify the line style
        linewidth=linewidth,  # Specify the width of the line
    )
    plt.axhline(
        y=0, color="r", linestyle="-", linewidth=0.5
    )  # Adjusted for consistency
    plt.title(f"Continuous plot of {variable_name}")
    plt.xlabel(xlabel)
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.ylim(bottom=ylim)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if principal_lines is not None:
        for line in principal_lines:
            plt.axvline(x=line, color="r", linestyle="--")

    if secondary_lines is not None:
        for line in secondary_lines:
            plt.axvline(x=line, color="b", linestyle="--")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{variable_name} time plot.png")


def discrete_time_plot(
    *args: ArrayLike,
    variable_name: str,
    xlabel="Muestras (n)",
    ylim=None,
    save_path=None,
    markerfmt="o",  # Default marker format
    linefmt="-",  # Default line format
    markersize=1,  # Adjust the size of the point
    linewidth=0.5,  # Adjust the width of the line
):
    """
    This function plots a time series plot with the following characteristics:
    - Line plot with points
    - Horizontal line at y=0
    - Title with the name of the variable
    - X and Y axis labels
    - Grid
    - Y axis starts at 0
    - X axis labels rotated 45 degrees

    Args:
    x (pd.Series): X axis values.
    y (pd.Series): Y axis values.
    variable_name (str, optional): Name of the variable to be displayed in the title. Defaults to str.

    xlabel (str, optional): X axis label. Defaults to "Date".
    """
    plt.figure(figsize=(10, 6))
    plt.stem(
        *args,
        markerfmt=markerfmt,  # Marker format
        linefmt=linefmt,  # Line format
    )
    plt.setp(plt.gca().lines, markersize=markersize)  # Set marker size
    plt.setp(plt.gca().lines, linewidth=linewidth)  # Set line width

    # plt.axhline(
    #     y=0, color="r", linestyle="--", linewidth=linewidth
    # )  # Adjusted the baseline width for visibility
    plt.title(f"Discrete Plot of {variable_name}")
    plt.xlabel(xlabel)
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.ylim(bottom=ylim)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{variable_name} time plot.png")


def split_signal_into_frames(
    signal, sample_rate, window_size, window_overlap, window_type="hann"
):
    """
    Split the signal into frames using a sliding window approach
    Args:
    signal (np.array): The input signal
    sample_rate (int): The sample rate of the signal
    window_size (float): The size of the window in seconds
    window_overlap (float): The overlap between consecutive windows in seconds
    window_type (str): The type of window to use

    Returns:
    np.array: The signal split into frames
    """
    window_length = int(window_size * sample_rate)
    step_size = window_length - int(window_overlap * sample_rate)
    window = get_window(window_type, window_length)
    num_frames = int(np.ceil(float(np.abs(len(signal) - window_length)) / step_size))

    # Zero padding at the end to make sure that all frames have equal number of samples
    # without truncating any part of the signal
    pad_signal_length = num_frames * step_size + window_length
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z)  # Pad signal

    indices = (
        np.tile(np.arange(0, window_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * step_size, step_size), (window_length, 1)).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply the window function to each frame
    windowed_frames = frames * window

    return windowed_frames


def number_count_detector(
    signal, sample_rate, window_size, window_overlap, count=10, margin=0.1
):
    """
    Detects the presence of voice in a number count using a simple energy-based approach, but
    using a safety margin and a statistic distribution decision to find the correct threshold.

    Args:
    signal (np.array): The input signal
    sample_rate (int): The sample rate of the signal
    window_size (float): The size of the window in seconds
    window_overlap (float): The overlap between consecutive windows in seconds
    count (int): The number of numbers to detect
    margin (float): The safety margin of seconds to add to the detected voice


    Returns:
    np.array: A binary array indicating the presence of voice
    """
    # ------------------ Windowing and Energy Calculation ------------------#
    # Split the signal into frames
    windowed_frames = split_signal_into_frames(
        signal, sample_rate, window_size, window_overlap
    )
    window_samples = round(window_size * sample_rate)

    thresholds = []

    # Calculating the energy of each frame
    energy = np.sum(windowed_frames**2, axis=1)

    # ------------------ Threshold Estimation ------------------#
    # Finding the threshold that gives the correct number of numbers detected
    for thres in np.arange(0, 1 + 1 / 100, 1 / 100):
        threshold = (thres) * max(energy)
        vad = (energy > threshold).astype(int)  # Voice Activity Detection
        voice = np.repeat(vad, window_samples)

        # Now adding a safety margin to the detected voice
        safety_margin = int(margin * sample_rate)

        # Find the start and end indices of each voice segment
        voice_segments = np.where(np.diff(voice))[0] + 1

        # Add the safety margin to these indices
        for start in voice_segments[::2]:
            voice[max(0, start - safety_margin) : start] = 1
        for end in voice_segments[1::2]:
            voice[end : min(len(voice), end + safety_margin)] = 1

        # Counting the number of numbers detected
        count_numbers = 0
        for i in range(len(voice)):
            if voice[i] == 1 and voice[i - 1] == 0:
                count_numbers += 1

        if count_numbers == count:
            thresholds.append(thres)
    print(f"Thresholds: {thresholds}")

    # Choosing the final threshold:
    if thresholds == []:
        print("No threshold found")
        threshold = 0.1 * max(energy)  # Standard threshold that usually works well
    else:
        threshold = np.percentile(thresholds, 25) * max(
            energy
        )  # 25th percentile of the thresholds
    print(f"Threshold used: {threshold/(np.max(energy))}")

    vad = (energy > threshold).astype(int)
    voice = np.repeat(vad, window_samples)

    # ------------------ Safety Margin ------------------#
    # Now adding a safety margin to the detected voice
    safety_margin = int(margin * sample_rate)

    # Find the start and end indices of each voice segment
    voice_segments = np.where(np.diff(voice))[0] + 1

    # Add the safety margin to these indices
    for start in voice_segments[::2]:
        voice[max(0, start - safety_margin) : start] = 1
    for end in voice_segments[1::2]:
        voice[end : min(len(voice), end + safety_margin)] = 1

    count_numbers = 0
    for i in range(len(voice)):
        if voice[i] == 1 and voice[i - 1] == 0:
            count_numbers += 1

    # ------------------ Plotting the results ------------------#

    if max(signal) > 0.97:
        raise ValueError(
            "The signal is too loud. Please reduce the volume and try again."
        )
    else:
        print(f"Number of numbers detected: {count_numbers}")
        print(f"Maximum amplitude: {max(signal)}")
    return voice


def export_numbers(signal, sample_rate, voice, count, output_path):
    """
    Exports the detected numbers in the signal to individual wav files

    Args:
    signal (np.array): The input signal
    sample_rate (int): The sample rate of the signal
    voice (np.array): A binary array indicating the presence of voice
    count (int): The number of numbers to detect
    output_path (str): The path to save the detected numbers
    """
    # Find the start and end indices of each voice segment
    voice_segments = np.where(np.diff(voice))[0] + 1

    # Make sure we have an even number of indices
    if len(voice_segments) % 2 != 0:
        voice_segments = np.append(voice_segments, len(voice))

    # Export each voice segment to a separate wav file
    for i in range(0, min(len(voice_segments), 2 * count), 2):
        start, end = voice_segments[i], voice_segments[i + 1]
        write(f"{output_path}{i//2}.wav", sample_rate, signal[start:end])


def audio_to_numbers(
    wavfile,
    count=10,
    margin=0.02,
    window_size=0.02,
    window_overlap=0,
    output_path="output",
):
    """
    Detects the presence of voice in a number count using a simple energy-based approach

    Args:
    wavfile (str): The path to the wav file
    count (int): The number of numbers to detect
    margin (float): The safety margin of seconds to add to the detected voice
    window_size (float): The size of the window in seconds
    window_overlap (float): The overlap between consecutive windows in seconds
    output_path (str): The path to save the detected numbers
    """
    sample_rate, signal = read(wavfile)
    voice = number_count_detector(
        signal, sample_rate, window_size, window_overlap, count, margin
    )
    export_numbers(signal, sample_rate, voice, count, output_path)


def m4a_to_wav(m4a_file, wav_file):
    """
    Convert an m4a file to a wav file

    Args:
    m4a_file (str): The path to the m4a file
    wav_file (str): The path to save the wav file
    """

    sound = pydub.AudioSegment.from_file(m4a_file)
    sound.export(wav_file, format="wav")

    # Read the audio file
    freq, audio_data = wavfile.read(wav_file)
    print(f"Audio frequency: {freq}Hz")

    # Now we will make the audio Mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
        print("Audio is stereo, converting to mono")

    audio_data = audio_data / 2**15

    # Normalization (if your audio data is in integers and needs to be normalized)
    # audio_data = audio_data / np.max(np.abs(audio_data))

    # Changing the audio frequency to 16kHz if necesary
    # Target frequency
    target_freq = 16000

    if freq != 16000:
        print(f"Resampling audio from {freq}Hz to {target_freq}Hz")
        # Calculate new length of the sample
        new_length = round(len(audio_data) * target_freq / freq)

        # Resample the audio to the target frequency
        audio_data = signal.resample(audio_data, new_length)
    return audio_data, freq, target_freq
