import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import square, hilbert

# Parameters
Fs = 10000      # Sampling frequency
T = 1.0         # Signal duration in seconds
t = np.linspace(0, T, int(Fs * T), endpoint=False)  # Time vector
N = len(t)      # Number of samples

# Carrier signal parameters
carrier_freq = 20  # Carrier frequency in Hz
amplitude = 2.0    # Carrier amplitude

# Modulating signals
def modulating_signal_square():
    return square(2 * np.pi * 2 * t)

def modulating_signal_sawtooth():
    return 0.5 * (1 + np.sign(np.sin(2 * np.pi * 2 * t)))

# Carrier signal
def generate_carrier():
    return amplitude * np.sin(2 * np.pi * carrier_freq * t)

# Modulation types
def generate_modulated_signals():
    carrier_signal = generate_carrier()
    modulating_signal = modulating_signal_square()

    am_signal = (1 + modulating_signal) * carrier_signal
    fm_signal = np.sin(2 * np.pi * (carrier_freq + modulating_signal * 10) * t)
    pm_signal = np.sin(2 * np.pi * carrier_freq * t + modulating_signal * np.pi / 2)

    return am_signal, fm_signal, pm_signal

# FFT and filtering
def compute_spectrum(signal):
    fft_signal = fft(signal)
    frequencies = fftfreq(N, 1 / Fs)
    return frequencies, fft_signal

def filter_signal(frequencies, spectrum, low_cutoff=15, high_cutoff=25):
    mask = (np.abs(frequencies) >= low_cutoff) & (np.abs(frequencies) <= high_cutoff)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[mask] = spectrum[mask]
    return filtered_spectrum

# Signal synthesis
def synthesize_signal(filtered_spectrum):
    return np.real(ifft(filtered_spectrum))

# Hilbert envelope detection
def detect_amplitude_envelope(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope / np.max(amplitude_envelope)

# Digital signal extraction
def extract_digital_signal(amplitude_envelope, threshold=0.1):
    return (amplitude_envelope > threshold).astype(int)

# Visualization
def plot_time_domain(ax, t, signal, color, title, xlabel='Time [s]', ylabel='Amplitude'):
    """Plot a time-domain signal."""
    ax.plot(t, signal, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_frequency_domain(ax, freqs, spectrum, color, title, xlabel='Frequency [Hz]', ylabel='Magnitude', xlim=None):
    """Plot a frequency-domain signal."""
    ax.plot(freqs[:N//2], np.abs(spectrum)[:N//2], color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)

def plot_signals_and_spectra():
    am_signal, fm_signal, pm_signal = generate_modulated_signals()
    modulating_signal = modulating_signal_square()

    freq_am, spectrum_am = compute_spectrum(am_signal)
    freq_fm, spectrum_fm = compute_spectrum(fm_signal - np.mean(fm_signal))  # Detrended FM
    freq_pm, spectrum_pm = compute_spectrum(pm_signal)

    filtered_am_spectrum = filter_signal(freq_am, spectrum_am)
    synthesized_am_signal = synthesize_signal(filtered_am_spectrum)

    amplitude_envelope = detect_amplitude_envelope(am_signal)
    digital_signal = extract_digital_signal(amplitude_envelope)

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Time-domain plots
    plot_time_domain(axs[0, 0], t, am_signal, '#4287f5', 'AM Signal')
    plot_time_domain(axs[0, 1], t, fm_signal, '#d7f261', 'FM Signal')
    plot_time_domain(axs[0, 2], t, pm_signal, '#9961f2', 'PM Signal')
    plot_time_domain(axs[0, 3], t, modulating_signal, 'purple', 'Modulating Signal')

    # Frequency-domain plots
    plot_frequency_domain(axs[1, 0], freq_am, spectrum_am, '#4287f5', 'Spectrum of AM Signal', xlim=(0, 100))
    plot_frequency_domain(axs[1, 1], freq_fm, spectrum_fm, '#d7f261', 'Spectrum of FM Signal', xlim=(0, 100))
    plot_frequency_domain(axs[1, 2], freq_pm, spectrum_pm, '#9961f2', 'Spectrum of PM Signal', xlim=(0, 100))
    plot_frequency_domain(axs[2, 0], freq_am, filtered_am_spectrum, '#4287f5', 'Filtered Spectrum of AM Signal', xlim=(0, 100))

    # Synthesized signals and envelope
    plot_time_domain(axs[2, 1], t, synthesized_am_signal, 'orange', 'Synthesized AM Signal')
    plot_time_domain(axs[2, 2], t, amplitude_envelope, 'brown', 'Amplitude Envelope', ylabel='Normalized Amplitude')

    axs[2, 3].step(t, digital_signal, color='black', where='post')
    axs[2, 3].set_title('Extracted Digital Signal')
    axs[2, 3].set_xlabel('Time [s]')
    axs[2, 3].set_ylabel('Digital Value')

    plt.tight_layout()
    plt.show()

# Execute visualization
plot_signals_and_spectra()


# Execute visualization
plot_signals_and_spectra()
