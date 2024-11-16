import numpy as np
import matplotlib.pyplot as plt

amplitude = 2
time = np.linspace(0, 1, 1000)
initial_phase = 0
frequencies = [1, 2, 4, 8]


def generate_sine_signals(frequencies, amplitude, time, initial_phase):
    signals = []
    for freq in frequencies:
        omega = 2 * np.pi * freq  # Angular frequency
        phase = omega * time + initial_phase
        signal = amplitude * np.cos(phase)
        signals.append(signal)
    return signals


def generate_square_signals(time, frequencies, amplitude):
    signals = []
    for freq in frequencies:
        signal = amplitude * np.sign(np.sin(2 * np.pi * freq * time))
        signals.append(signal)
    return signals


def calculate_spectrum(signals):
    return [np.abs(np.fft.fft(signal)) for signal in signals]


# Signal generation
sine_signals = generate_sine_signals(frequencies, amplitude, time, initial_phase)
square_signals = generate_square_signals(time, frequencies, amplitude)
frequencies_for_fft = np.fft.fftfreq(len(time), d=0.001)


def configure_subplot(ax, x_data, y_data, title, xlabel, ylabel, color, xlim=None):
    ax.plot(x_data, y_data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)


def plot_signals_and_spectra():
    """Displays plots of signals and their spectra."""
    fig, axs = plt.subplots(4, 4, figsize=(15, 10))

    # For each signal (harmonic and digital)
    for i, freq in enumerate(frequencies):
        # Harmonic signal
        configure_subplot(axs[i, 0], time, sine_signals[i],
                          f'Harmonic Signal {freq} Hz', 'Time [s]', 'Amplitude', 'red')

        # Digital signal
        configure_subplot(axs[i, 1], time, square_signals[i],
                          f'Digital Signal {freq} Hz', 'Time [s]', 'Amplitude', 'b')

        # Spectrum of harmonic signal
        configure_subplot(axs[i, 2], frequencies_for_fft, calculate_spectrum(sine_signals)[i],
                          f'Harmonic Spectrum {freq} Hz', 'Frequency [Hz]', 'Amplitude', 'red', xlim=(0, 50))

        # Spectrum of digital signal
        configure_subplot(axs[i, 3], frequencies_for_fft, calculate_spectrum(square_signals)[i],
                          f'Digital Spectrum {freq} Hz', 'Frequency [Hz]', 'Amplitude', 'b', xlim=(0, 50))

    plt.tight_layout()
    plt.show()


plot_signals_and_spectra()
