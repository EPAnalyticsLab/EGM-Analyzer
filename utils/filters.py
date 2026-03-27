import scipy.signal as signal


def bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(data, cutoff, fs, Q=30.0):
    nyquist = 0.5 * fs
    f0 = cutoff
    w0 = f0 / nyquist
    b, a = signal.iirnotch(w0, Q)
    y = signal.filtfilt(b, a, data)
    return y
