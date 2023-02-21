import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

FUNDAMENTAL_FREQUENCY_USE_LIBROSA = False
SHOW_SPECTOGRAM_WITH_FUNDAMENTAL_FREQUENCY = False and FUNDAMENTAL_FREQUENCY_USE_LIBROSA

class file_data():
    def __init__(self, path, fundamental_frequencies, waveform, sampling_rate):
        self.path = path
        filtered_fundamental_frequencies = [x for x in fundamental_frequencies if not np.isnan(x)]
        self.fundamental_frequencies = sorted(filtered_fundamental_frequencies)
        
        self.meanfun = np.mean(self.fundamental_frequencies)
        self.q25 = np.percentile(self.fundamental_frequencies, 25)
        self.iqr = np.percentile(self.fundamental_frequencies, 75) - self.q25
        self.dfrange = self.fundamental_frequencies[-1] - self.fundamental_frequencies[0]
        self.stdev = np.std(self.fundamental_frequencies)
        self.skew = skew(self.fundamental_frequencies)
        self.mfcc = librosa.feature.mfcc(y = waveform, sr = sampling_rate, htk = False, n_mfcc = 12)
        self.mfcc = np.array(list(map(lambda x: np.mean(x), self.mfcc)))
        
    def __str__(self):
        return f"{self.path}:\n\tmeanfun: {self.meanfun}\n\tq25: {self.q25}\n\tiqr: {self.iqr}\n\tdfrange: {self.dfrange}\n\tstdev: {self.stdev}\n\tskew: {self.skew}\n\tmfcc: {self.mfcc}"


def load_audio(path):
    waveform, sampling_rate = librosa.load(path)
    
    fundamental_frequencies = []
    if not FUNDAMENTAL_FREQUENCY_USE_LIBROSA:
        #We want to detect down to 50Hz, so we need 40 ms
        divide_into_n_parts = (waveform.shape[0]/sampling_rate)*25 # Divide into 40ms intervals
        if(int(divide_into_n_parts) != divide_into_n_parts):
            divide_into_n_parts = int(divide_into_n_parts) + 1
        
        splitted = np.array_split(waveform, divide_into_n_parts)
        
        for i in range(len(splitted)):
            section = splitted[i]
            if len(section) % 2 == 1:
                section = np.append(section, 0) # should not happen, but if it does, it will make calculations much easier without losing much
            middle_right_index = len(section) // 2 + 1
            
            mutual_coefficients = []
            for j in range(2, len(section) // 2):
                left = section[middle_right_index - j : middle_right_index]
                right = section[middle_right_index : middle_right_index + j]
                normalization_coeff = (np.linalg.norm(left) * np.linalg.norm(right))
                normalized_mutual_coefficient = np.dot(left, right) / normalization_coeff if normalization_coeff != 0 else 0
                mutual_coefficients.append(normalized_mutual_coefficient)
            
            max_value = max(mutual_coefficients)
            max_index = mutual_coefficients.index(max_value) + 2
            f0 = sampling_rate/max_index
            if librosa.note_to_hz('C2') < f0 and f0 < librosa.note_to_hz('C7'): # To match librosa's range
                fundamental_frequencies.append(round(f0))
            else:
                fundamental_frequencies.append(np.nan)
    else:
        fundamental_frequencies, _, _ = librosa.pyin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sampling_rate)
        print(fundamental_frequencies.shape)
        
    if SHOW_SPECTOGRAM_WITH_FUNDAMENTAL_FREQUENCY:
        S = np.abs(librosa.stft(waveform))
        times = librosa.times_like(fundamental_frequencies)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Power spectrogram with fundamental frequency')
        ax.plot(times, fundamental_frequencies, label='f0', color='cyan', linewidth=3)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    return file_data(path, fundamental_frequencies, waveform, sampling_rate)


print(load_audio("Dataset/Female/0.wav"))
print(load_audio("Dataset/Female/1.wav"))
print(load_audio("Dataset/Male/0.wav"))
print(load_audio("Dataset/Male/1.wav"))