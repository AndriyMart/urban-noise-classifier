import os
from operator import itemgetter

import librosa
import numpy
import pandas
import pandas as pd
import soundfile

features_csv_path = "./features/features.csv"
config_path = 'audio/UrbanSound8K.csv'
audio_folders = [
    "./audio/fold1/",
    "./audio/fold2/",
    "./audio/fold3/",
    "./audio/fold4/",
    "./audio/fold5/",
    "./audio/fold6/",
    "./audio/fold7/",
    "./audio/fold8/",
    "./audio/fold9/",
    "./audio/fold10/"
]


class FeatureExtractor:

    def __init__(self):
        pass

    def extract_features_from_csv(self):
        frame = pd.read_csv(features_csv_path, converters={'feature': pd.eval})
        return frame['label'].values, frame['feature'].values

    def extract_features_from_dataset_and_save_in_csv(self):
        samples, labels = self._get_samples()
        features = []

        for index, label in enumerate(labels):
            features.append({
                'label': label,
                'feature': samples[index]
            })

        frame = pd.DataFrame.from_records(features)
        frame.to_csv(features_csv_path)

    def _get_samples(self):
        # Returns audios in representation of mean mfccs and labels for the audios
        df = pandas.read_csv(config_path)

        return self._get_audios(), df['class'].values

    def _parse_audio(self, audio_array):
        # returns a copy of the array collapsed into one dimension,
        # 'F' parameter that represents the sorting order (Fortran style)
        # this function extracts frames from (frames x channels) NumPy array
        return audio_array.flatten('F')[:audio_array.shape[0]]

    def _get_audios(self):
        # Extracts audio files, does preprocessing and returns mean mfccs for each file.
        train_files_meta = []

        for train_path in audio_folders:
            for file_name in os.listdir(train_path):
                train_files_meta.append({
                    'name': file_name,
                    'path': train_path
                })

        train_files_meta = sorted(train_files_meta, key=itemgetter('name'))

        return list(map(self.process_train_file, train_files_meta))

    def process_train_file(self, train_file_meta):
        audio_data, sample_rate = soundfile.read(train_file_meta['path'] + train_file_meta['name'], always_2d=True)
        # audio_data stores a two-dimensional (frames x channels) NumPy array
        # sample_rate stores sample rate of the audio file

        frames = self._parse_audio(audio_data)

        return self._mean_mfccs(frames)

    def _mean_mfccs(self, frames):
        # Calculates a mean mfccs for given audio. MFCCS is a set of features represented by two-dimensional numerical
        # array. Mean mfccs is a flattened (one-dimensional) array of average values for each sub-array in mfcc.
        return [numpy.mean(feature) for feature in librosa.feature.mfcc(y=frames)]
