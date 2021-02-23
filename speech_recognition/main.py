import os
import json
import tensorflow as tf
import numpy as np
import tqdm as tqdm
import pickle
import librosa
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils, to_categorical
# from sklearn.preprocessing import LabelEncoder

from model import build_model, train, plot_history


def create_data(root_audio_dir: str, pkl_save_path: str, n_mfcc: int = 13,
                hop_length: int = 512, n_fft: int = 2048) -> None:
    """
    This method will take in raw audio samples, then store then in a json object where the keys are the labels, and the
    values are the calculated MFCCs per sample. The default values that are passed were chosen arbitrarily and can be
    changed at will. If the sample wav file is less than 3 seconds long, it will be zero-padded up to 3 seconds.

    :param root_audio_dir: String
        Path to directory storing all folders containing music samples.
    :param pkl_save_path: String
        Path for save file that will be created.
    :param n_mfcc: int
        Number of features that will be extracted, default bring 13 features per frame.
    :param hop_length: int
        The number of frames to shift along the time axis.
    :param n_fft: int
        Length of window to consider when calculating MFCCs
    :return: None
    """
    ret_data = {}
    for dir in tqdm.tqdm(os.listdir(root_audio_dir)):
        if dir not in ret_data:
            ret_data[dir] = []
        for filename in os.listdir(root_audio_dir + os.path.sep + dir):
            if filename.endswith(".wav"):
                # load the audio file
                signal, sr = librosa.load(root_audio_dir + os.path.sep + dir + os.path.sep + filename, sr=22050)
                # make all clips same length
                if (len(signal) / sr) < 3:
                    # print(f"filename: {filename} is not long enough, padding with zeroes")
                    signal = np.concatenate([[0] * ((sr * 3) - len(signal)), signal])

                signal = signal[:sr*3]
                # extract the MFCCs
                MFCCs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                # add dimension to input
                ret_data[dir].append(MFCCs.T)

    with open(pkl_save_path, "wb") as f:
        pickle.dump(ret_data, f)


def load_data(pkl_file_path: str) -> Dict[str, List[np.ndarray]]:
    """

    :param pkl_file_path: String
        Path to the location of pickle file storing labels and MFCCs
    :return: dict

        key --> Labels/Actors i.e. Actor_01, Actor_01 etc

        value --> [[MFCCs], [MFCCs], ...]
    """
    with open(pkl_file_path, "rb") as f:
        return pickle.load(f)


def create_dataset(pkl_file_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    This function will take in a pickle file storing labels and MFCCs, and return X, and y variables for splitting and
    training.
    :param pkl_file_path: String
        Path to pickle file containing labels and MFCCs
    :return: tuple
        X and y datasets for splitting, or list depending on how method is called.

    :example: X, y = create_dataset(pickle_file)
    """
    loaded_data = load_data(pkl_file_path)
    raw_data = ([], [])
    for actor, list_mfccs in loaded_data.items():
        for single_mfcc in list_mfccs:
            raw_data[0].append(single_mfcc)
            raw_data[1].append(actor)

    return raw_data


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    CURRENT_DIRECTORY = "D:/coding/kaggle/speech_recognition"
    root_audio_dir = "D:/coding/kaggle/speech_recognition/data"
    CREATE_DATA = False

    if CREATE_DATA:
        create_data(root_audio_dir, CURRENT_DIRECTORY + "/data.pkl", n_mfcc=20, hop_length=512, n_fft=2048)

    X, y = create_dataset("data.pkl")
    y = [int(actor.split("_")[1]) for actor in y]

    # print(X[0])
    # print(y[0])

    # create train, validation, test split
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_size)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_validation = np.array(X_validation)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_validation = np.array(y_validation)

    from keras.utils import np_utils, to_categorical
    from sklearn.preprocessing import LabelEncoder

    # Label encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.transform(y_test))
    y_validation = np_utils.to_categorical(lb.transform(y_validation))

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]


    loss = "categorical_crossentropy"

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape=input_shape, loss=loss)

    history = train(model=model, batch_size=48, X_train=X_train, y_train=y_train,
                    X_validation=X_validation, y_validation=y_validation)

    plot_history(history=history)

