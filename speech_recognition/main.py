import os
import tensorflow as tf
import numpy as np
import tqdm as tqdm
import pickle
import librosa
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from sklearn.preprocessing import LabelEncoder

from model import build_model, train, plot_history


def save_mfccs(root_audio_dir: str, pkl_save_path: str, n_mfcc: int = 13,
               hop_length: int = 512, n_fft: int = 2048) -> None:
    """
    This method will take in raw audio samples, then store them in a pickled dictionary object where the keys are the
    labels, and the values are the calculated MFCCs per sample. If the sample wav file is less than 3 seconds long,
    it will be zero-padded up to 3 seconds. The default values that are passed were chosen partly arbitrarily and
    partly from experience. They can/should be changed at will.

    :param root_audio_dir: String
        Path to directory storing all folders containing music samples.
    :param pkl_save_path: String
        Location pickled dictionary should be saved to.
    :param n_mfcc: int
        Number of features that will be extracted, default being 13 features per frame.
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


def load_mfccs(pkl_file_path: str) -> Dict[str, List[np.ndarray]]:
    """
    This function will load in a saved pickle object (This should contain the labels and MFCCs).

    :param pkl_file_path: String
        Path to the location of pickle file storing labels and MFCCs
    :return: dict

        key --> Labels/Actors i.e. Actor_01, Actor_01 etc

        value --> [[MFCCs], [MFCCs], ...]
    """
    with open(pkl_file_path, "rb") as f:
        return pickle.load(f)


def create_dataset(pkl_file_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    This function will take in a pickle file storing labels and MFCCs, and return X, and y variables for splitting and
    then training.
    :param pkl_file_path: String
        Path to pickle file containing labels and MFCCs
    :return: tuple
        X and y datasets for splitting.
    """
    loaded_data = load_mfccs(pkl_file_path)
    # raw_data = ([], [])
    X = []
    y = []
    for actor, list_mfccs in loaded_data.items():
        for single_mfcc in list_mfccs:
            # raw_data[0].append(single_mfcc)
            # raw_data[1].append(actor)
            X.append(single_mfcc)
            y.append(actor)

    # split labels to just contain numerical information on actor
    y = [int(actor.split("_")[1]) for actor in y]
    return X, y


def split_dataset(X, y, test_size=.3, validation_set=False, validation_test_size=0.1):
    """
    This method will be used to split data for testing and validation purposes along the given parameters.
    :param X: List
        X is a list of 2-dimensional numpy arrays. They hold the MFCCs from the 'create_data' method, i.e. training data
    :param y: List
        y is a list storing the labels belonging to the MFCCs in X
    :param test_size: Float
        The percentage of training data that will be withheld for fine tuning (hyper)parameters during training.
    :param validation_set: Boolean
        Boolean flag to determine whether a validation dataset should be created in addition to the training and
        testing set
    :param validation_test_size: Float
        The percentage of the remaining training set (after test set has been extracted) to be witheld from training
        altogether.
    :return: numpy arrays
        Training, testing and validation splits (optional)

            RETURN ORDER WITHOUT VALIDATION_SET:
            np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

            RETURN ORDER WITH VALIDATION_SET:
            np.array(X_train), np.array(X_test), np.array(X_validation), np.array(y_train), np.array(y_test),
            np.array(y_validation)

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    if validation_set:
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_test_size)
        # add a new axis to np array (create batches)
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]

        return np.array(X_train), np.array(X_test), np.array(X_validation), np.array(y_train), np.array(y_test), \
               np.array(y_validation)
    else:
        # add a new axis to np array (create batches)
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return X_train, X_test, y_train, y_test


def encode_target_variable(y_train, y_test, y_validation=None):
    """
    This method will encode target variables into categorical variables. This will be required for training with
    'categorical_crossentropy' loss.
    :param y_train: List
        Set examples uses for training
    :param y_test: List
        Subset of examples withheld for parameter tuning.
    :param y_validation: None or List
        If not none, then an additional subset of data withheld entirely from the training purposes and used to
        determine accuracy on unseen examples.
    :return: Tuple of Lists
        Either encoded training and testing sets, or encoded training, testing and validation.
    """
    # Label encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.transform(y_test))
    if len(y_validation) > 0:
        y_validation = np_utils.to_categorical(lb.transform(y_validation))

        return y_train, y_test, y_validation
    else:
        return y_train, y_test


if __name__ == '__main__':
    # set up config for tensorflow GPU support
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    CURRENT_DIRECTORY = "../speech_recognition"
    root_audio_dir = "../speech_recognition/data"
    CREATE_DATA = False

    if CREATE_DATA:
        save_mfccs(root_audio_dir, CURRENT_DIRECTORY + "/data.pkl", n_mfcc=20, hop_length=512, n_fft=2048)

    X, y = create_dataset("data.pkl")
    # y = [int(actor.split("_")[1]) for actor in y]

    # create train, validation, test split
    test_size = 0.2
    validation_test_size = 0.2
    X_train, X_test, X_validation, y_train, y_test, y_validation = \
        split_dataset(X, y, test_size=test_size, validation_set=True, validation_test_size=validation_test_size)

    # # Label encode the target variables
    y_train, y_test, y_validation = encode_target_variable(y_train=y_train, y_test=y_test, y_validation=y_validation)
    print(f"X_train shape: {X_train.shape}")

    loss = "categorical_crossentropy"
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    # create model
    model = build_model(input_shape=input_shape, loss=loss)
    # retain metrics after training for analysis
    model_history = train(model=model, batch_size=48, X_train=X_train, y_train=y_train,
                    X_validation=X_validation, y_validation=y_validation, save_path=None)
    # visualize training performance
    plot_history(history=model_history)
