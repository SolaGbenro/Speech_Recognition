import tensorflow as tf
import matplotlib.pyplot as plt


def build_model(input_shape, loss='categorical_crossentropy', learning_rate=0.001):
    """
    This Method will build a deep CNN with the addition of BatchNormalization and then print the model summary before
    returning the model.
    :param input_shape: Tuple
        The first index will come directly from the MFCC feature matrix. First will be the number of rows in the
        transposed feature matrix (value is variable, default=130 for 3 seconds of audio), next will be the columns in
        the transposed feature matrix (Value is variablem default=13).
    :param loss: String
        categorical_crossentropy produces a one-hot array containing the probable match for each category.

        sparse_categorical_crossentropy produces a category index of the most likely matching category.
    :param learning_rate: Float
        Starting point for the Adam optimizer
    :return: tf.keras Model
        First this will print the model summary, then return the model
    """
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    tf.keras.layers.Dropout(0.3)

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    tf.keras.layers.Dropout(0.3)

    # 4th conv layer
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    tf.keras.layers.Dropout(0.5)

    # softmax output layer
    model.add(tf.keras.layers.Dense(24, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, batch_size, X_train, y_train, X_validation, y_validation, save_path="default_model_name.h5",
          epochs=200, patience=5, verbose=1):
    """
    This function institutes the earlystop callback. It is initiated with a high epoch count with the condition that
    after a 'patience' number of epochs with no improvement, training with stop-early.
    :param model: tf.keras Model
        Built using the build_model() function.
    :param batch_size: Int
        Batch_size is the number of samples to take in at once. Larger numbers may help with generalization.
    :param X_train: numpy array
        MFCC feature matrix
    :param y_train: numpy array
        one-hot-encoded categorical variables for the actors i.e. Actor_01, Actor_02
    :param X_validation: numpy array
        Subset of original data not seen during training. used for val_accuracy
    :param y_validation: numpy array
        Subset of original data not seen during training. used for val_accuracy
    :param save_path: String
        Location to save model after fitting
    :param epochs: Int
        By default the epochs are set to a high number because of the institution of the early_stop callback
    :param patience: Int
        Number of epochs to wait with no improvement before stopping training
    :param verbose: Int
        How much information should be printed during the training process.
    :return: tf.keras history.history
        The newly created models metric history which can be used for in-depth analysis
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, verbose=verbose)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_validation, y_validation), callbacks=[earlystop_callback])

    if save_path:
        model.save(save_path)
    return history


def plot_history(history, save_plot=False):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return: Plot
        Plot of accuracy vs validation accuracy, and loss vs validation loss
    """

    fig, axs = plt.subplots(2, figsize=(10, 8))

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    if save_plot:
        fig.savefig('accuracy_loss.png', dpi=1000)
    plt.tight_layout()
    plt.show()
