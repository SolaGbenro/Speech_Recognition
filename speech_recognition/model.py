import tensorflow as tf
import matplotlib.pyplot as plt


def build_model(input_shape, loss='categorical_crossentropy', learning_rate=0.001):
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # TODO: REMOVE DROPOUT BELOW AFTER TESTING
    tf.keras.layers.Dropout(0.1)

    # # 2nd conv layer
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # # TODO: REMOVE DROPOUT BELOW AFTER TESTING
    # tf.keras.layers.Dropout(0.1)

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

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


def train(model, batch_size, X_train, y_train, X_validation, y_validation, save_path="model_2.h5",
          epochs=200, patience=5, verbose=1):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, verbose=verbose)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_validation, y_validation), callbacks=[earlystop_callback])

    model.save(save_path)
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return: Plot
        Plot of accuracy vs validation accuracy, and loss vs validation loss
    """

    fig, axs = plt.subplots(2)

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

    plt.show()
