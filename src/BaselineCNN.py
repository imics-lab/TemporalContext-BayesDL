from tensorflow import keras #added to save model
from tensorflow.keras import layers 
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def modelCNN(x_train, y_train, k_size):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = keras.Sequential(
		[
			keras.Input(shape=(n_timesteps,n_features)),
			layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Dropout(0.5),
			layers.MaxPooling1D(pool_size=2),
			layers.Flatten(),
			layers.Dense(100, activation='relu'),
			layers.Dense(n_outputs, activation='softmax')
   		]
	)
	return model


def train_model(model, x_train, y_train, x_valid, y_valid,
    BATCH_SIZE ,    # Typical values are 8, 16 or 32
    NUM_EPOCHS ): # Max number run unless earlystopping callback fires
    # see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    callback = EarlyStopping(monitor='val_loss', mode = 'min', patience=7)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x_train,y_train,
        batch_size = BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[callback],
        validation_data=(x_valid,y_valid),
        verbose = 1) #0 = silent, 1 = progress bar, 2 = one line per epoch
    return history, model


def run_model(model, x_test):
    predictions = model.predict(x_test, verbose = 0, batch_size = 64)
    #must use values not one-hot encoding, use argmax to convert
    y_pred = np.argmax(predictions, axis=-1) # axis=-1 means last axis
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1, keepdims=True)
    return y_pred, probabilities
