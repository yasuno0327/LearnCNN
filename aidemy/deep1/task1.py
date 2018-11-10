import keras

(train_data, train_label), (test_data, test_label) = keras.datasets.mnist.load_data()

# Get size
# print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

# Initialize model instance
model = keras.models.Sequential()
model.add(keras.layers.Dense(128))

model.add(keras.layers.Activation("sigmoid"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
