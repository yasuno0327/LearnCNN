import keras

(train_data, train_label), (test_data, test_label) = keras.datasets.mnist.load_data()

model = keras.models.Sequential()

model.add(keras.layers.Dense(256, input_dim=784))
model.add(keras.layers.Activation("sigmoid"))

model.add(keras.layers.Dense(128))
model.add(keras.layers.Activation("sigmoid"))

model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation("sigmoid"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, train_label, verbose=1)

score = model.evaluate(train_data, train_label, verbose=1)

print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))