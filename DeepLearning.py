import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_train.min())
print(x_train.max())
print(x_train[0])

# Flatten and normalize inputs
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Convert labels to one-hot encoding
num_cat = 10
y_train = utils.to_categorical(y_train, num_cat)
y_test = utils.to_categorical(y_test, num_cat)

# Build model
model = models.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(784,)))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# Compile and train model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Predict the class of the second test sample
predicted_class = model.predict(x_test[1].reshape(1, 784)).argmax(axis=1)
print("Predicted class:", predicted_class[0])
