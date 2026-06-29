import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ---------------------------------------
# LOAD MNIST DATASET
# ---------------------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

# ---------------------------------------
# PREPROCESSING
# ---------------------------------------

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ---------------------------------------
# BUILD CNN MODEL
# ---------------------------------------

model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation="relu",
    input_shape=(28,28,1)
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(
    64,
    (3,3),
    activation="relu"
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(
    128,
    activation="relu"
))

model.add(Dropout(0.3))

model.add(Dense(
    10,
    activation="softmax"
))

# ---------------------------------------
# COMPILE MODEL
# ---------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ---------------------------------------
# EVALUATE MODEL
# ---------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy :", round(accuracy*100,2), "%")

# ---------------------------------------
# SAVE MODEL
# ---------------------------------------

model.save("digit_model.keras")

print("\nModel Saved Successfully!")