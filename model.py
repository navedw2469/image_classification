import pandas as pd
import numpy as np
import tensorflow as tf


# Load your CSV files, skipping the first row (header)
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')


# Extract labels and pixel values
train_labels = train_data['label'].values
train_images = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)

test_labels = test_data['label'].values
test_images = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)


# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

model.save('fashion_mnist_cnn_model.keras')
