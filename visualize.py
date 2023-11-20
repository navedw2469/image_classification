import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('fashion_mnist_cnn_model.keras')
test_data = pd.read_csv('fashion-mnist_test.csv')

test_labels = test_data['label'].values
test_images = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)

f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=12
THIRD_IMAGE=20
CONVOLUTION_NUMBER = 2


layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(True)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(True)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(True)

plt.show()