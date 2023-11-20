from flask import Flask, render_template, request, url_for
import os
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the saved model
loaded_model = tf.keras.models.load_model('fashion_mnist_cnn_model.keras')

# Map of class labels
class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return render_template('upload.html', error="No file selected. Please choose a file.")

        if file:
            # Save the uploaded file
            file_path = 'static/uploads/uploaded_image.jpg'
            file.save(file_path)

            # Pass the file path to the template for display
            return render_template('upload.html', image_path=url_for('static', filename='/uploads/uploaded_image.jpg'))

    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess the image
    file_path = 'static/uploads/uploaded_image.jpg'
    img = preprocess_image(file_path)

    # Make predictions
    predictions = loaded_model.predict(img)

    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    # Pass the file path and predicted label to the template
    return render_template('result.html', image_path=url_for('static', filename='/uploads/uploaded_image.jpg'), predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
