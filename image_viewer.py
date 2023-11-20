import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, PhotoImage, Canvas

# Load the CSV file
df = pd.read_csv("fashion-mnist_test.csv")

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

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.current_index = 1

        # Create a canvas for displaying images
        self.canvas = Canvas(root, width=580, height=580, bg='white')
        self.canvas.pack()

        # Create "Previous" and "Next" buttons
        prev_button = Button(root, text="Previous", command=self.show_previous)
        prev_button.pack(side='left')

        next_button = Button(root, text="Next", command=self.show_next)
        next_button.pack(side='right')

        # Display the initial image
        self.display_image()

    def display_image(self):
        # Extract the label and pixel values of the current row
        label = df.iloc[self.current_index, 0]
        pixels = df.iloc[self.current_index, 1:]

        # Convert pixel values to a 28x28 NumPy array
        image_array = np.array(pixels).reshape(28, 28)

        # Display the image on the canvas
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        plt.savefig('temp.png')  # Save the figure
        plt.clf()  # Clear the figure
        self.photo = PhotoImage(file='temp.png')  # Load the saved image
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        # Update the title with the label
        self.root.title(f"Label: {class_labels[label]}")

    def show_previous(self):
        if self.current_index > 1:
            self.current_index -= 1
            self.display_image()

    def show_next(self):
        if self.current_index < len(df) - 1:
            self.current_index += 1
            self.display_image()

if __name__ == "__main__":
    root = Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageViewer class
    viewer = ImageViewer(root)

    # Run the Tkinter event loop
    root.mainloop()
