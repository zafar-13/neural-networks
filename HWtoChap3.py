import numpy as np
from PIL import Image

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.rand(input_size, 1)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.sigmoid(weighted_sum)
        return output

    def backward(self, inputs, target, output):
        error = target - output
        gradient = self.sigmoid_derivative(output)

        self.weights += self.learning_rate * np.dot(inputs.T, error * gradient)
        self.bias += self.learning_rate * np.sum(error * gradient)

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target_data = targets[i]
                output = self.forward(input_data)
                self.backward(input_data, target_data, output)

    def predict(self, inputs):
        return self.forward(inputs)

def test_images(directory_path):
    test_images = []
    for i in range(0, 9):
        image_path = "img_{i}.jpg"
        image = Image.open(image_path)
        image = image.convert("L").resize((20, 20))

        image_array = np.array(image) / 255.0
        test_images.append(image_array)
    return np.array(test_images)

def generate_data():
    np.random.seed(42)
    images = np.random.random(size=(10, 20, 20))
    labels = np.eye(10)[:, 0:1]

    return images, labels

# Test images for evaluation
def generate_test_data():
    np.random.seed(123)
    test_images = np.random.random(size=(5, 20, 20))

    return test_images

def flatten_images(images):
    return images.reshape(len(images), -1)

def main():

    images, labels = generate_data()
    perceptron = Perceptron(input_size=400)
    test_images_directory = "/Users/zafar/Desktop/Northeastern University/Neural Networks/Test images"

    test_images = test_images(test_images_directory)

    flattened_test_images = flatten_images(test_images)
    perceptron.train(flatten_images, labels, epochs=1000)


    for i in range(len(flattened_test_images)):
        test_input = flattened_test_images[i]
        prediction = perceptron.predict(test_input)
        print(f"Test Image {i + 1} Prediction: {prediction}")

if __name__ == "__main__":
    main()