import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

image_ids = np.arange(1, 101)

train_ids, test_ids = train_test_split(image_ids, test_size=0.3, random_state=42)  
val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = imread(image_path)
    image_resized = resize(image, target_size, anti_aliasing=True)
    image_normalized = image_resized / 255.0
    return image_normalized


test_images_dir = '/Users/zafar/Desktop/Northeastern University/Neural Networks/Test images'

test_image_filenames = [f"img_{i}.jpg" for i in range(10)]

test_images = np.array([load_and_preprocess_image(os.path.join(test_images_dir, filename)) for filename in test_image_filenames])