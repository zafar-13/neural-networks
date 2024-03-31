import numpy as np

def convolve2D(image, kernel):

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output_dim = np.array(image.shape) - np.array(kernel.shape) + 1  # Output dimensions
    output = np.zeros(output_dim)
    
    # Iterate over every pixel in the image
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            # Apply the kernel on the current region of interest in the image
            output[x, y] = (image[x:x+3, y:y+3] * kernel).sum()
    
    return output

# Example usage
image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

convolved_image = convolve2D(image, kernel)
print(convolved_image)
