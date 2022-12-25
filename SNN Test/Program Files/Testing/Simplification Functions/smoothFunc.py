import cv2
import numpy as np

def smooth_colors(image, smoothing_factor):
    # Convert image to floating point format
    image = np.float32(image)

    # Smooth the colors of the image using a Gaussian blur
    image = cv2.GaussianBlur(image, (31, 31), smoothing_factor)

    # Clip image values to range [0, 255]
    image = np.clip(image, 0, 255)

    # Convert image back to uint8 format
    image = np.uint8(image)

    return image

def smooth_colors_ui(image):
    # Create a window to display the image and trackbar
    cv2.namedWindow('Color Smoothing')

    # Create a trackbar to adjust the smoothing factor
    cv2.createTrackbar('Smoothing Factor', 'Color Smoothing', 0, 10, lambda x: x)

    while True:
        # Get the current value of the smoothing factor from the trackbar
        smoothing_factor = cv2.getTrackbarPos('Smoothing Factor', 'Color Smoothing')

        # Smooth the colors of the image
        smoothed_image = smooth_colors(image, smoothing_factor)

        # Display the smoothed image
        cv2.imshow('Color Smoothing', smoothed_image)

        # Check if the user has pressed the 'ESC' key
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Destroy the window
    cv2.destroyAllWindows()

# Load an image
image = cv2.imread(r'D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces\face5.jpg')

scale_percent = 600  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim)
# Smooth the colors of the image using the UI
smooth_colors_ui(image)