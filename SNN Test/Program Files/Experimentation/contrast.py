import cv2
import numpy as np

def adjust_contrast(image, contrast_factor):
    # Convert image to floating point format
    image = np.float32(image)

    # Calculate new contrast adjusted image
    image = image * contrast_factor

    # Clip image values to range [0, 255]
    image = np.clip(image, 0, 255)

    # Convert image back to uint8 format
    image = np.uint8(image)

    return image

'''
def adjust_brightness(image, brightness_factor):
    # Convert image to floating point format
    image = np.float32(image)

    # Calculate new brightness adjusted image
    image = image + brightness_factor

    # Clip image values to range [0, 255]
    image = np.clip(image, 0, 255)

    # Convert image back to uint8 format
    image = np.uint8(image)

    return image
'''
def increase_brightness(img, value=0):
    img = np.float32(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = np.uint8(img)
    return img
'''
def adjust_contrast_brightness_ui(image):
    # Create a window to display the image and scroll bars
    cv2.namedWindow('Contrast and Brightness Adjustment')

    # Create trackbars to adjust the contrast and brightness factors
    cv2.createTrackbar('Contrast Factor', 'Contrast and Brightness Adjustment', 10, 100, lambda x: x)
    cv2.createTrackbar('Brightness Factor', 'Contrast and Brightness Adjustment', 0, 100, lambda x: x)

    while True:
        # Get the current values of the contrast and brightness factors from the trackbars
        contrast_factor = cv2.getTrackbarPos('Contrast Factor', 'Contrast and Brightness Adjustment')
        brightness_factor = cv2.getTrackbarPos('Brightness Factor', 'Contrast and Brightness Adjustment')

        # Convert contrast and brightness factors to floats in the range [0, 1]
        contrast_factor = contrast_factor / 10.0
        brightness_factor = brightness_factor / 1.0
        print(contrast_factor, brightness_factor)
        # Adjust the contrast and brightness of the image
        contrast_adjusted_image = adjust_contrast(image, contrast_factor)
        #contrast_brightness_adjusted_image = adjust_brightness(contrast_adjusted_image, brightness_factor)
        contrast_brightness_adjusted_image = increase_brightness(contrast_adjusted_image, brightness_factor)
        # Display the contrast and brightness adjusted image
        cv2.imshow('Contrast and Brightness Adjustment', contrast_brightness_adjusted_image)

        # Check if the user has pressed the 'ESC' key
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Destroy the window
    cv2.destroyAllWindows()

# Load an image
image = cv2.imread(r'D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairTriangulatedFaces\simplifiedFace16.jpg')

scale_percent = 600  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim)
'''
# Adjust the contrast and brightness of the image using the UI
# adjust_contrast_brightness_ui(image)