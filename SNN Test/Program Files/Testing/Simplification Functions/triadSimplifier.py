import cv2
import dlib
import numpy as np

# Load the shape predictor file
predictor_path = "D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Load the image
image_path = r"D:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\Data\scienceFairFaces\face6.jpg"
image = cv2.imread(image_path)
scale_percent = 600  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Dlib's detector to find the bounding boxes of the faces in the image
detector = dlib.get_frontal_face_detector()
face_bounds = detector(gray, 1)

# Loop over the faces
for (i, face) in enumerate(face_bounds):
    # Determine the facial landmarks for the face
    shape = predictor(gray, face)

    # Convert the facial landmark coordinates to a numpy array
    points = shape_to_array(shape)
    # Select the three points
    point1 = points[1]
    point2 = points[48]
    point3 = points[5]

    # Create a mask with the same size as the image
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Fill in the mask with the average color of the points inside the triangle
    mask = cv2.fillPoly(mask, np.array([[point1, point2, point3]], dtype=np.int32), (0, 0, 0))
    average_color = cv2.mean(image, mask=mask)[:3]
    mask = cv2.fillPoly(mask, np.array([[point1, point2, point3]], dtype=np.int32), average_color)

    # Apply the mask to the image
    image = cv2.bitwise_and(image, mask)

    # Draw the triangle on top of the image
    cv2.line(image, tuple(point1), tuple(point2), (0, 0, 255), 2)
    cv2.line(image, tuple(point2), tuple(point3), (0, 0, 255), 2)
    cv2.line(image, tuple(point3), tuple(point1), (0, 0, 255), 2)
    print("done")
else:
    print("No face found")
# Show the image
cv2.imshow("Filled Triangle", image)
cv2.waitKey(0)
