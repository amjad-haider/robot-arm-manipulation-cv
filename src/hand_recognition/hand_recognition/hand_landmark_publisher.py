import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
GESTURE_TEXT_COLOR = (0, 0, 255)
IMAGE_FLIPPED = True

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Annotates an RGB image with detected hand landmarks, handedness, and gesture information.

    Parameters:
        rgb_image (np.ndarray): The input RGB image to annotate.
        detection_result: The result object containing hand landmarks, handedness, and gestures.

    Returns:
        np.ndarray: The annotated RGB image with hand landmarks, handedness, and gesture labels drawn.
    """
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    gestures_list = detection_result.gestures
    gestures = gestures_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    if IMAGE_FLIPPED:
        if handedness[0].category_name == 'Left':
            handedness_text = 'Right'
        else:
            handedness_text = 'Left'
    else:
        handedness_text = handedness[0].category_name

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness_text}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    if gestures:
        gesture = gestures[0] # Get the top gesture
        gesture_text = f"{gesture.category_name} ({gesture.score:.2f})"
        
        # Calculate position for the gesture text (below the handedness text)
        text_y_gesture = text_y + (MARGIN * 3) 
        
        cv2.putText(annotated_image, gesture_text,
                    (text_x, text_y_gesture), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, GESTURE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4)
	
# recognizer = GestureRecognizer.create_from_options(options)
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

# gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        if IMAGE_FLIPPED:
            frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
        annotated_image = draw_landmarks_on_image(frame, gesture_recognition_result)
        cv2.imshow('MediaPipe Gesture Recognition', annotated_image)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
