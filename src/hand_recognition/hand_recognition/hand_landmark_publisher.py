import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory
import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
GESTURE_TEXT_COLOR = (0, 0, 255)
IMAGE_FLIPPED = True

package_name = 'hand_recognition'
model_path = os.path.join(
    get_package_share_directory(package_name),
    'resource/gesture_recognizer.task'
)

def draw_landmarks_on_image(rgb_image, detection_result):
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
class HandLandmarkPublisher(Node):

    def __init__(self):
        super().__init__('Hand_Landmark_Publisher')
        # --- ROS 2 Publishers ---
        self.image_publisher = self.create_publisher(Image, 'gesture/annotated_image', 10)
        self.gesture_publisher = self.create_publisher(String, 'gesture/recognized_gestures', 10)
        self.waypoint_publisher = self.create_publisher(PoseStamped, 'gesture/waypoints', 10)
        self.waypoint = PoseStamped() 
        self.header = Header()
        self.waypoint.header = self.header
        self.waypoint.pose.orientation.x = 0.0
        self.waypoint.pose.orientation.y = 0.0
        self.waypoint.pose.orientation.z = 0.0
        self.waypoint.pose.orientation.w = 1.0
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a gesture recognizer instance with the video mode:
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2)
        self.recognizer = GestureRecognizer.create_from_options(options)
        # --- OpenCV Initialization ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open video stream.")
            rclpy.shutdown()
            
        self.bridge = CvBridge()
        self.get_logger().info("Gesture Publisher Node has started.")

    def timer_callback(self):
        success, frame = self.cap.read()
        if not success:
            self.get_logger().warn("Ignoring empty camera frame.")
            return
        if IMAGE_FLIPPED:
            frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Perform recognition
        gesture_recognition_result = self.recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
        # Draw results on the *original* BGR frame
        annotated_image = draw_landmarks_on_image(frame, gesture_recognition_result)
        # 1. Publish Annotated Image
        ros_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ros_image_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(ros_image_msg)

        # 2. Publish Gesture String
        gesture_str = "None"
        if gesture_recognition_result.gestures:
            gesture_str = ""
            for i, gestures in enumerate(gesture_recognition_result.gestures):
                if gestures:
                    handedness = gesture_recognition_result.handedness[i][0].category_name
                    if IMAGE_FLIPPED:
                        handedness = "Left" if handedness == "Right" else "Right"
                    gesture_name = gestures[0].category_name
                    gesture_str += f"{handedness}: {gesture_name}; "

        # Publish waypoints for every detected hand, regardless of gesture recognition
        for i, hand_landmarks_centroid in enumerate(gesture_recognition_result.hand_landmarks):
            self.waypoint.header.stamp = self.get_clock().now().to_msg()
            self.waypoint.header.frame_id = 'camera_link'

            # These are the 7 landmarks from the example (palm base)
            palm_landmark_indices = [0, 1, 2, 5, 9, 14, 17]
            coords = np.array([
                    [hand_landmarks_centroid[idx].x * width, hand_landmarks_centroid[idx].y * height] 
                    for idx in palm_landmark_indices
                ])
                
            # Calculate the average (mean)
            centroid_2d = np.mean(coords, axis=0)
            self.waypoint.pose.position.x = float(centroid_2d[0])
            self.waypoint.pose.position.y = float(centroid_2d[1])
            self.waypoint_publisher.publish(self.waypoint)
        gesture_msg = String()
        gesture_msg.data = gesture_str
        self.gesture_publisher.publish(gesture_msg)

        # 3. (Optional) Display the frame locally
        cv2.imshow('MediaPipe Gesture Recognition', annotated_image)
        # We don't need 'q' to quit, ROS will handle shutdown
        cv2.waitKey(1)



    def destroy_node(self):
        # Clean up resources
        self.get_logger().info("Shutting down node, closing resources.")
        self.cap.release()
        cv2.destroyAllWindows()
        # self.recognizer.close() # .close() may not exist in all versions, let GC handle it
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Explicitly destroy the node on shutdown
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()