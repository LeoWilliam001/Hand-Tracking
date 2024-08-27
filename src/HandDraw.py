# import cv2
# import numpy as np
# import mediapipe as mp

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# draw_color = (255, 255, 255)  # Color for drawing
# erase_color = (0, 0, 0)       # Color for erasing

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Create a blank canvas to draw
# canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# # Initialize previous position variables
# prev_x, prev_y = 0, 0

# # Initialize drawing state
# drawing = False

# # Function to draw lines on canvas
# def draw_line(canvas, start, end, color, thickness=2):
#     cv2.line(canvas, start, end, color, thickness)

# # Function to erase the canvas
# def erase_canvas(canvas, color):
#     canvas[:] = color

# # Main loop
# while True:
#     # Read frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally
#     frame = cv2.flip(frame, 1)

#     # Convert frame to RGB for MediaPipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect hand landmarks
#     results = hands.process(frame_rgb)

#     # Initialize a variable to track the number of fingers up
#     fingers_up = []

#     # Draw landmarks and get hand positions
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for id, lm in enumerate(hand_landmarks.landmark):
#                 # Get x, y coordinates of each landmark
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)

#                 if id in [8]:  # Index, middle, ring, and pinky tips
#                     fingers_up.append(id)

#                 if id == 8:  # Index finger tip
#                     # Only draw if drawing mode is active and index finger is the only finger up
#                     if drawing and len(fingers_up) == 1:
#                         if prev_x != 0 and prev_y != 0:
#                             draw_line(canvas, (prev_x, prev_y), (cx, cy), draw_color)
#                         prev_x, prev_y = cx, cy
#                     else:
#                         # Reset previous positions to avoid unintended drawing
#                         prev_x, prev_y = 0, 0

#     # Display frame and canvas
#     cv2.imshow('Frame', frame)
#     cv2.imshow('Canvas', canvas)

#     # Check for key press to control functionality
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break
#     elif key & 0xFF == ord('e'):
#         erase_canvas(canvas, erase_color)  # Clear the canvas when 'e' is pressed
#     elif key & 0xFF == ord('s'):
#         drawing = not drawing  # Toggle drawing mode on 's' key press

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)       # Color for erasing

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize previous position variables
prev_x, prev_y = 0, 0

# Initialize drawing state
drawing = False

# Function to draw lines on canvas
def draw_line(canvas, start, end, color, thickness=5):
    cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)

# Function to erase the canvas
def erase_canvas(canvas, color):
    canvas[:] = color

# Function to add text to the canvas
def add_text(canvas, text, position, font_scale=1.5, font_thickness=3, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)

    # Initialize a variable to track the number of fingers up
    fingers_up = []

    # Draw landmarks and get hand positions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get x, y coordinates of each landmark
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id in [8]:  # Index finger tip
                    fingers_up.append(id)

                if id == 8:  # Index finger tip
                    # Only draw if drawing mode is active and index finger is the only finger up
                    if drawing and len(fingers_up) == 1:
                        if prev_x != 0 and prev_y != 0:
                            draw_line(canvas, (prev_x, prev_y), (cx, cy), draw_color)
                        prev_x, prev_y = cx, cy
                    else:
                        # Reset previous positions to avoid unintended drawing
                        prev_x, prev_y = 0, 0

    # Add text to the canvas
    add_text(canvas, 'Drawing', (10, 50))

    # Display frame and canvas
    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)

    # Check for key press to control functionality
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('e'):
        erase_canvas(canvas, erase_color)  # Clear the canvas when 'e' is pressed
    elif key & 0xFF == ord('s'):
        drawing = not drawing  # Toggle drawing mode on 's' key press

# Release resources
cap.release()
cv2.destroyAllWindows()
