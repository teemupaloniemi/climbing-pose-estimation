import math
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe drawing and pose solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize the filename variable
filename = ''

# Print usage instructions
print("Usage:\n    1. Provide file path. (give 'l' or 'list' to list current directory)\n    2. press 'a' to start tracking\n    3. press 'k' to keep tracked buffer stored\n    4. Clear everything :) by pressing 'c'\n    5. Quit by pressing 'q' or giving 'q' as a filename\n")

# Main loop to process user inputs and video files
while filename != 'q':
    filename = input("Give filename with ending>")

    if filename in ['l', 'list']:
        # List files in the current directory
        for item in os.listdir('.'):
            print(item)
        continue

    if filename == 'q':
        break

    # Initialize tracking lists
    cogs = []
    handsl = []
    handsr = []
    feetl = []
    feetr = []
    keep = False
    perm = False

    # Capture video from the given file
    cap = cv2.VideoCapture(filename)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Preprocess the image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Body part weights for calculating center of gravity
            # roughly approximating this https://robslink.com/SAS/democd79/body_part_weights.htm
            weights = [
                8.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                12.5, 12.5, 1.87, 1.87, 0.65, 0.65, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 12.5, 12.5, 15, 15, 2.35, 2.35, 1, 1, 0.0, 0.0
            ]

            land = False
            if results.pose_landmarks is not None:
                land = True

                # Calculate weighted sums for each coordinate
                sum_x = sum(landmark.x * weights[i] for i, landmark in enumerate(results.pose_landmarks.landmark))
                sum_y = sum(landmark.y * weights[i] for i, landmark in enumerate(results.pose_landmarks.landmark))
                sum_z = sum(landmark.z * weights[i] for i, landmark in enumerate(results.pose_landmarks.landmark))

                # Store hand and foot positions if tracking is permanent
                if perm:
                    handsl.append((results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y))
                    handsr.append((results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y))
                    feetl.append((results.pose_landmarks.landmark[31].x, results.pose_landmarks.landmark[31].y))
                    feetr.append((results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y))

                # Calculate the center of gravity
                cog_x = sum_x / 100 
                cog_y = sum_y / 100
                cog_z = sum_z / 100

            # Draw the pose annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            shape = np.zeros_like(image, np.uint8)
            shape = cv2.rectangle(shape, (0, 0), (image.shape[0]*2, image.shape[1]), (0, 0, 0), -1)
            alpha = 0.75
            out = image.copy()
            out = cv2.addWeighted(shape, alpha, image, 1 - alpha, 0)
            image = out
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Handle key press events for tracking options
            if cv2.waitKey(5) == ord('a'):
                perm = True

            if cv2.waitKey(5) == ord('k'):
                keep = True

            if cv2.waitKey(5) == ord('c'):
                # Clear all tracking lists
                handsl = []
                handsr = []
                feetl = []
                feetr = []
                cogs = []
                keep = False

            # Maintain a buffer of the last 128 tracked positions
            if not keep:
                if len(handsr) > 128:
                    handsr.pop(0)
                if len(handsl) > 128:
                    handsl.pop(0)
                if len(feetl) > 128:
                    feetl.pop(0)
                if len(feetr) > 128:
                    feetr.pop(0)
                if len(cogs) > 128:
                    cogs.pop(0)

            # Calculate aspect ratio
            asp = image.shape[0] / image.shape[1]

            # Draw the tracked positions on the image
            if land and perm:
                cogs.append((cog_x, cog_y, cog_z))
                for cog in cogs:
                    cv2.circle(image, (math.floor(cog[0] * image.shape[0] / asp), math.floor(cog[1] * image.shape[1] * asp)), 2, (100 * cog[2], 100 * cog[2], 1000 * cog[2] * 128), -1)
                for i in range(len(handsr)):
                    cv2.circle(image, (math.floor(handsl[i][0] * image.shape[0] / asp), math.floor(handsl[i][1] * image.shape[1] * asp)), 2, (64, 255, 0), -1)
                    cv2.circle(image, (math.floor(handsr[i][0] * image.shape[0] / asp), math.floor(handsr[i][1] * image.shape[1] * asp)), 2, (0, 255, 0), -1)
                    cv2.circle(image, (math.floor(feetl[i][0] * image.shape[0] / asp), math.floor(feetl[i][1] * image.shape[1] * asp)), 2, (255, 64, 0), -1)
                    cv2.circle(image, (math.floor(feetr[i][0] * image.shape[0] / asp), math.floor(feetr[i][1] * image.shape[1] * asp)), 2, (255, 0, 0), -1)

            # Resize and display the image based on aspect ratio
            if asp < 1:
                cv2.imshow("image", cv2.resize(image, (1000, 600)))
            else:
                cv2.imshow("image", cv2.resize(image, (600, 1000)))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(5) == ord('q'):
                break

    # Release video capture
    cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()

