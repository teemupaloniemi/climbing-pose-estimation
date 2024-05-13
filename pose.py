import math
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
filename = ''
while (filename != 'q'):
    filename = input("Give filename with ending>")
    cogs = []
    handsl = []
    handsr = []
    feetl = []
    feetr = []
    keep = False
    perm = False
    cap = cv2.VideoCapture(filename)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image, (math.floor(1440/1.5), math.floor(1080/2)), interpolation= cv2.INTER_LINEAR)
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # roughly approximating this https://robslink.com/SAS/democd79/body_part_weights.htm
        weights = [
            8.26,  # 0 - Head (nose) - Approximated as part of the head
            0.0,   # 1 - left eye (inner) - Excluded, included in head
            0.0,   # 2 - left eye - Excluded, included in head
            0.0,   # 3 - left eye (outer) - Excluded, included in head
            0.0,   # 4 - right eye (inner) - Excluded, included in head
            0.0,   # 5 - right eye - Excluded, included in head
            0.0,   # 6 - right eye (outer) - Excluded, included in head
            0.0,   # 7 - left ear - Excluded, included in head
            0.0,   # 8 - right ear - Excluded, included in head
            0.0,   # 9 - mouth (left) - Excluded, included in head
            0.0,   # 10 - mouth (right) - Excluded, included in head
            12.5,  # 11 - left shoulder - Part of upper arm
            12.5,  # 12 - right shoulder - Part of upper arm
            1.87,  # 13 - left elbow - Part of forearm
            1.87,  # 14 - right elbow - Part of forearm
            0.65,  # 15 - left wrist - Hand
            0.65,  # 16 - right wrist - Hand
            0.0,   # 17 - left pinky - Excluded, included in hand
            0.0,   # 18 - right pinky - Excluded, included in hand
            0.0,   # 19 - left index - Excluded, included in hand
            0.0,   # 20 - right index - Excluded, included in hand
            0.0,   # 21 - left thumb - Excluded, included in hand
            0.0,   # 22 - right thumb - Excluded, included in hand
            12.5,  # 23 - left hip - Pelvis
            12.5,  # 24 - right hip - Pelvis
            15,  # 25 - left knee - Thigh
            15,  # 26 - right knee - Thigh
            2.35,  # 27 - left ankle - Leg
            2.35,  # 28 - right ankle - Leg
            1,  # 29 - left heel - Foot
            1,  # 30 - right heel - Foot
            0.0,   # 31 - left foot index - Excluded, included in foot
            0.0    # 32 - right foot index - Excluded, included in foot
        ]
        print(sum(weights))

        land = False
        if results.pose_landmarks is not None:
            land = True
            
            # Initialize sums for each coordinate
            sum_x = sum(landmark.x * weights[i] for i, landmark in enumerate(results.pose_landmarks.landmark))
            sum_y = sum(landmark.y * weights[i] for i, landmark in enumerate(results.pose_landmarks.landmark))
            if perm:
                handsl.append((results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y))
                handsr.append((results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y))
                feetl.append((results.pose_landmarks.landmark[31].x, results.pose_landmarks.landmark[31].y))
                feetr.append((results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y))
     
            cog_x = sum_x / 100 
            cog_y = sum_y / 100
            
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        shape = np.zeros_like(image, np.uint8)
        shape = cv2.rectangle(shape, (0,0), (image.shape[0]*2, image.shape[1]), (0, 0, 0), -1)
        alpha = 0.8
        out = image.copy()
        out = cv2.addWeighted(shape, alpha, image, 1 - alpha, 0)
        image = out
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if cv2.waitKey(1) == ord('a'):
            perm = True
        if cv2.waitKey(1) == ord('k'):
            keep = True
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
        if land and perm:
            cogs.append((cog_x, cog_y))
            for cog in cogs:
                cv2.circle(image, (math.floor(cog[0] * image.shape[0] * 1.8), math.floor(cog[1] * image.shape[1] / 1.8)), 2, (0,0,255), -1)
            for i in range(len(handsr)):
                cv2.circle(image, (math.floor(handsl[i][0] * image.shape[0] * 1.8), math.floor(handsl[i][1] * image.shape[1] / 1.8)), 2, (128,255,255), -1)
                cv2.circle(image, (math.floor(handsr[i][0] * image.shape[0] * 1.8), math.floor(handsr[i][1] * image.shape[1] / 1.8)), 2, (0,255,255), -1)
                cv2.circle(image, (math.floor(feetl[i][0] * image.shape[0] * 1.8), math.floor(feetl[i][1] * image.shape[1] / 1.8)), 2, (255,128,255), -1)
                cv2.circle(image, (math.floor(feetr[i][0] * image.shape[0] * 1.8), math.floor(feetr[i][1] * image.shape[1] / 1.8)), 2, (255,0,255), -1)


        cv2.imshow("iamge", image)
        
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()
        
cv2.destroyAllWindows()
