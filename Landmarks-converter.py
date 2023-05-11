import itertools
import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

path = "D:\\asl_data\\asl_alphabet_train"

width = 200
height = 200

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'space']

value = 30

# # def calc_landmark_list(image, landmarks):
# #     image_width, image_height = image.shape[1], image.shape[0]

# #     landmark_point = []

# #     # Keypoint
# #     for _, landmark in enumerate(landmarks.landmark):
# #         landmark_x = min(int(landmark.x * image_width), image_width - 1)
# #         landmark_y = min(int(landmark.y * image_height), image_height - 1)
# #         # landmark_z = landmark.z

# #         landmark_point.append([landmark_x, landmark_y])

# #     return landmark_point


# # def pre_process_landmark(landmark_list):
# #     temp_landmark_list = copy.deepcopy(landmark_list)

# #     # Convert to relative coordinates
# #     base_x, base_y = 0, 0
# #     for index, landmark_point in enumerate(temp_landmark_list):
# #         if index == 0:
# #             base_x, base_y = landmark_point[0], landmark_point[1]

# #         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
# #         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

# #     # Convert to a one-dimensional list
# #     temp_landmark_list = list(
# #         itertools.chain.from_iterable(temp_landmark_list))

# #     # Normalization
# #     max_value = max(list(map(abs, temp_landmark_list)))

# #     def normalize_(n):
# #         return n / max_value

# #     temp_landmark_list = list(map(normalize_, temp_landmark_list))

# #     return temp_landmark_list[2:]


# image = cv2.imread(path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# results = hands.process(image)

# if results.multi_hand_landmarks is not None:
#     hand_landmarks = results.multi_hand_landmarks[0]
#     # Landmark calculation
#     landmark_list = []

#     # Keypoint
#     for _, landmark in enumerate(hand_landmarks.landmark):
#         landmark_x = min(int(landmark.x * width), width - 1)
#         landmark_y = min(int(landmark.y * height), height - 1)

#         landmark_list.append([landmark_x, landmark_y])

#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, point in enumerate(landmark_list):
#         if index == 0:
#             base_x, base_y = point[0], point[1]

#         landmark_list[index][0] = landmark_list[index][0] - base_x
#         landmark_list[index][1] = landmark_list[index][1] - base_y

#     # Convert to a one-dimensional list
#     landmark_list = list(
#         itertools.chain.from_iterable(landmark_list))

#     # Normalization
#     max_value = max(list(map(abs, landmark_list)))

#     def normalize_(n):
#         return n / max_value

#     landmark_list = list(map(normalize_, landmark_list))[2:]

#     print(landmark_list)


x = []
y = []

count = 0

for idx, label in enumerate(classes):
    if idx == 1: break
    folder_path = os.path.join(path, label)
    for image_filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks is not None:
            count += 1
            print(count)
            hand_landmarks = results.multi_hand_landmarks[0]
            # Landmark calculation
            landmark_list = []
            # Keypoint
            for _, landmark in enumerate(hand_landmarks.landmark):
                landmark_x = min(int(landmark.x * width), width - 1)
                landmark_y = min(int(landmark.y * height), height - 1)
                landmark_list.append([landmark_x, landmark_y])
            # Convert to relative coordinates
            base_x, base_y = 0, 0
            for index, point in enumerate(landmark_list):
                if index == 0:
                    base_x, base_y = point[0], point[1]
                landmark_list[index][0] = landmark_list[index][0] - base_x
                landmark_list[index][1] = landmark_list[index][1] - base_y
            # Convert to a one-dimensional list
            landmark_list = list(
                itertools.chain.from_iterable(landmark_list))
            # Normalization
            max_value = max(list(map(abs, landmark_list)))

            def normalize_(n):
                return n / max_value
            landmark_list = list(map(normalize_, landmark_list))[2:]
            x.append(landmark_list)
            y.append(label)

y = np.array(y)
print(y.shape)
