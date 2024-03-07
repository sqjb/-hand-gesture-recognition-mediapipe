import csv

import numpy as np
import tensorflow as tf
import copy
import itertools
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
mp_hands = mp.solutions.hands

RANDOM_SEED = 42

# 模型路径
model_path = './hand_gesture/hand_gesture.hdf5'

NUM_CLASSES = 4
gesture_name = ['open','close','pointer','ok']


# 手部点连接顺序
bone_links = [
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20)
]
# 手部点颜色
joint_colors = (
        (255, 0, 0),  # thumb  - Blue
        (0, 255, 0),  # index  - Green
        (0, 0, 255),  # middle - Read
        (255, 255, 0),  # ring   - Pink
        (0, 255, 255),  # pink   - Yellow
)

# 处理输入数据
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 把数据转换为相对深度
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 转为1维向量
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 归一化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# 绘制手势
def draw_hand(img, coord,type):
    gesture = gesture_name[type]
    for i in range(len(bone_links)):
        bone_link = bone_links[i]
        for j in bone_link:
            cv2.circle(img, tuple(coord[j]), 1, joint_colors[i], -1)
        for j, nj in zip(bone_link[:-1], bone_link[1:]):
            cv2.line(img, tuple(coord[j]), tuple(coord[nj]), joint_colors[i])
    # 写分类信息
    cv2.putText(img,gesture,(int(coord[0][0]),int(coord[0][1])),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    return img


if __name__ == '__main__':

    model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
    model.summary()

    # 加载训练好的路径
    model = tf.keras.models.load_model(model_path)

    # mediapipe 手势
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.1,
    )
    # 加载数据
    img_name = './test_image/pointer.jpg'
    img = cv2.imread(img_name)

    h, w = img.shape[:2]
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hand_result = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for joint_idx, xyz in enumerate(hand_landmarks.landmark):
                hand_result[joint_idx][0] = xyz.x * w
                hand_result[joint_idx][1] = xyz.y * h
                hand_result[joint_idx][2] = xyz.z * w

    coord = np.array(hand_result[:, :2], dtype=np.int32)

    hand_ske = pre_process_landmark(coord)

    # 推理
    predict_result = model.predict(np.array([hand_ske]))
    print(np.squeeze(predict_result))
    type = np.argmax(np.squeeze(predict_result))

    img = draw_hand(img,coord,type)
    cv2.imwrite(f"./vis_result/{img_name.split('/')[-1]}",img)
    cv2.imshow('result', img)
    cv2.waitKey(-1)