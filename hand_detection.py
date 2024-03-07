from mediapipe import solutions
from mediapipe.tasks import python
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
mp_hands = mp.solutions.hands
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


# 在图像上画手部点
def draw_hand(img, coord):

    for i in range(len(bone_links)):
        bone_link = bone_links[i]
        for j in bone_link:
            cv2.circle(img, tuple(coord[j]), 1, joint_colors[i], -1)
        for j, nj in zip(bone_link[:-1], bone_link[1:]):
            cv2.line(img, tuple(coord[j]), tuple(coord[nj]), joint_colors[i])  


if __name__ == "__main__":

    save_res = True # 存可视化图片

    # mediapipe 进行手部关键点检测
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.1,
    )

    # 加载测试图片和测试
    img = cv2.imread('./test_image/open.jpeg')
    h, w = img.shape[:2]
    # mediapipe推理
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hand_result = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for joint_idx, xyz in enumerate(hand_landmarks.landmark):
                hand_result[joint_idx][0] = xyz.x * w
                hand_result[joint_idx][1] = xyz.y * h
                hand_result[joint_idx][2] = xyz.z * w
    # 得到骨骼点数据
    coord = np.array(hand_result[:, :2], dtype=np.int32)
    draw_hand(img, coord)
    if save_res:
        cv2.imwrite('./vis_result/res_detection.png',img)
    
    cv2.imshow('result', img)
    cv2.waitKey(0)