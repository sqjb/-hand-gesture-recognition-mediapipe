import numpy as np
import tensorflow as tf
import copy
import itertools
from sklearn.model_selection import train_test_split
import cv2

RANDOM_SEED = 42

# 训练数据
dataset = './hand_gesture/hand_gesture.csv'
# 存储的模型结构
model_save_path = './hand_gesture/hand_gesture.hdf5'

# 识别的手势个数和类型
NUM_CLASSES = 4
gesture_name = ['open','close','pointer','ok']



if __name__ == '__main__':

    # 构造模型
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
    # 输出模型结构
    model.summary() 

    # 训练
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    
    # 配置模型保持的逻辑 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False)
    # 训练策略
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    
    # 配置训练的损失函数
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 开始训练
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )
    # 输出loss信息  
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
