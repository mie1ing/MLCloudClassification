import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.optimizers import Adam

# 基本配置
image_size = (128, 128)
batch_size = 16
train_dir = 'training_data'  # 确保目录存在，子目录为类别名称

# 1) 解析类别名称（来自子目录）
class_names = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
if not class_names:
    raise RuntimeError(f'在目录 "{train_dir}" 下未找到任何类别子目录，请确认数据集结构。')

num_classes = len(class_names)

# 2) 构建一个 label 查找表（无需 PIL）
keys_tensor = tf.constant(class_names)
vals_tensor = tf.constant(list(range(num_classes)), dtype=tf.int32)
label_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1,
)

# 允许的图片扩展名（全部转为小写比较）
_valid_ext_regex = r'.*\.(jpg|jpeg|png|bmp)$'

def has_valid_ext(path: tf.Tensor) -> tf.Tensor:
    lower = tf.strings.lower(path)
    return tf.strings.regex_full_match(lower, _valid_ext_regex)

# 3) tf.data 读取与预处理：按扩展名使用专用解码器，避免 decode_image 的误判
def load_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    lower = tf.strings.lower(path)

    is_jpeg = tf.strings.regex_full_match(lower, r'.*\.jpe?g$')
    is_png  = tf.strings.regex_full_match(lower, r'.*\.png$')
    is_bmp  = tf.strings.regex_full_match(lower, r'.*\.bmp$')

    def _decode_jpeg():
        return tf.image.decode_jpeg(img_bytes, channels=3)
    def _decode_png():
        return tf.image.decode_png(img_bytes, channels=3)
    def _decode_bmp():
        return tf.image.decode_bmp(img_bytes, channels=3)

    # 根据扩展名选择解码器（已通过 has_valid_ext 过滤，正常不会走到默认分支）
    img = tf.case(
        pred_fn_pairs=[
            (is_jpeg, _decode_jpeg),
            (is_png,  _decode_png),
            (is_bmp,  _decode_bmp),
        ],
        default=_decode_jpeg,  # 兜底（基本不会触发）
        exclusive=False
    )

    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0  # 归一化到 [0,1]
    return img

def get_label(path: tf.Tensor) -> tf.Tensor:
    # 路径结构: .../training_data/<class_name>/<filename>
    parts = tf.strings.split(path, os.sep)
    class_name = parts[-2]
    label = label_table.lookup(class_name)
    return label  # 稀疏整型标签

def load_and_label(path: tf.Tensor):
    return load_image(path), get_label(path)

# 4) 构建数据集：先过滤扩展名，再映射，最后忽略坏样本
files_pattern = os.path.join(train_dir, '*', '*')
dataset = tf.data.Dataset.list_files(files_pattern, shuffle=True, seed=42)

# 只保留合法图片扩展名，屏蔽掉 .DS_Store 等非图片文件
dataset = dataset.filter(has_valid_ext)

# 读取与标注
dataset = dataset.map(load_and_label, num_parallel_calls=tf.data.AUTOTUNE)

# 忽略无法解析的样本
dataset = dataset.ignore_errors()

# 打乱、批处理、预取（只 batch 一次）
dataset = dataset.shuffle(1000, seed=42)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# 5) 构建模型（将数据增强放到模型里，训练时生效，推理时自动关闭）
model = Sequential([
    Input(shape=(image_size[0], image_size[1], 3)),
    # 在线数据增强
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.2),
    RandomTranslation(0.1, 0.1),

    # 卷积主干
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6) 开始训练：直接使用已 batch 的 dataset，让 Keras 自动确定步数
model.fit(dataset, epochs=10)

# 7) 保存模型
model.save('cloud_classifier.keras')