# python
import os
import numpy as np
import tensorflow as tf
from pprint import pprint

# 1) 加载模型
MODEL_PATH = "cloud_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# 2) 与训练保持一致的配置
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["altocumulus", "altostratus", "cirrocumulus",
               "cirrostratus", "cirrus", "cumulonimbus", "cumulus",
               "nimbostratus", "stratocumulus", "stratus"]

_VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 3) 与训练一致的解码与预处理
@tf.function
def _decode_by_ext(path: tf.Tensor, img_bytes: tf.Tensor) -> tf.Tensor:
    lower = tf.strings.lower(path)
    is_jpeg = tf.strings.regex_full_match(lower, r".*\.jpe?g$")
    is_png  = tf.strings.regex_full_match(lower, r".*\.png$")
    is_bmp  = tf.strings.regex_full_match(lower, r".*\.bmp$")

    def _decode_jpeg():
        return tf.image.decode_jpeg(img_bytes, channels=3)
    def _decode_png():
        return tf.image.decode_png(img_bytes, channels=3)
    def _decode_bmp():
        return tf.image.decode_bmp(img_bytes, channels=3)

    img = tf.case(
        pred_fn_pairs=[
            (is_jpeg, _decode_jpeg),
            (is_png,  _decode_png),
            (is_bmp,  _decode_bmp),
        ],
        default=_decode_jpeg,
        exclusive=False
    )
    return img

def load_image_for_infer(path: str) -> np.ndarray:
    path_t = tf.convert_to_tensor(path)
    img_bytes = tf.io.read_file(path_t)
    img = _decode_by_ext(path_t, img_bytes)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # [1, H, W, C]
    return img.numpy()

# 4) 预测（同时兼容多分类 softmax 与二分类 sigmoid）
def predict_image(path: str):
    x = load_image_for_infer(path)
    logits = model.predict(x, verbose=0)

    # 转为概率
    logits_tf = tf.convert_to_tensor(logits)
    if logits_tf.shape[-1] == 1:
        # 二分类（sigmoid）
        p1 = tf.sigmoid(logits_tf)[0, 0].numpy().item()
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        probs = tf.nn.softmax(logits_tf, axis=-1)[0].numpy()

    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx] if 0 <= top_idx < len(CLASS_NAMES) else str(top_idx)
    top_prob = float(probs[top_idx])
    return top_class, top_prob, probs

# 5) 批量预测（遍历文件夹）
def predict_folder(folder: str):
    results = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(_VALID_EXTS):
            p = os.path.join(folder, fn)
            cls, score, _ = predict_image(p)
            results.append((fn, cls, score))
    return results

if __name__ == "__main__":
    # 单张图片
    cls, score, probs = predict_image("../atmo2_20250826_111547.jpg")
    print(f"预测类别: {cls}, 置信度: {score:.4f}")
    pprint(probs)
    # 批量
    # for name, cls, score in predict_folder("/path/to/dir"):
    #     print(f"{name}\t{cls}\t{score:.4f}")