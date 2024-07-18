import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

def load_labels(label_path):
    with open(label_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def plot_predictions(image_path, model, class_names, top_k=5):
    probs, classes = predict(image_path, model, top_k)
    im = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_title('Input Image')
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([class_names[str(cls)] for cls in classes])
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predictions')
    plt.tight_layout()
    plt.show()
