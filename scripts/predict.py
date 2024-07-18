import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import sys
sys.path.insert(0, '..')
from utilities import process_image,  plot_predictions

def load_labels(label_path):
    with open(label_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    img_array = np.asarray(im)
    processed_image = process_image(img_array)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_k_probs, top_k_indices = tf.math.top_k(predictions, k=top_k)
    top_k_probs = top_k_probs.numpy().flatten()
    top_k_indices = top_k_indices.numpy().flatten()
    top_k_classes = [str(index+1) for index in top_k_indices]
    return top_k_probs, top_k_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('model_path', type=str, help="Path to the saved model")
    parser.add_argument('--top_k', type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument('--category_names', type=str, default=None, help="Path to a JSON file mapping labels to flower names")
    return parser.parse_args()

def main():
    args = parse_args()
    model = load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Lambda})

    top_indices, top_probs = predict(args.image_path,model,args.top_k)
    if args.category_names:
        class_names = load_labels(args.category_names)
        top_classes = [class_names[str(i)] for i in top_indices]
    else:
        top_classes = top_indices
    print("Top K Predictions:")
    for i in range(args.top_k):
        print(f"{i+1}: {top_classes[i]} with probability {top_probs[i]:.4f}")

if __name__ == "__main__":
    main()
