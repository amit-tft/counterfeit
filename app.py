import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

def load_frozen_model(pb_path):
    """Load a frozen TensorFlow model from a .pb file."""
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# Define constants
CLASS_NAMES = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
    'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
    'Texaco', 'Unicef', 'Vodafone', 'Yahoo'
]
LABEL_MAP = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
PB_PATH = '/model/mymodel.pb'

# Load the model
graph_def = load_frozen_model(PB_PATH)

def preprocess_image(image, target_size):
    """Load and preprocess an image for the model."""
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    image_np = np.array(image_resized)
    return np.expand_dims(image_np, axis=0), np.array(image)  # Return original RGB image for visualization

def draw_label(image, left, top, label, color, text_color, bg_color):
    """Draw a label with a background on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)  # Increase font size
    text_bbox = draw.textbbox((left, top), label, font=font)
    draw.rectangle(text_bbox, fill=bg_color)
    draw.text((left, top), label, fill=text_color, font=font)

def draw_legend(image, text, position, box_color, text_color):
    """Draw a beautified legend on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 16)
    padding = 5
    text_bbox = draw.textbbox((position[0], position[1]), text, font=font)
    top_left = (position[0], position[1] - (text_bbox[3] - text_bbox[1]) - padding)
    bottom_right = (top_left[0] + (text_bbox[2] - text_bbox[0]) + padding * 2, position[1])
    draw.rectangle([top_left, bottom_right], fill=box_color)
    draw.text((top_left[0] + padding, top_left[1]), text, fill=text_color, font=font)

def predict(image):
    # Start a TensorFlow session and load the graph
    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name="")

        # Input tensor
        input_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

        # Output tensors
        output_tensors = {
            'boxes': sess.graph.get_tensor_by_name('detection_boxes:0'),
            'scores': sess.graph.get_tensor_by_name('detection_scores:0'),
            'classes': sess.graph.get_tensor_by_name('detection_classes:0'),
            'num_detections': sess.graph.get_tensor_by_name('num_detections:0')
        }

        # Preprocess the image
        preprocessed_image, original_image = preprocess_image(image, (300, 300))

        # Run inference
        output = sess.run(output_tensors, feed_dict={input_tensor: preprocessed_image})

        # Extract outputs
        detection_boxes = output['boxes'][0]  # [N, 4]
        detection_scores = output['scores'][0]  # [N]
        detection_classes = output['classes'][0]  # [N]
        num_detections = int(output['num_detections'][0])  # []

        # Convert original_image to PIL.Image for drawing
        original_image = Image.fromarray(original_image)

        # Visualize the detection results on the original image
        draw = ImageDraw.Draw(original_image)
        width, height = original_image.size
        for i in range(num_detections):
            if detection_scores[i] > 0.5:  # Consider detections with score > 0.5
                box = detection_boxes[i]
                class_id = int(detection_classes[i])
                score = detection_scores[i]

                # Convert box coordinates from normalized [0, 1] to pixel values
                ymin, xmin, ymax, xmax = box
                left, right, top, bottom = (xmin * width, xmax * width, ymin * height, ymax * height)

                # Choose color based on score
                if score < 0.9:
                    box_color = (255, 0, 0)  # Red color for bounding box
                    text_color = (255, 255, 255)  # White color for text
                    bg_color = (255, 0, 0)  # Red background for label
                else:
                    box_color = (0, 255, 0)  # Green color for bounding box
                    text_color = (0, 0, 0)  # Black color for text
                    bg_color = (0, 255, 255)  # Yellow background for label

                # Draw bounding box with increased width
                draw.rectangle([left, top, right, bottom], outline=box_color, width=4)

                # Use the class name from the label map
                class_name = LABEL_MAP.get(class_id, f'Class {class_id}')
                label = f'{class_name}: {score:.2f}'

                # Draw the label on the image
                draw_label(original_image, left, top, label, box_color, text_color, bg_color)

        # Draw legends
        draw_legend(original_image, 'High Score (>90%): Product is likely original', (10, height - 50), (0, 255, 0), (0, 0, 0))
        draw_legend(original_image, 'Low Score (<90%): Product may be counterfeit', (10, height - 20), (255, 0, 0), (255, 255, 255))

        return original_image

# Streamlit app
st.title("Logo Counterfeit Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Click to detect counterfeit"):
        result = predict(image)
        st.image(result, caption='Processed Image', use_column_width=True)
