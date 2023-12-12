import numpy as np
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def predict_class(user_image, features_file_path, threshold=0.6):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

    # Load stored class-wise features
    with open(features_file_path, 'r') as f:
        class_features = json.load(f)

    # Load the trained VGG16 model for similarity calculation (if saved)
    # Load the model in your local environment as you did in Google Colab

    # Convert OpenCV image to array and preprocess
    user_img = cv2.resize(user_image, (224, 224))
    user_img = preprocess_input(user_img)
    user_features = model.predict(np.expand_dims(user_img, axis=0))

    # Calculate similarity with stored class-wise features
    highest_similarity_score = 0.0
    most_similar_class = None

    for class_label, class_images_features in class_features.items():
        for img_features in class_images_features:
            img_features = np.array(img_features)
            img_features = img_features.reshape(1, -1)  # Flatten the feature array
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity(user_features.reshape(1, -1), img_features)[0][0]
            
            if similarity_score > threshold and similarity_score > highest_similarity_score:
                highest_similarity_score = similarity_score
                most_similar_class = class_label

    if most_similar_class is not None:
        return most_similar_class
    else:
        return "No class prediction found."

# Example usage

features_file_path = '/Users/sivagar/Documents/projects/farrer_hos/projects/document_classification/classwise_features.json'
image ="/Users/sivagar/Documents/projects/farrer_hos/projects/document_classification/ilovepdf_pages-to-jpg/no_error_page-0014.jpg"
user_image = img = cv2.imread(image)
predicted_class = predict_class(user_image, features_file_path)
print("Predicted class:", predicted_class)
