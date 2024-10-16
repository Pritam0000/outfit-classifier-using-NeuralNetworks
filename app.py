import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

from classifier import build_classifier, load_trained_classifier
from autoencoder import build_autoencoder, load_trained_autoencoder
from data_loader import load_fashion_mnist, add_noise
from lime_explainer import explain_prediction, explain_all_classes

# Load data and models
@st.cache_resource
def load_data_and_models():
    (train_images, train_labels), (test_images, test_labels), class_names = load_fashion_mnist()
    
    classifier_path = 'trained_classifier.h5'
    autoencoder_path = 'trained_autoencoder.h5'
    
    if not os.path.exists(classifier_path):
        st.warning("Classifier model not found. Training a new one...")
        classifier = build_classifier((28, 28, 1), len(class_names))
        classifier.fit(train_images[..., np.newaxis], train_labels, epochs=5, validation_split=0.1)
        classifier.save(classifier_path)
    else:
        classifier = load_trained_classifier(classifier_path)
    
    if not os.path.exists(autoencoder_path):
        st.warning("Autoencoder model not found. Training a new one...")
        autoencoder, _ = build_autoencoder((28, 28, 1))
        autoencoder.fit(train_images[..., np.newaxis], train_images[..., np.newaxis], epochs=5, validation_split=0.1)
        autoencoder.save(autoencoder_path)
    else:
        autoencoder = load_trained_autoencoder(autoencoder_path)
    
    return test_images, test_labels, class_names, classifier, autoencoder

st.title('Fashion MNIST Analyzer')

# Load data and models
test_images, test_labels, class_names, classifier, autoencoder = load_data_and_models()

# Sidebar
st.sidebar.header('Settings')
image_index = st.sidebar.number_input('Select image index', min_value=0, max_value=len(test_images)-1, value=0)
noise_factor = st.sidebar.slider('Noise factor', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader('Original Image')
    fig, ax = plt.subplots()
    ax.imshow(test_images[image_index], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader('Noisy Image')
    noisy_image = add_noise(test_images[image_index], noise_factor)
    fig, ax = plt.subplots()
    ax.imshow(noisy_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader('Denoised Image')
    denoised_image = autoencoder.predict(noisy_image[np.newaxis, :, :, np.newaxis])[0]
    fig, ax = plt.subplots()
    ax.imshow(denoised_image[:, :, 0], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader('Classification')
    prediction = classifier.predict(denoised_image[np.newaxis, :, :, np.newaxis])
    predicted_class = np.argmax(prediction)
    st.write(f'Predicted class: {class_names[predicted_class]}')
    st.write(f'Actual class: {class_names[test_labels[image_index]]}')

st.subheader('LIME Explanation')
if st.button('Generate LIME Explanation'):
    with st.spinner('Generating explanation...'):
        fig = explain_prediction(classifier, denoised_image[:, :, 0], test_labels[image_index], class_names)
        st.pyplot(fig)

        st.subheader('Explanations for All Classes')
        fig = explain_all_classes(classifier, denoised_image[:, :, 0], test_labels[image_index], class_names)
        st.pyplot(fig)

st.sidebar.markdown('---')
st.sidebar.subheader('Upload Your Own Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert('L')
    
    # Resize and preprocess the image
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image).astype('float32') / 255.0  # Normalize pixel values between 0 and 1
    
    # Add batch dimension and channel dimension (to match model input shape: (1, 28, 28, 1))
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Analyze Uploaded Image'):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Original Uploaded Image')
            fig, ax = plt.subplots()
            ax.imshow(image_array[0, :, :, 0], cmap='gray')  # Display the image correctly
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader('Classification')
            prediction = classifier.predict(image_array)  # Shape already matches (1, 28, 28, 1)
            predicted_class = np.argmax(prediction)
            st.write(f'Predicted class: {class_names[predicted_class]}')

        st.subheader('LIME Explanation for Uploaded Image')
        with st.spinner('Generating explanation...'):
            fig = explain_prediction(classifier, image_array[0, :, :, 0], predicted_class, class_names)
            st.pyplot(fig)

            st.subheader('Explanations for All Classes')
            fig = explain_all_classes(classifier, image_array[0, :, :, 0], predicted_class, class_names)
            st.pyplot(fig)
