import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from classifier import build_classifier, load_trained_classifier
from autoencoder import build_autoencoder, load_trained_autoencoder
from data_loader import load_fashion_mnist, add_noise
from lime_explainer import explain_prediction

# Load Data
@st.cache_resource
def load_data():
    (train_images, train_labels), (test_images, test_labels), class_names = load_fashion_mnist()
    return train_images, train_labels, test_images, test_labels, class_names

train_images, train_labels, test_images, test_labels, class_names = load_data()

# Paths to Saved Models
CLASSIFIER_PATH = 'trained_classifier.h5'
AUTOENCODER_PATH = 'trained_autoencoder.h5'

# Function to Load or Train Models
def get_models(retrain=False, epochs=5):
    if retrain or not os.path.exists(CLASSIFIER_PATH):
        st.warning("Training classifier model...")
        classifier = build_classifier((28, 28, 1), len(class_names))
        classifier.fit(train_images[..., np.newaxis], train_labels, epochs=epochs, validation_split=0.1)
        classifier.save(CLASSIFIER_PATH)
    else:
        classifier = load_trained_classifier(CLASSIFIER_PATH)

    if retrain or not os.path.exists(AUTOENCODER_PATH):
        st.warning("Training autoencoder model...")
        autoencoder, _ = build_autoencoder((28, 28, 1))
        autoencoder.fit(train_images[..., np.newaxis], train_images[..., np.newaxis], 
                        epochs=epochs, validation_split=0.1)
        autoencoder.save(AUTOENCODER_PATH)
    else:
        autoencoder = load_trained_autoencoder(AUTOENCODER_PATH)

    return classifier, autoencoder

# Sidebar Configuration
st.sidebar.header('Settings')
image_index = st.sidebar.number_input('Select Image Index', min_value=0, max_value=len(test_images) - 1, value=0)
noise_factor = st.sidebar.slider('Noise Factor', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Retrain Option
st.sidebar.subheader('Retrain Models')
epochs = st.sidebar.number_input('Select Number of Epochs', min_value=1, max_value=50, value=5)
if st.sidebar.button('Retrain Classifier & Autoencoder'):
    classifier, autoencoder = get_models(retrain=True, epochs=epochs)
    st.sidebar.success('Models retrained successfully!')
else:
    classifier, autoencoder = get_models()

# Main App
st.title('Fashion MNIST Analyzer')

col1, col2 = st.columns(2)

# Display Original Image
with col1:
    st.subheader('Original Image')
    fig, ax = plt.subplots()
    ax.imshow(test_images[image_index], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Display Noisy Image
    st.subheader('Noisy Image')
    noisy_image = add_noise(test_images[image_index], noise_factor)
    fig, ax = plt.subplots()
    ax.imshow(noisy_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

# Denoised and Classification
with col2:
    st.subheader('Denoised Image')
    denoised_image = autoencoder.predict(noisy_image[np.newaxis, :, :, np.newaxis])[0]
    fig, ax = plt.subplots()
    ax.imshow(denoised_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    st.pyplot(fig)

    # Predict on Original Image
    st.subheader('Classification')
    prediction = classifier.predict(test_images[image_index][np.newaxis, :, :, np.newaxis])
    predicted_class = np.argmax(prediction)
    st.write(f'Predicted Class: {class_names[predicted_class]}')
    st.write(f'Actual Class: {class_names[test_labels[image_index]]}')

# LIME Explanation
st.subheader('LIME Explanation')
if st.button('Generate Explanation'):
    with st.spinner('Generating explanation...'):
        try:
            fig = explain_prediction(
                classifier, 
                test_images[image_index].reshape(28, 28, 1), 
                test_labels[image_index], 
                class_names
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in generating explanation: {str(e)}")

# Upload Image Option
st.sidebar.subheader('Upload Your Own Image')
uploaded_file = st.sidebar.file_uploader("Upload an image (28x28 grayscale)...", type=["jpg", "png"])

if uploaded_file:
    try:
        # Load and Process Uploaded Image
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

        if st.sidebar.button('Analyze Uploaded Image'):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Uploaded Image')
                fig, ax = plt.subplots()
                ax.imshow(image_array[0, :, :, 0], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)

            with col2:
                st.subheader('Classification')
                prediction = classifier.predict(image_array)
                predicted_class = np.argmax(prediction)
                st.write(f'Predicted Class: {class_names[predicted_class]}')

    except Exception as e:
        st.sidebar.error(f"Error processing uploaded image: {str(e)}")
