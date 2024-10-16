import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt

def explain_prediction(model, image, actual_label, class_names):
    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
    
    # Ensure the image is in grayscale
    image_rgb = gray2rgb(image)  # Convert grayscale to RGB for LIME

    def predict_fn(img):
        return model.predict(rgb2gray(img)[:, :, np.newaxis])  # Ensure correct shape for prediction


    
    explanation = explainer.explain_instance(image_rgb,
                                             classifier_fn=predict_fn,
                                             top_labels=10,
                                             hide_color=0,
                                             num_samples=1000,
                                             segmentation_fn=segmenter)
    
    temp, mask = explanation.get_image_and_mask(actual_label, positive_only=True, num_features=10, hide_rest=False, min_weight=0.01)
    fig, ax = plt.subplots()
    ax.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax.set_title(f'Positive Regions for {class_names[actual_label]}')
    plt.close(fig)
    return fig

def explain_all_classes(model, image, actual_label, class_names):
    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
    
    image_rgb = gray2rgb(image)
    
    def predict_fn(img):
        return model.predict(rgb2gray(img)[:, :, np.newaxis])
    
    explanation = explainer.explain_instance(image_rgb,
                                             classifier_fn=predict_fn,
                                             top_labels=10,
                                             hide_color=0,
                                             num_samples=1000,
                                             segmentation_fn=segmenter)
    
    fig, m_axs = plt.subplots(2, 5, figsize=(12, 6))
    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False, min_weight=0.01)
        c_ax.imshow(label2rgb(mask, image_rgb, bg_label=0), interpolation='nearest')
        c_ax.set_title(f'Positive for {class_names[i]}\nActual {class_names[actual_label]}')
        c_ax.axis('off')
    plt.tight_layout()
    plt.close(fig)
    return fig