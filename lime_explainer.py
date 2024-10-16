import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

def explain_prediction(classifier, image, label, class_names):
    # Ensure the input is reshaped to (28, 28) for LIME visualization
    if image.ndim == 3 and image.shape[-1] == 1:
        image_2d = image.reshape(28, 28)  # Drop the channel dimension
    else:
        image_2d = image  # Already in (28, 28)

    # Create LIME image explainer
    explainer = lime_image.LimeImageExplainer()

    # Wrapper to match classifier's input/output format
    def classifier_predict(images):
        # LIME provides images as (N, H, W, C=3); convert them to (N, 28, 28, 1)
        images = np.array([rgb2gray(img).reshape(28, 28, 1) for img in images])
        return classifier.predict(images)

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image_2d,
        classifier_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Extract the mask and boundaries for visualization
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Plot the explanation
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp, mask), cmap='gray')
    ax.set_title(f"LIME Explanation for Class: {class_names[label]}")
    ax.axis('off')

    plt.close(fig)  # Close the figure to prevent duplicate rendering in Streamlit
    return fig
