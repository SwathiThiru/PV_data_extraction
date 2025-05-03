"""
This file contains functions to enhance image processing.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import os
import cv2
import numpy as np
from PIL import Image


def enhance_image(image):
    """
    Enhances the input image by applying grayscale conversion, noise reduction,
    thresholding, and sharpening.

    Parameters:
        image (PIL.Image): Input image to be enhanced.

    Returns:
        PIL.Image: Enhanced image.
    """

    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 1. Convert to Grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Apply GaussianBlur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 3. Apply Thresholding (Binarization)
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Sharpen the Image using kernel filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(binary_img, -1, kernel)

    # Convert the grayscale sharpened image back to 3 channels (RGB)
    #sharpened_img_3channel = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2BGR)

    # Convert back to PIL Image for saving
    final_img = Image.fromarray(sharpened_img)

    return final_img


def process_images_in_folder(folder_path):
    """
        Processes and enhances all .jpg images in the specified folder.

        Parameters:
            folder_path (str): Path to the folder containing .jpg images.

        Returns:
            None : the enhanced images are saved in a subfolder named 'enhanced_images'
    """
    # Create output folder for enhanced images
    enhanced_folder = os.path.join(folder_path, "enhanced_images")
    if not os.path.exists(enhanced_folder):
        os.makedirs(enhanced_folder)

    # Loop through all .jpg files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)

            # Open the image using PIL
            image = Image.open(img_path)

            # Enhance the image
            enhanced_image = enhance_image(image)

            # Save the enhanced image in the "enhanced_images" folder
            enhanced_image_path = os.path.join(enhanced_folder, filename)
            enhanced_image.save(enhanced_image_path)
            print(f"Enhanced image saved at: {enhanced_image_path}")


if __name__ == "__main__":
    # Example folder path (replace this with your desired folder path)
    folder_path = "../../TATREvaluationDataset/solarModuleRecognition/images"

    # Process all images in the folder
    process_images_in_folder(folder_path)
