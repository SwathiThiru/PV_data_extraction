"""
This file augments an image to increase training dataset size and
to reduce overfitting and then it extracted the OCR data from the enhanced images.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
"""

import os
import cv2
import numpy as np
import argparse
import json
from PIL import Image
from PIL import ImageFilter
import pytesseract
from skimage.filters import threshold_sauvola
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, load_img


def get_args():
    parser = argparse.ArgumentParser(
        description="Apply brightness, channel shift, and random mask augmentations to a single image.")
    parser.add_argument('--in_path', required=True, help="Path to the input image")
    return parser.parse_args()

def enhance_image(image):
    """This function enhances a given image by applying several preprocessing steps,
    including converting to grayscale, reducing noise with Gaussian blur, binarization,
    and sharpening. It returns the processed image as a PIL Image object.

    Parameters:
        image: a PIL Image object representing the input image to be enhanced.

    Returns:
        final_img: a PIL Image object representing the enhanced, sharpened, and binarized
                   version of the input image.

    Example:
        #>>> img = Image.open("sample_image.jpg")
        #>>> enhanced_img = enhance_image(img)
        #>>> enhanced_img.show()  # Opens the enhanced image for display
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

    # 5. Rescale the image (if too small, you can upscale it)
    #resized_img = cv2.resize(sharpened_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL Image for saving
    final_img = Image.fromarray(sharpened_img)

    return final_img

def main():
    args = get_args()
    pytesseract.pytesseract.tesseract_cmd = r'S:\23502\2\216_PVM\Aktuell\01_Orga\21631_MAS_TeamModulbewertung\03_Arbeitsordner\Swathi_Thiruvengadam\Tesseract-OCR\tesseract.exe'

    with Image.open(args.in_path).convert('RGB') as img:
        # improve the input image
        # improve the input image
        img = img.filter(ImageFilter.SHARPEN)
        img = enhance_image(img)
        # Use Tesseract to do OCR on the image
        custom_config = r'-c tessedit_char_blacklist=`!@|[]{}€£¥§©®™¢γδζηθικλμξπρΣσςτψωΩ|'
        extracted_data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

        words_data = [
            {"bbox": [extracted_data['left'][i], extracted_data['top'][i],
                      extracted_data['left'][i] + extracted_data['width'][i],
                      extracted_data['top'][i] + extracted_data['height'][i]],
             "text": extracted_data['text'][i].strip(),
             "line_num": extracted_data['line_num'][i],
             "block_num": extracted_data['block_num'][i]}
            for i in range(len(extracted_data['text'])) if extracted_data['text'][i].strip()
        ]

        directory = os.path.dirname(args.in_path)
        base_name = os.path.splitext(os.path.basename(args.in_path))[0]
        output_path = os.path.join(directory, base_name + '_words.json')

        # Write the extracted word data to a JSON file
        with open(output_path, 'w') as json_file:
            json.dump(words_data, json_file)
            print("OCR file created at path : ", output_path)



if __name__ == "__main__":
    main()
