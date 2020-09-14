import os
import argparse
import csv
import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import re
import codecs
from pdf2image import convert_from_path, convert_from_bytes

# from crop import crop

def preprocess_for_ocr(img, enhance=1):
    """
    @param img: image to which the pre-processing steps being applied
    """
    if enhance > 1:
        img = Image.fromarray(img)

        contrast = ImageEnhance.Contrast(img)

        img = contrast.enhance(enhance)

        img = np.asarray(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img = cv2.GaussianBlur(img, (5,5), 0)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def ocr(img, oem=1, psm=3):
    """
    @param img: The image to be OCR'd
    @param oem: for specifying the type of Tesseract engine( default=1 for LSTM OCR Engine)
    """
    config = ('-l ara --oem {oem} --psm {psm}'.format(oem=oem,psm=psm))
    # config = ('-l eng --tessdata-dir "/usr/share/tesseract-ocr/tessdata" --oem {oem} -- psm {psm}'.format(oem=oem,psm=psm))


    img = Image.fromarray(img)
    text = pytesseract.image_to_string(img, config=config)
    return text


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = ap.parse_args()
    file = args.image

    output_file = codecs.open(r"text.txt", "w", encoding='utf8')


    if file.endswith(".pdf"):
        image = convert_from_bytes(open(file, 'rb').read())
        for page in image:
            page = np.asarray(page)
            page = preprocess_for_ocr(page)
            text = ocr(page)
            text = text.replace('\n', '')
            output_file.write(text)
    else:
        image = cv2.imread(file)
        image = preprocess_for_ocr(image)
        text = ocr(image)
        text = text.replace('\n', '')
        output_file.write(text)




    # file.close()




