# from PIL import Image
import argparse
import time
import os
import cv2
from crop import crop
from detect_table_class import NutritionTableDetector
from nutrient_list import *
from process import *
from regex import *
from spacial_map import *
from text_detection import text_detection, load_text_model


def load_model():
    """
    load trained weights for the model
    """
    global obj
    obj = NutritionTableDetector()
    print("Weights Loaded!")


coordinates = []
cntlb = 0


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global coordinates, cntlb

    if event == cv2.EVENT_LBUTTONDOWN and cntlb < 2:
        cntlb += 1
    coordinates.append(x * 2)
    coordinates.append(y * 2)


cv2.destroyAllWindows()


def detect(img_path, debug):
    counter = 0
    flag = False

    image = cv2.imread(img_path)
    boxes, scores, classes, num = obj.get_classification(image)
    # print(scores, classes, boxes, num)
    # Get the dimensions of the image
    width = image.shape[1]
    height = image.shape[0]

    """
    @param img_path: Pathto the image for which labels to be extracted
    """

    # Start the time
    start_time = time.time()
    # Make the table detector class and predict the score

    if debug:
        time_taken = time.time() - start_time
        print("Time taken to detect the table: %.5fs" % time_taken)

    # Select the bounding box with most confident output
    ymin = boxes[0][counter][0] * height
    xmin = boxes[0][counter][1] * width
    ymax = boxes[0][counter][2] * height
    xmax = boxes[0][counter][3] * width
    # print(xmin, ymin, xmax, ymax, scores[0][0])
    coords = (xmin, ymin, xmax, ymax)

    # if counter == 3:
    #     coords = my_coords
    # Crop the image with the given bounding box
    cropped_image = crop(image, coords, "./data/result/output.jpg", 0, True)
    # Apply several filters to the image for better results in OCR
    cropped_image = preprocess_for_ocr(cropped_image, 3)

    if debug:
        cv2.imwrite('./data/result/output-opt.png', cropped_image)

    # detecting the text
    text_blob_list = text_detection(cropped_image)
    if debug:
        time_taken = time.time() - start_time
        print("Time Taken to detect bounding boxes for text: %.5fs" % time_taken)
        # print(text_blob_list)

    text_location_list = []  # store all the metadata of every text box
    nutrient_dict = {}  # Dictionary to store nutrient labels and their values

    img_counter = 0
    # Apply OCR to to blobs and save data in organized dict
    for blob_cord__ in text_blob_list:
        blob_cord = blob_cord__
        if blob_cord__[0] == 0:
            myList = list(blob_cord__)
            myList[0] += 10
            blob_cord = tuple(myList)

        if debug:
            img_counter += 1
            word_image = crop(cropped_image, blob_cord, "./data/result/{}.jpg".format(img_counter), 0.005, True)
            print(blob_cord, img_counter)
        else:
            word_image = crop(cropped_image, blob_cord, "./", 0.005, False)
        # word_image = preprocess_for_ocr(word_image)
        text = ocr(word_image, 1, 7)

        if debug:
            print(text)

        if text:
            center_x = (blob_cord[0] + blob_cord[2]) / 2
            center_y = (blob_cord[1] + blob_cord[3]) / 2
            box_center = (center_x, center_y)

            new_location = {
                'bbox': blob_cord,
                'text': text,
                'box_center': box_center,
                'string_type': string_type(text)
            }
            text_location_list.append(new_location)

    # Spatial algorithm that maps all boxes according to their location and append the string
    for text_dict in text_location_list:
        if (text_dict['string_type'] == 2):
            for text_dict_test in text_location_list:
                if position_definer(text_dict['box_center'][1], text_dict_test['bbox'][1],
                                    text_dict_test['bbox'][3]) and text_dict_test['string_type'] == 1:
                    text_dict['text'] = text_dict['text'].__add__(' ' + text_dict_test['text'])
                    text_dict['string_type'] = 0

    fuzdict = make_fuzdict('data/nutrients.txt')

    # Add the nutritional label and its value to the nutrient_dict
    for text_dict in text_location_list:
        if (text_dict['string_type'] == 0):
            text = clean_string(text_dict['text'])
            # print(text)
            # print(text_dict['text'])
            #             if check_for_label(text, make_list('data/nutrients.txt')):
            if fuz_check_for_label(text, fuzdict, debug):
                label_name, label_value = get_fuz_label_from_string(text, fuzdict, debug)
                nutrient_dict[label_name] = separate_unit(label_value)

    if debug:
        time_taken = time.time() - start_time
        print("Total Time Taken: %.5fs" % time_taken)

    return nutrient_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    ap.add_argument("-d", "--debug", action='store_true', help="print some debug info")
    # ap.add_argument("-l", "--language", required=True, help="language to detect")

    args = ap.parse_args()

    load_model()
    load_text_model()

    path = args.image

    result = detect(path, False)


if __name__ == '__main__':
    main()
