import cv2
import pandas as pd
import matplotlib.pyplot as plt
import imutils

def get_pred_boxes(contours, image):
    height, width = image.shape[0], image.shape[1]
    
    pred_boxes = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if ( (h*w > 0.015*height*width) and (y<0.5*height) and (0.25*width< x <0.75*width)):
            pred_boxes.append([x, y, x+w, y+h])
    
    pred_boxes = removeDuplicateBoxes(pred_boxes)
    return pred_boxes


def get_gt_boxes(i):
    digitStruct = pd.read_json('data/train/digitStruct.json')
    digitStruct.set_index('filename', inplace=True)
    gt_boxes = []
    boxes = digitStruct.loc[f'{str(i)}.png', 'boxes']
    for box in boxes:
        x = int(box['left'])
        y = int(box['top'])
        w = int(box['width'])
        h = int(box['height'])
        #label = int(box['label'])
        gt_boxes.append([x, y, x+w, y+h])
    
    return gt_boxes


def get_gt_labels(i):
    digitStruct = pd.read_json('data/train/digitStruct.json')
    digitStruct.set_index('filename', inplace=True)
    boxes = digitStruct.loc[f'{str(i)}.png', 'boxes']
    labels = []
    for box in boxes:
        label = int(box['label'])%10
        labels.append(label)
    
    return labels


def get_match_boxes(pred_boxes, gt_boxes, gt_labels):
    match_boxes = []
    for box in pred_boxes:
        for i, gt_box in enumerate(gt_boxes):
            iou = bb_intersection_over_union(box, gt_box)
            if iou > 0.4:
                match_boxes.append({'box': box, 'iou':iou, 'label': gt_labels[i]})

    if len(match_boxes) > len(gt_boxes):
        match_boxes = sorted(match_boxes, key=lambda c: c['iou'], reverse=True)
        match_boxes = match_boxes[:len(gt_boxes)]
    #match_boxes = [e['box'] for e in match_boxes]
    return match_boxes


def removeDuplicateBoxes(pred_boxes):
    # Corner Case: Image has only 1 digit and 1 contour predicted 
    if(len(pred_boxes)==1):
        return pred_boxes
    
    filtered_boxes = []
    #pred_contours = [list(x) for x in set([tuple(L) for L in pred_contours])]
    i=0
    while(i<len(pred_boxes)):
        filtered_boxes.append(pred_boxes[i])
        # Corner Case: Last two boxes are not duplicates
        if i == len(pred_boxes)-1:
            break    
        j=i+1
        while(j<len(pred_boxes)):
            iou = bb_intersection_over_union(pred_boxes[i], pred_boxes[j])
            if iou>0.8:
                pred_boxes.pop(j)  
            j+=1
        i+=1     
    return filtered_boxes


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def crop_boxes(image, boxes):
    digit_images = []
    for box in boxes:
        [x1, y1, x2, y2] = box
        digit = image[y1:y2, x1:x2]
        digit_images.append(digit)
    return digit_images

def get_cropped_digits(image_no):
    # Image Processing Pipeline
    image = cv2.imread(f'data/train/{image_no}.png')  
#     plt.figure()
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
    bfilter = cv2.bilateralFilter(image,11,17,17)
    gray = cv2.cvtColor(bfilter,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,30,150)
    keypoints = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    pred_boxes = get_pred_boxes(sorted_contours, image)
    gt_boxes = get_gt_boxes(image_no)
    gt_labels = get_gt_labels(image_no)

    match_boxes = get_match_boxes(pred_boxes, gt_boxes, gt_labels)
    labels = [e['label'] for e in match_boxes]
    match_boxes = [e['box'] for e in match_boxes]
    
    
    digit_images = crop_boxes(gray, match_boxes)
    return digit_images, labels


def get_cropped_template(temp_no):
    # Image Processing Pipeline
    template = cv2.imread(f'../Templates/NEW_NUMBERS/{temp_no}.jpg')  
    gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray,30,150)
    keypoints = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    digit_contour = max(contours, key=cv2.contourArea)
    # Box around tempalte digit
    [x, y, w, h] = cv2.boundingRect(digit_contour)
    box = [[x, y, x+w, y+h]]
    temp_image = crop_boxes(thresh, box)[0] # function returns a list
    return temp_image


