import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_image = mpimg.imread('cutouts/bbox-example-image.jpg')
test_image = mpimg.imread('cutouts/temp-matching-example-2.jpg')
templist = ['cutouts/cutout1.jpg', 'cutouts/cutout2.jpg', 'cutouts/cutout3.jpg',
            'cutouts/cutout4.jpg', 'cutouts/cutout5.jpg', 'cutouts/cutout6.jpg']

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes 
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    # Read in templates one by one
    # Use cv2.matchTemplate() to search the image
    #     using whichever of the OpenCV search methods you prefer
    # Use cv2.minMaxLoc() to extract the location of the best match
    # Determine bounding box corners for the match
    for temp in template_list:
        temp_image = cv2.imread(temp)
        w, h = temp_image.shape[1], temp_image.shape[0]
        #Used squared difference method
        result = cv2.matchTemplate(img, temp_image, eval("cv2.TM_SQDIFF"))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
    # Return the list of bounding boxes
    return bbox_list

bboxes = find_matches(train_image, templist)
result_train = draw_boxes(train_image, bboxes)
bboxes = find_matches(test_image, templist)
result_test = draw_boxes(test_image, bboxes)
plt.imshow(result_train)
plt.show()
plt.imshow(result_test)
plt.show()
