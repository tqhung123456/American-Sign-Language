import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

'''function to load folder into arrays and
then it returns that same array'''


def loadImages(path):
    image_files = sorted([os.path.join(path, folder, folder + "1.jpg")
                         for folder in os.listdir(path)])
    return image_files

# Display one image


def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Display two images


def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# defining global variable path
path = "D:\\asl_data\\asl_alphabet_train"

n = 1

img_path = loadImages(path)

res_img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in img_path]

original = res_img[n]


# Gaussian blur
no_noise = []
for i in range(len(res_img)):
    blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
    no_noise.append(blur)


image = no_noise[n]
display(original, image, 'Original', 'Blured')

# Brightness increasing
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

image = increase_brightness(image)
display(original, image, 'Original', 'Brighter')

# Segmentation
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.equalizeHist(gray)
# display_one(gray)
ret, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Displaying segmented images
# display(original, thresh, 'Original', 'Segmented')

# Further noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(
    dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Displaying segmented back ground
# display(original, sure_bg, 'Original', 'Segmented Background')

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(image, markers)
markers[markers == 1] = 0
markers[markers > 1] = 255
image[markers == -1] = [255, 0, 0]

# Displaying markers on the image
display(image, markers, 'Original', 'Marked')

# Now apply the mask we created on the initial image
final_img = cv2.bitwise_and(image, image, mask=markers.astype(np.uint8))

# cv2.imread reads the image as BGR, but matplotlib uses RGB
# BGR to RGB so we can plot the image with accurate colors
b, g, r = cv2.split(final_img)
final_img = cv2.merge([r, g, b])

# Plot the final result
display_one(final_img, "Croped")
