import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


def cropper(img):
    return img[72:132, 0:320]  # Crop y1:y2, x1:x2


def resize(img, dst_x, dst_y):
    rx = float(dst_x) / img.shape[1]
    ry = float(dst_y) / img.shape[0]
    return cv2.resize(img, (dst_x, dst_y))


def noise_gaus(image, prob):
    noise = np.zeros(image.shape, np.uint8)
    m = (0, 0, 0)
    s = (255 * prob, 255 * prob, 255 * prob)
    cv2.randn(noise, mean=m, stddev=s)
    return noise + image


def equalize(img):
    for i in range(3):
        img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    return img


def blur_gaus(img):
    kernel = np.ones((3, 3), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)


def normalize(img):
    return cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)


def sho(img, word=""):
    if word != "": print(word)
    plt.imshow(img)
    plt.show()


def read_and_process_image(path: str, show=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return process_image(img, show)


def process_image(img, show=False):
    if show: sho(img, "Original")
    img = cropper(img)
    if show: sho(img, "Cropped")
    img = resize(img, 128, 32)
    if show: sho(img, "Resized")
    img = noise_gaus(img, 0.1)
    if show: sho(img, "Gaussian Noised")
    img = equalize(img)
    if show: sho(img, "Histogram Equalized")
    img = blur_gaus(img)
    if show: sho(img, "Gaussian Blurred")
    img = normalize(img)
    if show: sho(img, "Normalized")
    return img


def flip(img):
    return cv2.flip(img, 1)


def preprocess_and_save(csv_path: str):
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    data_path = 'data/'
    save_path = data_path + "IMG_Augmented/"
    steering_correction = 0.2
    steering_angles = []
    new_file_names = []

    for line in lines:
        steering_center = float(line[3])
        # there are so many low steers
        # only 1/3 of them mayst remain
        if abs(steering_center) < 0.1:
            if random.randint(1, 3) != 1:
                continue
        # adjust steers of left and right cameras
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction

        # get rid of whitespaces in first 3 columns of csv and make list
        paths = [path.strip() for path in line[0:3]]

        # add data path before each element
        source_paths = [data_path + path for path in paths]

        # images cen lef rig are read and processed from source paths
        imgs = [read_and_process_image(path) for path in source_paths]

        # flip those 3 images and append to image list
        imgs += [flip(image) for image in imgs]

        # delete /IMG from path and add save_path before it and make list
        save_paths = [save_path + path[4:] for path in paths]

        # add flip save paths
        save_paths += [save_path + "flip_" + path[4:] for path in paths]

        for i in range(len(imgs)):
            assert len(imgs) == len(save_paths)
            # print("saving ", save_paths[i])
            cv2.imwrite(save_paths[i], imgs[i])

        steering_angles += [steering_center, steering_left, steering_right]
        # and for flips
        steering_angles += [-steering_center, -steering_left, -steering_right]

        # now we have imgs = [cen, lef, rig, fcen, flef, frig]
        # and steering_angles = [cen, lef, rig, fcen, flef, frig]
        new_file_names += save_paths
        # and file names array of 6 like data/IMG_Augmented/flip_center_bla.jpg

    # save new csv from new_file_names and steering_angles

    with open('data/new_csv.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(new_file_names)):
            writer.writerow([new_file_names[i]] + [steering_angles[i]])


if __name__ == '__main__':
    preprocess_and_save('data/driving_log.csv')
