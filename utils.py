import os
import matplotlib.pyplot as plt
import csv

def is_valid_root_dir(root_dir):
    """it only contains a set of sub directory.
    One of sub directories is Images, and the remaining ones are masks"""
    if not os.path.isdir(root_dir):
        return False

    image_dir_flag = True
    for fname in os.listdir(root_dir):
        if os.path.isfile(fname):
            return False
        if 'Images' not in fname and 'Masks' not in fname:
            image_dir_flag = False

    return image_dir_flag

def visualize_image(images, num_of_images = 4):
    if num_of_images > images.shape[0]:
        num_of_images = images.shape[0]

    f, axarr = plt.subplots(num_of_images, 1)
    for i in range(num_of_images):
        axarr[i].imshow(images[i,...,0], cmap='gray')
    plt.show()

def is_file_existed(file_name):
    if os.path.exists(file_name) and os.path.isfile(file_name):
        return True
    else:
        return False

def write_csv_file1(file_name, image_names, scores):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for (image_name, score) in zip(image_names, scores):
            writer.writerow([image_name, score])

def write_csv_file(file_name, scores):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for score in scores:
            writer.writerow([score])

def read_csv_file(file_name, id=0):
    scores = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            scores.append(int(row[id]))

    return scores