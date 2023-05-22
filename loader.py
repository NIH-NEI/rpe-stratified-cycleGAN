import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import numbers
from keras.utils import Sequence
import csv
from operator import itemgetter
import matplotlib.pyplot as plt

# If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
## potentially change to mean and std. dev.
def normalize_array(array):
    array = array / 127.5 - 1
    return array

# def write_mean_and_std(file_name, imgA_mean, imgA_std, imgB_mean, imgB_std):
#     with open(file_name, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow(['Image Mean', 'Image Std'])
#         csv_writer.writerow([imgA_mean, imgA_std])
#         csv_writer.writerow([imgB_mean, imgB_std])
#
# def read_mean_and_std(file_name):
#     imageA_mean = 0
#     imageA_std = 0
#     imageB_mean = 0
#     imageB_std = 0
#
#     with open(file_name) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 1:
#                 imageA_mean = row[0]
#                 imageA_std = row[1]
#             elif line_count == 2:
#                 imageB_mean = row[0]
#                 imageB_std = row[1]
#             line_count += 1
#
#     return imageA_mean, imageA_std, imageB_mean, imageB_std

def create_image_array(image_list, image_path, trained_image_size, nr_of_channels, normalization=True):

    image_array = []
    for image_name in image_list:
        if nr_of_channels == 1:  # Gray scale image
            image = imread(os.path.join(image_path, image_name), as_gray=True)
            image = image[:, :, np.newaxis]
        else:                   # RGB image
            image = imread(os.path.join(image_path, image_name), as_gray=False)

        img_size = image.shape
        if img_size[0] != trained_image_size or img_size[1] != trained_image_size:
            image = resize(image, (trained_image_size[0], trained_image_size[1], nr_of_channels), preserve_range=True)

        if normalization:
            image = normalize_array(image)
        image_array.append(image)

    # indices, score_sorted = zip(*sorted(enumerate(score_list), key=itemgetter(1)))
    #
    # ix = 1
    # for id, score in zip(indices, score_sorted):
    #     imsave('scored_image'+str(ix) + '.png', image_array[id][...,0])
    #     ix = ix+1

    return np.array(image_array), img_size

class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B,
                 trained_image_size, nr_of_channels, batch_size=1):
        self.batch_size = batch_size
        self.nr_of_channels = nr_of_channels
        # just in case input images have different image size
        self.trained_image_size = trained_image_size

        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A, _ = create_image_array(batch_A, '', self.trained_image_size, self.nr_of_channels, True)
        real_images_B, _ = create_image_array(batch_B, '', self.trained_image_size, self.nr_of_channels, True)

        return real_images_A, real_images_B  # input_data, target_data

def load_training_data(trainA_dir, trainB_dir, trained_image_size, nr_of_channels, batch_size=1, generator=False):

    trainA_image_names = sorted(os.listdir(trainA_dir))
    trainB_image_names = sorted(os.listdir(trainB_dir))

    if generator:
        return data_sequence(trainA_dir, trainB_dir, trainA_image_names, trainB_image_names, trained_image_size,
                             nr_of_channels, batch_size=batch_size)
    else:
        trainA_images, _ = create_image_array(trainA_image_names, trainA_dir, trained_image_size, nr_of_channels)
        trainB_images, _ = create_image_array(trainB_image_names, trainB_dir, trained_image_size, nr_of_channels)

        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names}


def load_test_data(testA_dir, testB_dir, trained_image_size, nr_of_channels, batch_size=1):
    testA_image_names = sorted(os.listdir(testA_dir))
    testB_image_names = sorted(os.listdir(testB_dir))

    testA_images, _ = create_image_array(testA_image_names, testA_dir, trained_image_size, nr_of_channels)
    testB_images, image_size = create_image_array(testB_image_names, testB_dir, trained_image_size, nr_of_channels)

    return {"testA_images": testA_images, "testB_images": testB_images,
            "testA_image_names": testA_image_names,
            "testB_image_names": testB_image_names}, image_size

def load_quality_training_labels(train_labels_path):
    with open(train_labels_path, newline='') as csvfile:
        label_reader = csv.reader(csvfile, delimiter=',')
        dict_labels = {}
        for row in label_reader:
            dict_labels[row[0]] = row[1]
    return dict_labels

def load_ladder_data(labeled_train_dir, unlabeled_train_dir, test_dir, trained_image_size, nr_of_channels):
    ##step 1: load labeled images
    train_img_dir = os.path.join(labeled_train_dir, 'AO_Images')
    train_image_names = sorted(os.listdir(train_img_dir))
    labeled_train_images, _ = create_image_array(train_image_names, train_img_dir,
                                         (trained_image_size, trained_image_size), nr_of_channels, False)

    ##step 2: load training labels
    train_label_path = os.path.join(labeled_train_dir, 'train_labels.csv')
    dict_labels = load_quality_training_labels(train_label_path)
    training_labels = np.ndarray((len(train_image_names),), dtype=np.float32)
    for i, image_name in enumerate(train_image_names):
        image_name_base = os.path.splitext(image_name)[0]
        training_labels[i] = int(dict_labels[image_name_base])

    ##step 3: load unlabeled images
    unlabeled_image_names = sorted(os.listdir(unlabeled_train_dir))
    unlabeled_train_images, _ = create_image_array(unlabeled_image_names, unlabeled_train_dir,
                                                   (trained_image_size, trained_image_size),
                                                   nr_of_channels, False)

    ##step 4: load test images
    test_img_dir = os.path.join(test_dir, 'AO_Images')
    test_image_names = sorted(os.listdir(test_img_dir))
    test_images, _ = create_image_array(test_image_names, test_img_dir,
                                        (trained_image_size, trained_image_size), nr_of_channels, False)

    ## step 5: load test labels
    test_label_path = os.path.join(test_dir, 'test_labels.csv')
    dict_labels = load_quality_training_labels(test_label_path)
    test_labels = np.ndarray((len(test_image_names),), dtype=np.float32)
    for i, image_name in enumerate(test_image_names):
        image_name_base = os.path.splitext(image_name)[0]
        test_labels[i] = int(dict_labels[image_name_base])

    return {"training images": labeled_train_images, "training labels": training_labels,
            "unlabeled training images": unlabeled_train_images,
            "unlabeled image names": unlabeled_image_names,
            "test images": test_images, "test labels": test_labels}

def load_image_quality_data(labeled_train_dir, unlabeled_train_dir, test_dir,
                            reduced_image_size, nr_of_channels):
    ##step 1: load labeled images
    img_dir = os.path.join(os.path.dirname(__file__), labeled_train_dir, 'AO_Images')
    train_image_names = sorted(os.listdir(img_dir))
    labeled_train_images, _ = create_image_array(train_image_names, img_dir,
                                                 (reduced_image_size, reduced_image_size), nr_of_channels, False)

    ##step 2: load labels
    label_path = os.path.join(os.path.dirname(__file__), labeled_train_dir, 'train_labels.csv')
    dict_labels = load_quality_training_labels(label_path)
    labels = np.ndarray((len(train_image_names),1), dtype=np.float32)
    for i, image_name in enumerate(train_image_names):
        image_name_base = os.path.splitext(image_name)[0]
        labels[i] = int(dict_labels[image_name_base])

    ##step 3: load unlabeled images
    img_dir = os.path.join(os.path.dirname(__file__), unlabeled_train_dir)
    train_image_names = sorted(os.listdir(img_dir))
    all_images, _ = create_image_array(train_image_names, img_dir,
                                       (reduced_image_size, reduced_image_size),
                                       nr_of_channels, False)

    ##step 4: check duplicated images
    reduced_labeled_train_images = resize(labeled_train_images, (labeled_train_images.shape[0], 4, 4, 1),
                                          preserve_range=True)
    reduced_all_images = resize(all_images, (all_images.shape[0], 4, 4, 1),
                                          preserve_range=True)
    duplicated_list = []
    for i in range(reduced_labeled_train_images.shape[0]):
        for j in range(reduced_all_images.shape[0]):
            res = np.isclose(reduced_labeled_train_images[i], reduced_all_images[j], atol=0.01)
            if not np.any(res == False):
                duplicated_list.append(j)
                break

    unlabeled_train_images = np.ndarray((all_images.shape[0]-len(duplicated_list),
                                         labeled_train_images.shape[1], labeled_train_images.shape[2],
                                         labeled_train_images.shape[3]), dtype=np.float32)

    img_id = 0
    unlabeled_train_image_names = []
    for i in range(all_images.shape[0]):
        if i not in duplicated_list:
            unlabeled_train_images[img_id] = all_images[i]
            unlabeled_train_image_names.append(train_image_names[i])
            img_id = img_id + 1

    # step 5: load test data
    img_dir = os.path.join(os.path.dirname(__file__), test_dir, 'AO_Images')
    test_image_names = sorted(os.listdir(img_dir))
    test_images, _ = create_image_array(test_image_names, img_dir,
                                        (reduced_image_size, reduced_image_size), nr_of_channels, False)

    label_path = os.path.join(os.path.dirname(__file__), test_dir, 'test_labels.csv')
    dict_labels = load_quality_training_labels(label_path)
    test_labels = np.ndarray((len(test_image_names),1), dtype=np.float32)
    for i, image_name in enumerate(test_image_names):
        image_name_base = os.path.splitext(image_name)[0]
        test_labels[i] = int(dict_labels[image_name_base])

    return {"training images": labeled_train_images,
            "training labels": labels,
            "unlabeled training images": unlabeled_train_images,
            "unlabeled image names": unlabeled_train_image_names,
            "test images": test_images,
            "test labels": test_labels}, duplicated_list

# def load_training_data(AO_dir, spectralis_dir, trained_image_size, nr_of_channels):
#     AO_image_names = sorted(os.listdir(AO_dir))
#     spectralis_image_names = sorted(os.listdir(spectralis_dir))
#     AO_images, _ = create_image_array(AO_image_names, AO_dir, (trained_image_size, trained_image_size),
#                                       nr_of_channels)
#     spectralis_images, _ = create_image_array(spectralis_image_names, spectralis_dir,
#                                               (trained_image_size, trained_image_size), nr_of_channels)
#     return AO_images, spectralis_images

def load_label_training_data(small_AO_dir, small_spectralis_dir, large_AO_dir,
                             large_spectralis_dir, trained_image_size, nr_of_channels,
                             duplicate_list):
    ### first load fold with small number of AO and spectralis images
    small_AO_image_names = sorted(os.listdir(small_AO_dir))
    small_spectralis_image_names = sorted(os.listdir(small_spectralis_dir))
    small_AO_images, _ = create_image_array(small_AO_image_names, small_AO_dir,
                                            (trained_image_size, trained_image_size), nr_of_channels)
    small_spectralis_images, _ = create_image_array(small_spectralis_image_names,
                                                    small_spectralis_dir,
                                                    (trained_image_size, trained_image_size), nr_of_channels)

    ###Load folds with large number of AO and spectralis images
    large_AO_image_names = sorted(os.listdir(large_AO_dir))
    large_spectralis_image_names = sorted(os.listdir(large_spectralis_dir))
    large_AO_images, _ = create_image_array(large_AO_image_names, large_AO_dir,
                                            (trained_image_size, trained_image_size), nr_of_channels)
    large_spectralis_images, _ = create_image_array(large_spectralis_image_names,
                                                    large_spectralis_dir,
                                                    (trained_image_size, trained_image_size), nr_of_channels)

    ###remove duplicate overlap image between small and large AO images
    nonoverlap_AO_images = np.ndarray((large_AO_images.shape[0] - len(duplicate_list),
                                       large_AO_images.shape[1], large_AO_images.shape[2],
                                       large_AO_images.shape[3]), dtype=np.float32)
    nonoverlap_spectralis_images = np.ndarray((large_spectralis_images.shape[0] - len(duplicate_list),
                                               large_spectralis_images.shape[1], large_spectralis_images.shape[2],
                                               large_spectralis_images.shape[3]), dtype=np.float32)
    img_id = 0
    for i, (ao_img, spectralis_img) in enumerate(zip(large_AO_images, large_spectralis_images)):
        if i not in duplicate_list:
            nonoverlap_AO_images[img_id] = ao_img
            nonoverlap_spectralis_images[img_id] = spectralis_img
            img_id = img_id + 1

    AO_images = np.concatenate((small_AO_images, nonoverlap_AO_images), axis=0)
    spectralis_images = np.concatenate((small_spectralis_images, nonoverlap_spectralis_images), axis=0)


    return {"AO images": AO_images, "spectralis images": spectralis_images}

def load_data(data_dir, trained_image_size, nr_of_channels, batch_size=1, generator=False):
    trainA_path = os.path.join(data_dir, 'train', 'Images')
    trainB_path = os.path.join(data_dir, 'train', 'Masks')
    testA_path = os.path.join(data_dir, 'test', 'Images')
    testB_path = os.path.join(data_dir, 'test', 'Masks')

    trainA_image_names = sorted(os.listdir(trainA_path))
    trainB_image_names = sorted(os.listdir(trainB_path))
    testA_image_names = sorted(os.listdir(testA_path))
    testB_image_names = sorted(os.listdir(testB_path))

    if generator:
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, trained_image_size,
                             nr_of_channels, batch_size=batch_size)
    else:
        trainA_images, _ = create_image_array(trainA_image_names, trainA_path, trained_image_size, nr_of_channels)
        trainB_images, _ = create_image_array(trainB_image_names, trainB_path, trained_image_size, nr_of_channels)
        testA_images, _ = create_image_array(testA_image_names, testA_path, trained_image_size, nr_of_channels)
        testB_images, image_size = create_image_array(testB_image_names, testB_path, trained_image_size, nr_of_channels)
        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names,
                "testA_image_names": testA_image_names,
                "testB_image_names": testB_image_names}, image_size

if __name__ == '__main__':
    load_data()