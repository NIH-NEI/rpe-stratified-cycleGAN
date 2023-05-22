from keras.applications import MobileNet, VGG16, VGG19, DenseNet121, VGG19, ResNet50
from keras.layers import Input, Dense, Conv2D, BatchNormalization
from keras.layers import Activation, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import write_csv_file

from keras.utils import to_categorical
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import losses
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import pickle, os, glob, zipfile
from loader import load_image_quality_data

def basic_conv_block(input, chs, rep):
    x = input
    for i in range(rep):
        x = Conv2D(chs, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def create_cnn(input_shape, num_of_classes):
    input = Input(shape=input_shape)
    x = basic_conv_block(input, 64, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 128, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 256, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_of_classes, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_vgg19(input_shape, num_of_classes):
    net = VGG19(input_shape=input_shape, weights=None, include_top=False)
    input = Input(input_shape)
    x = net(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_of_classes, activation="softmax")(x)
    model = Model(input, x)
    return model

def create_res50(input_shape, num_of_classes):
    net = ResNet50(input_shape=input_shape, weights=None, include_top=False)
    input = Input(input_shape)
    x = net(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_of_classes, activation="softmax")(x)
    model = Model(input, x)
    return model

def create_dense121(input_shape, num_of_classes):
    net = DenseNet121(input_shape=input_shape, weights=None, include_top=False)
    input = Input(input_shape)
    x = net(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_of_classes, activation="softmax")(x)
    model = Model(input, x)
    return model

key2backbone = {
    'cnn': create_cnn,
    'vgg': create_vgg19,
    'res': create_res50,
    'dense': create_dense121,
}

class PseudoCallback(Callback):
    def __init__(self, model, x_train_labeled, y_train_labeled, X_train_unlabeled,
                 x_text, y_test, batch_size, num_of_classes):
        self.batch_size = batch_size
        self.model = model
        self.n_classes = num_of_classes

        # input data..
        self.X_train_labeled = x_train_labeled
        self.y_train_labeled = y_train_labeled
        self.X_train_unlabeled = X_train_unlabeled
        self.X_test = x_text
        self.Y_test = y_test
        self.y_train_unlabeled_prediction = np.random.randint(
            num_of_classes, size=(self.X_train_unlabeled.shape[0], 1))
        # steps_per_epoch
        self.train_steps_per_epoch = (self.X_train_labeled.shape[0] + self.X_train_unlabeled.shape[0]) // batch_size
        self.test_stepes_per_epoch = self.X_test.shape[0] // batch_size
        if self.test_stepes_per_epoch == 0:
            self.test_stepes_per_epoch = 1

        #Training parameters..
        self.alpha_t = 0.0

        #Accuracy report
        self.labeled_accuracy = []

    def train_mixture(self):
        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        flag_join = np.r_[np.repeat(0.0, self.X_train_labeled.shape[0]),
                         np.repeat(1.0, self.X_train_unlabeled.shape[0])].reshape(-1,1)
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        return X_train_join[indices], y_train_join[indices], flag_join[indices]

    def train_generator(self):
        while True:
            X, y, flag = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch = (X[i*self.batch_size:(i+1)*self.batch_size]/255.0).astype(np.float32)
                y_batch = to_categorical(y[i*self.batch_size:(i+1)*self.batch_size], self.n_classes)
                y_batch = np.c_[y_batch, flag[i*self.batch_size:(i+1)*self.batch_size]]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.Y_test.shape[0])
            np.random.shuffle(indices)
            for i in range(len(indices)//self.batch_size):
                current_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
                X_batch = (self.X_test[current_indices] / 255.0).astype(np.float32)
                y_batch = to_categorical(self.Y_test[current_indices], self.n_classes)
                y_batch = np.c_[y_batch, np.repeat(0.0, y_batch.shape[0])]
                yield X_batch, y_batch

    def loss_function(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        unlabeled_flag = y_true[:, self.n_classes]
        entropies = categorical_crossentropy(y_true_item, y_pred)
        coefs = 1.0-unlabeled_flag + self.alpha_t * unlabeled_flag # 1 if labeled, else alpha_t
        return coefs * entropies

    def accuracy(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
        # update alpha(t)
        if epoch < 10:
            self.alpha_t = 0.0
        elif epoch >= 70:
            self.alpha_t = 3.0
        else:
            self.alpha_t = (epoch - 10.0) / (70.0-10.0) * 3.0
        # updated unlabeled
        self.y_train_unlabeled_prediction = np.argmax(
            self.model.predict(self.X_train_unlabeled), axis=-1,).reshape(-1, 1)
        # print(self.y_train_unlabeled_prediction)
        y_train_labeled_prediction = np.argmax(
            self.model.predict(self.X_train_labeled), axis=-1).reshape(-1, 1)
        # print(y_train_labeled_prediction)

        # compute labeled data ground-truth
        self.labeled_accuracy.append(np.mean(
            self.y_train_labeled == y_train_labeled_prediction))
        print("labeled : ", self.labeled_accuracy[-1])

    def on_train_end(self, logs):
        y_true = np.ravel(self.Y_test)
        emb_model = Model(self.model.input, self.model.layers[-2].output)
        embedding = emb_model.predict(self.X_test / 255.0)
        proj = TSNE(n_components=2).fit_transform(embedding)
        cmp = plt.get_cmap("tab10")
        plt.figure()
        for i in range(10):
            select_flag = y_true == i
            plt_latent = proj[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
        plt.savefig(f"embedding_{self.X_train_labeled.shape[0]:05}.png")


def train(backbone, input_shape, x_train_labeled, y_train_labeled, x_train_unlabeled, x_test, y_test,
          num_of_classes, batch_size, epochs):
    model = key2backbone[backbone](input_shape, num_of_classes)
    model.summary()

    pseudo = PseudoCallback(model, x_train_labeled, y_train_labeled, x_train_unlabeled,
                            x_test, y_test, batch_size, num_of_classes)
    model.compile("adam", loss=pseudo.loss_function, metrics=[pseudo.accuracy])

    model_name = 'image_quality_test.hdf5'
    # model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)
    hist = model.fit_generator(pseudo.train_generator(), steps_per_epoch=pseudo.train_steps_per_epoch,
                               validation_data=pseudo.test_generator(), callbacks=[pseudo, model_checkpoint],
                               validation_steps=pseudo.test_stepes_per_epoch, epochs=epochs).history

    hist["labeled_accuracy"] = pseudo.labeled_accuracy
    print(hist["labeled_accuracy"])

    return model

def supervised_train(backbone, input_shape, x_train_labeled, y_train_labeled, num_of_classes):
    model = key2backbone[backbone](input_shape, num_of_classes)
    model.summary()

    x_train_labeled = x_train_labeled / 255.0
    y_train_labeled = to_categorical(y_train_labeled, num_of_classes)

    model.compile("adam", loss=losses.categorical_crossentropy, metrics=[categorical_accuracy])
    model.fit(x_train_labeled, y_train_labeled, batch_size=8, epochs=20, verbose=1, shuffle=True,
              validation_split=0.1)
    return model

def test(backbone, model_path, input_shape, num_of_classes, test_images):
    model = key2backbone[backbone](input_shape, num_of_classes)
    model.summary()
    model.load_weights(model_path)

    test_images = (test_images / 255.0).astype(np.float32)
    res = np.argmax(model.predict(test_images, verbose=1), axis=-1, ).reshape(-1, 1)
    return res


if __name__ == "__main__":
    input_dict = load_image_quality_data(labeled_train_dir='data\\MICCAI2020\\res_reduced32\\train',
                                         unlabeled_train_dir='data\\MICCAI2020\\res_reduced8\\train',
                                         test_dir='data\\MICCAI2020\\res_reduced32\\test',
                                         reduced_image_size=256, nr_of_channels=1)

    num_of_classes = 3
    trained_model = train('res', input_shape=(256, 256, 1), x_train_labeled=input_dict["training images"],
          y_train_labeled=input_dict["training labels"],
          x_train_unlabeled=input_dict["unlabeled training images"], x_test=input_dict["test images"],
          y_test=input_dict["test labels"], num_of_classes=num_of_classes, batch_size=16,
                          epochs=20)

    # trained_model = supervised_train(input_shape=(256, 256, 1), x_train_labeled=input_dict["training images"],
    #                                  y_train_labeled=input_dict["training labels"], num_of_classes=num_of_classes)

    test_images = input_dict["all images"] / 255.0
    y_test_pred1 =trained_model.predict(test_images, verbose=1)
    y_test_pred = np.argmax(trained_model.predict(test_images, verbose=1), axis=-1, ).reshape(-1, 1)
    # print(input_dict["test labels"])
    # print('predicted\n')
    write_csv_file('image_scores.csv', input_dict["all image names"], y_test_pred)

    # x_test = (input_dict["training images"] / 255.0).astype(np.float32)
    # image_quality_scores = test('image_quality_test.hdf5', (256, 256, 1), 6, input_dict["all images"])
    # write_csv_file('image_scores.csv', input_dict["all image names"], image_quality_scores)
