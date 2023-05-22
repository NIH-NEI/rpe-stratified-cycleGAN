import keras
from keras.models import *
from keras.layers import *
from keras import metrics
from sklearn.metrics import accuracy_score
import sys

import tensorflow as tf

class AddBeta(Layer):
    def __init__(self, **kwargs):
        super(AddBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            return

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)

        self.built = True
        super(AddBeta, self).build(input_shape)

    def call(self, x, training=None):
        return tf.add(x, self.beta)

class G_Guass(Layer):
    def __init__(self, **kwargs):
        super(G_Guass, self).__init__(**kwargs)

    def wi(self, init, name):
        if init == 1:
            return self.add_weight(name='guess_' + name,
                                   shape=(self.size,),
                                   initializer='ones',
                                   trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_' + name,
                                   shape=(self.size,),
                                   initializer='zeros',
                                   trainable=True)
        else:
            raise ValueError("Invalid argument '%d' provided for init in G_Gauss layer" % init)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[0][-1]

        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]
        super(G_Guass, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        z_c, u = x

        def compute(y):
            return y[0] * tf.sigmoid(y[1] * u + y[2]) + y[3] * u + y[4]

        mu = compute(self.a[:5])
        v = compute(self.a[5:])

        z_est = (z_c - mu) * v + mu
        return z_est

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)

def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

def add_noise(inputs, noise_std):
    return Lambda( lambda x: x + tf.random_normal(tf.shape(x)) * noise_std )(inputs)

def get_ladder_network_fc(layer_sizes = [784, 1000, 500, 250, 250, 250, 10],
                          noise_std = 0.3,
                          denoising_cost = [100.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]):
    L = len(layer_sizes) - 1  # number of layers

    inputs_l = Input((layer_sizes[0], ))
    inputs_u = Input((layer_sizes[0], ))

    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[1:]]
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[:-1]]
    betas = [AddBeta() for l in range(L)]

    def encoder(inputs, noise_std):
        h = add_noise(inputs, noise_std)
        all_z = [None for _ in range(len(layer_sizes))]
        all_z[0] = h

        for l in range(1, L + 1):
            z_pre = fc_enc[l - 1](h)
            z = Lambda(batch_normalization)(z_pre)
            z = add_noise(z, noise_std)

            if l == L:
                h = Activation('softmax')(betas[l - 1](z))
            else:
                h = Activation('relu')(betas[l - 1](z))

            all_z[l] = z

        return h, all_z

    y_c_l, _ = encoder(inputs_l, noise_std)
    y_l, _ = encoder(inputs_l, 0.0)

    y_c_u, corr_z = encoder(inputs_u, noise_std)
    y_u, clean_z = encoder(inputs_u, 0.0)

    # Decoder
    d_cost = []  # to store the denoising cost of all layers
    for l in range(L, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = y_c_u
        else:
            u = fc_dec[l](z_est)
        u = Lambda(batch_normalization)(u)
        z_est = G_Guass()([z_c, u])
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    u_cost = tf.add_n(d_cost)

    y_c_l = Lambda(lambda x: x[0])([y_c_l, y_l, y_c_u, y_u, u, z_est, z])

    tr_m = Model([inputs_l, inputs_u], y_c_l)
    tr_m.add_loss(u_cost)
    # tr_m.compile(keras.optimizers.Adam(lr=0.02), 'mean_squared_error', metrics=['accuracy'])
    tr_m.compile(keras.optimizers.Adam(lr=0.02), 'categorical_crossentropy', metrics=['accuracy'])

    tr_m.metrics_names.append("den_loss")
    tr_m.metrics_tensors.append(u_cost)

    te_m = Model(inputs_l, y_l)
    tr_m.test_model = te_m

    return tr_m

def ladder_network_test(unlabeled_images, model_path, n_classes):
    unlabeled_images = np.squeeze(unlabeled_images)
    img_shape = unlabeled_images.shape
    unlabeld_img_shape = unlabeled_images.shape
    inp_size = img_shape[1] * img_shape[2]

    unlabeled_images = unlabeled_images.reshape(unlabeld_img_shape[0], inp_size).astype('float32') / 255
    model = get_ladder_network_fc(layer_sizes=[inp_size, 2000, 1000, 500, 250, 250, n_classes])
    model.load_weights(model_path)
    res = model.test_model.predict(unlabeled_images, batch_size=100)
    res_indices = res.argmax(-1)

    return res_indices

def ladder_network_training(labeled_images, labels, unlabeled_images, test_images, test_labels,
                            model_path, n_classes, epochs = 100, batch_size=32, main_frame=None):
    #squeeze data first
    labeled_images = np.squeeze(labeled_images)
    unlabeled_images = np.squeeze(unlabeled_images)

    labeled_img_shape = labeled_images.shape
    unlabeld_img_shape = unlabeled_images.shape
    inp_size = labeled_img_shape[1] * labeled_img_shape[2]

    labeled_images = labeled_images.reshape(labeled_img_shape[0], inp_size).astype('float32') / 255
    unlabeled_images = unlabeled_images.reshape(unlabeld_img_shape[0], inp_size).astype('float32') / 255

    x_train_unlabeled = unlabeled_images

    x_train_labeled = labeled_images
    y_train_labeled = keras.utils.to_categorical(labels, n_classes)

    n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
    x_train_labeled_rep = np.concatenate([x_train_labeled] * n_rep)
    y_train_labeled_rep = np.concatenate([y_train_labeled] * n_rep)

    less_training_data = x_train_unlabeled.shape[0] - x_train_labeled_rep.shape[0]
    if less_training_data > 0:
        x_train_labeled_rep = np.concatenate((x_train_labeled_rep, x_train_labeled[0:less_training_data]))
        y_train_labeled_rep = np.concatenate((y_train_labeled_rep, y_train_labeled[0:less_training_data]))


    # initialize the model
    model = get_ladder_network_fc(layer_sizes=[inp_size, 2000, 1000, 500, 250, 250, n_classes],
                                  noise_std=0.0001)
    model.summary()

    # train the model for 100 epochs
    best_accuracy = 0
    x_test = test_images.reshape(test_images.shape[0], inp_size).astype('float32') / 255
    y_test = keras.utils.to_categorical(test_labels,  n_classes)
    for epoch in range(1, epochs+1):
        model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1,
                  batch_size=batch_size)
        y_test_pr = model.test_model.predict(x_test)
        accuracy = accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1))
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            model.save_weights(model_path)
        main_frame.write_text('Ladder network test accuracy and best accuracy: {} and {}\n'.format(accuracy,
                                                                                                   best_accuracy))
        main_frame.show_progress(float(epoch) / float(epochs) * 100)

        # Flush out prints each loop iteration
        sys.stdout.flush()


if __name__ == '__main__':
    from keras.datasets import mnist
    import random
    from sklearn.metrics import accuracy_score
    import numpy as np

    # get the dataset
    inp_size = 28*28 # size of mnist dataset
    n_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, inp_size).astype('float32') / 255
    x_test = x_test.reshape(10000, inp_size).astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    # only select 100 training samples
    idxs_annot = range(x_train.shape[0])
    random.seed(0)
    idxs_annot = np.random.choice(x_train.shape[0], 100)

    x_train_unlabeled = x_train
    x_train_labeled = x_train[idxs_annot]
    y_train_labeled = y_train[idxs_annot]

    n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
    x_train_labeled_rep = np.concatenate([x_train_labeled] * n_rep)
    y_train_labeled_rep = np.concatenate([y_train_labeled] * n_rep)

    # initialize the model
    model = get_ladder_network_fc(layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes])
    model.summary()

    # train the model for 100 epochs
    for _ in range(100):
        model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1)
        y_test_pr = model.test_model.predict(x_test, batch_size=100)
        print("Test accuracy : %f" % accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))
