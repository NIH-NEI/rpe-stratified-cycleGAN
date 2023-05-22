from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, Embedding, Reshape
from keras import layers
from keras.layers import UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras.layers import InputSpec
from instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from utils import visualize_image
from keras.objectives import categorical_crossentropy
from keras.utils import to_categorical

from collections import OrderedDict
from skimage.io import imsave
import time
import sys
import os
import numpy as np
import json
import datetime
import random
import math
import csv

import keras.backend as K
import tensorflow as tf
from loader import load_data, load_training_data, load_test_data
from skimage.transform import resize

np.random.seed(seed=12345)

class LabelCycleGAN():

    def __init__(self, image_shape=(256*1, 256*1, 1), num_classes=3, epochs = 200, saving_interval = 20,
                 batch_size = 4, cycle_loss_type = 0, date_time_string_addition='_labelcyclegan'):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.lambda_C = 1.0  # weight for classifier
        self.learning_rate_D = 2e-4
        self.learning_rate_G = 2e-4
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = batch_size
        self.epochs = epochs  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = saving_interval
        self.synthetic_pool_size = 50
        self.num_classes = num_classes

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = False
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = True
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        D_A = self.modelDiscriminator()
        D_B = self.modelDiscriminatorandClassifier()
        loss_weights_D_A = [0.5]
        loss_weights_D_B = [0.5, 0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Discriminator builds
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        [guess_B, guess_B_labels] = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=[guess_B, guess_B_labels], name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D_A)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=[self.lse, self.sparse_categorical_lse],
                         loss_weights=loss_weights_D_B)

        # Use Networks to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=[guess_B, guess_B_labels], name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGeneratorandClassifier(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        self.G_A2B.summary()

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        real_B_labels = Input(shape=(1,), dtype='int32', name='real_B_labels')
        synthetic_B = self.G_A2B([real_A, real_B_labels])
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        [dB_guess_synthetic, dB_guess_synthetic_labels]  = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)

        ### will test it later...
        # reconstructed_B = self.G_A2B([synthetic_A, dB_guess_synthetic_labels])
        reconstructed_B = self.G_A2B([synthetic_A, real_B_labels])
        model_outputs = [reconstructed_A, reconstructed_B]

        if cycle_loss_type == 0:
            compile_losses = [self.cycle_loss, self.cycle_loss,
                              self.lse, self.lse, self.sparse_categorical_lse]
        elif cycle_loss_type == 1:
            compile_losses = [self.cycle_gradient_loss, self.cycle_gradient_loss,
                              self.lse, self.lse, self.sparse_categorical_lse]
        else:
            compile_losses = [self.cycle_gradient_laplacine_loss, self.cycle_gradient_laplacine_loss,
                              self.lse, self.lse, self.sparse_categorical_lse]

        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D, self.lambda_C]

        model_outputs.append(dA_guess_synthetic)
        model_outputs.append(dB_guess_synthetic)
        model_outputs.append(dB_guess_synthetic_labels)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B, real_B_labels],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        self.G_A2B.summary()

    def load_training_data(self, spectralis_images, ao_images, ao_labels):
        print('--- Caching data ---')
        sys.stdout.flush()

        self.A_train = spectralis_images
        self.B_train = ao_images
        self.B_label = ao_labels
        print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('Label-CycleGAN-Images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()

    def load_test_data(self, input_dir1, input_dir2):
        # ======= Data ==========
        print('--- Caching data ---')
        sys.stdout.flush()

        data, self.test_original_size = load_test_data(testA_dir=input_dir1, testB_dir=input_dir2,
                                                       trained_image_size=self.img_shape,
                                                       nr_of_channels=self.channels, batch_size=self.batch_size)

        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]
        print('Data has been loaded')

    def train_model(self, main_frame):
        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ===== Tests ======
        # Simple Model
#         self.G_A2B = self.modelSimple('simple_T1_2_T2_model')
#         self.G_B2A = self.modelSimple('simple_T2_2_T1_model')
#         self.G_A2B.compile(optimizer=Adam(), loss='MAE')
#         self.G_B2A.compile(optimizer=Adam(), loss='MAE')
#         # self.trainSimpleModel()
#         self.load_model_and_generate_synthetic_images()

        # ======= Initialize training ==========
        sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        self.train(main_frame, epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        #self.load_model_and_generate_synthetic_images()

    def test_model(self, weight_path, weight_path1, dirA_name, dirB_name, main_frame):
        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ===== Tests ======
        # Simple Model
        # self.G_A2B = self.modelSimple('simple_T1_2_T2_model')
        # self.G_B2A = self.modelSimple('simple_T2_2_T1_model')
        # self.G_A2B.compile(optimizer=Adam(), loss='MAE')
        # self.G_B2A.compile(optimizer=Adam(), loss='MAE')
        # self.trainSimpleModel()
        self.load_model_and_generate_synthetic_images(weight_path, weight_path1, dirA_name, dirB_name, main_frame)

        # ======= Initialize training ==========
        # sys.stdout.flush()
        # # plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        # self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        # self.load_model_and_generate_synthetic_images()

# ===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(
                x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        # x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelDiscriminatorandClassifier(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x1 = Flatten()(x)
            label = Dense(self.num_classes, activation="softmax")(x1)
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x1 = Flatten()(x)
            label = Dense(self.num_classes, activation="softmax")(x1)
            x = Dense(1)(x1)
        # x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=[x, label], name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGeneratorandClassifier(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
        img_size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        label_embedding = Embedding(self.num_classes, img_size)(label)
        label_embedding = Reshape(self.img_shape)(label_embedding)

        # input_data = layers.multiply([input_img, label_embedding])
        input_data = layers.concatenate([input_img, label_embedding])
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_data)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=[input_img, label], outputs=x, name=name)

# ===============================================================================
# Training
    def train(self, main_frame, epochs,  batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
            # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict([real_images_A, real_labels_B])
            synthetic_images_A = self.G_B2A.predict(real_images_B)

            # visualize_image(real_images_A)
            # visualize_image(synthetic_images_B)
            # visualize_image(real_images_B)
            # visualize_image(synthetic_images_A)

            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B, real_labels_B_query = synthetic_pool_B.query(synthetic_images_B, real_labels_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=[ones, real_labels_B])
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=[zeros, real_labels_B_query])
                DA_loss = DA_loss_real + DA_loss_synthetic
                DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            target_data.append(ones)
            target_data.append(ones)
            target_data.append(to_categorical(real_labels_B, self.num_classes))

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B, real_labels_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % self.save_interval == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, real_labels_B, synthetic_images_A,
                                     synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImageLabelPool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        label_shape = (batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            A_train = self.A_train
            B_train = self.B_train
            B_label = self.B_label
            random_order_A = np.random.randint(len(A_train), size=len(A_train))
            random_order_B = np.random.randint(len(B_train), size=len(B_train))
            epoch_iterations = max(len(random_order_A), len(random_order_B))
            min_nr_imgs = min(len(random_order_A), len(random_order_B))

            # If we want supervised learning the same images form
            # the two domains are needed during each training iteration
            if self.use_supervised_learning:
                random_order_B = random_order_A

            for loop_index in range(0, epoch_iterations, batch_size):
                if loop_index + batch_size >= min_nr_imgs:
                    # If all images soon are used for one domain,
                    # randomly pick from this domain
                    if len(A_train) <= len(B_train):
                        indexes_A = np.random.randint(len(A_train), size=batch_size)

                        # if all images are used for the other domain
                        if loop_index + batch_size >= epoch_iterations:
                            indexes_B = random_order_B[epoch_iterations - batch_size:
                                                        epoch_iterations]
                        else:  # if not used, continue iterating...
                            indexes_B = random_order_B[loop_index:
                                                        loop_index + batch_size]

                    else:  # if len(B_train) <= len(A_train)
                        indexes_B = np.random.randint(len(B_train), size=batch_size)
                        # if all images are used for the other domain
                        if loop_index + batch_size >= epoch_iterations:
                            indexes_A = random_order_A[epoch_iterations - batch_size:
                                                        epoch_iterations]
                        else:  # if not used, continue iterating...
                            indexes_A = random_order_A[loop_index:
                                                        loop_index + batch_size]

                else:
                    indexes_A = random_order_A[loop_index:
                                                loop_index + batch_size]
                    indexes_B = random_order_B[loop_index:
                                                loop_index + batch_size]

                sys.stdout.flush()
                real_images_A = A_train[indexes_A]
                real_images_B = B_train[indexes_B]
                real_labels_B = B_label[indexes_B]

                # Run all training steps
                run_training_iteration(loop_index, epoch_iterations)

            # ================== within epoch loop end ==========================
            if epoch % self.save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch,
                      '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B, real_labels_B)

            # if epoch % 20 == 0:
            if epoch % self.save_interval == 0:
                # self.saveModel(self.D_A, epoch)
                # self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)
            main_frame.show_progress(float(epoch) / float(epochs) * 100)

            # Flush out prints each loop iteration
            sys.stdout.flush()

# ===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def sparse_categorical_lse(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        return loss

    def cycle_gradient_laplacine_loss(self, y_true, y_pred):
        dx_true, dy_true = tf.image.image_gradients(y_true)
        dxx_true, dxy_true = tf.image.image_gradients(dx_true)
        dyx_true, dyy_true = tf.image.image_gradients(dy_true)
        laplace_true = tf.math.add(dxx_true, dyy_true)

        dx_pred, dy_pred = tf.image.image_gradients(y_pred)
        dxx_pred, dxy_pred = tf.image.image_gradients(dx_pred)
        dyx_pred, dyy_pred = tf.image.image_gradients(dy_pred)
        laplace_pred = tf.math.add(dxx_pred, dyy_pred)
        loss = tf.reduce_mean(tf.abs(y_pred - y_true) + tf.abs(dx_pred - dx_true)
                              + tf.abs(dy_pred - dy_true) + tf.abs(laplace_pred-laplace_true))
        return loss

    def cycle_gradient_loss(self, y_true, y_pred):
        dx_true, dy_true = tf.image.image_gradients(y_true)
        dx_pred, dy_pred = tf.image.image_gradients(y_pred)
        loss = tf.reduce_mean(tf.abs(y_pred - y_true) + tf.abs(dx_pred - dx_true)
                              + tf.abs(dy_pred - dy_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        # toimage(image, cmin=-1, cmax=1).save(path_name)
        imsave(path_name, image)

    def saveImages(self, epoch, real_image_A, real_image_B, real_label_B, num_saved_images=1):
        directory = os.path.join('Label-CycleGAN-Images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))
            os.makedirs(os.path.join(directory, 'Atest'))
            os.makedirs(os.path.join(directory, 'Btest'))

        testString = ''

        real_image_Ab = None
        real_image_Ba = None
        # for i in range(num_saved_images + 1):
        for i in range(num_saved_images):
            if len(real_image_A.shape) < 4:
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)

            synthetic_image_B = self.G_A2B.predict([real_image_A, real_label_B])
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict([synthetic_image_A, real_label_B])

            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                 'Label-CycleGAN-Images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'A' + testString, epoch, i))
            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                 'Label-CycleGAN-Images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'B' + testString, epoch, i))

    def save_tmp_images(self, real_image_A, real_image_B, real_label_B,
                        synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict([synthetic_image_A, real_label_B])

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 'Label-CycleGAN-Images/{}/{}.png'.format(
                                     self.date_time, 'tmp'))
        except:  # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_samples_{}_epoch_{}.hdf5'.format(self.date_time, model.name,
                                                                                    len(self.A_train), epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_samples_{}_epoch_{}.json'.format(self.date_time, model.name,
                                                                                  len(self.A_train), epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('Label-CycleGAN-Images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('Label-CycleGAN-Images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            # 'number of A test examples': len(self.A_test),
            # 'number of B test examples': len(self.B_test),
        })

        with open('Label-CycleGAN-Images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model, path_to_weights):
        # model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self, weight_path, weight_path1, dirA_name, dirB_name, main_frame):
        ###in this case, only save results for A to B
        self.load_model_and_weights(self.G_A2B, weight_path)
        self.load_model_and_weights(self.G_B2A, weight_path1)

        weight_base = os.path.basename(weight_path)
        weight_base_no_extension = os.path.splitext(weight_base)[0]
        weight_components = weight_base_no_extension.split('_')
        epoch = int(weight_components[-1])
        samples = int(weight_components[-3])

        # visualize_image(self.A_test)
        # visualize_image(synthetic_images_B)
        # visualize_image(self.B_test)
        # visualize_image(synthetic_images_A)

        directory = os.path.join('generate_images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        def save_image(image, name, domain):
            if self.channels == 1:
                image = image[:, :, 0]
            # toimage(image, cmin=-1, cmax=1).save(os.path.join(
            #     'generate_images', 'synthetic_images', domain, name))
            image = resize(image, self.test_original_size, preserve_range=True)
            # image = image * img_std + img_mean
            imsave(os.path.join('generate_images', self.date_time, domain, name), image)

        # Test A images
        test_labels = np.ndarray((self.A_test.shape[0],), dtype=np.float32)
        for i in range(3):
            test_labels[...] = i
            synthetic_images_B = self.G_A2B.predict([self.A_test, test_labels])

            sub_dir_name = dirB_name + '_' + str(i)
            directory = os.path.join('generate_images', self.date_time, sub_dir_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            for j in range(len(synthetic_images_B)):
                # Get the name from the image it was conditioned on
                name = self.testB_image_names[j].strip('.tif') + '_samples' + str(samples) + \
                       '_iterations' + str(epoch) + '_synthetic.png'
                synt_B = synthetic_images_B[j]
                save_image(synt_B, name, sub_dir_name)

            main_frame.show_progress(float(i+1) / 3.0 * 100)

        # Test A images
        synthetic_images_A = self.G_B2A.predict(self.B_test)
        directory = os.path.join('generate_images', self.date_time, dirA_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(len(synthetic_images_A)):
            # Get the name from the image it was conditioned on
            name = self.testB_image_names[i].strip('.tif') + str(samples) + '_iterations' + str(epoch) \
                       + '_synthetic.png'
            synt_A = synthetic_images_A[i]
            save_image(synt_A, name, dirA_name)
            main_frame.show_progress(float(i + 1) / float(len(synthetic_images_A)) * 100)

        main_frame.write_text('synthetic images have been generated')

# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

class ImageLabelPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            self.labels = []

    def query(self, images, labels):
        if self.pool_size == 0:
            return images, labels
        return_labels = []
        return_images = []
        for image, label in zip(images, labels):
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                    self.labels = label
                else:
                    self.images = np.vstack((self.images, image))
                    self.labels = np.vstack((self.labels, label))

                if len(return_images) == 0:
                    return_images = image
                    return_labels = label
                else:
                    return_images = np.vstack((return_images, image))
                    return_labels = np.vstack((return_labels, label))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    tmp_labels = self.labels[random_id]

                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    self.labels[random_id] = label
                    if len(return_images) == 0:
                        return_images = tmp
                        return_labels = tmp_labels
                    else:
                        return_images = np.vstack((return_images, tmp))
                        return_labels = np.vstack((return_labels, tmp_labels))
                else:
                    if len(return_images) == 0:
                        return_images = image
                        return_labels = label
                    else:
                        return_images = np.vstack((return_images, image))
                        return_labels = np.vstack((return_labels, label))

        return return_images, return_labels