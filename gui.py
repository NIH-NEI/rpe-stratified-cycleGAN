import os
from models import CycleGAN, LabelCycleGAN
from utils import is_valid_root_dir, is_file_existed, write_csv_file, read_csv_file, write_csv_file1
from loader import load_image_quality_data, load_label_training_data, load_training_data
# from ladder_net import ladder_network_training, ladder_network_test
import pseudo_label
import tensorflow as tf
import numpy as np
from dialogs import CycleTrainingInputDialog, LabelCycleTrainingInputDialog, display_error, \
    display_option, TestInputDialog, FixedLabelCycleTrainingInputDialog, SupervisedCycleTrainingInputDialog

try:
    from PyQt5 import QtCore, QtWidgets, Qt, QtGui
except ImportError:
    pass

class MainFrame(QtWidgets.QDialog):
    def __init__(self, frame_width, frame_height, parent=None):
        self.training_settings = {
            'training_size': 256,
            'batch_size': tf.contrib.eager.num_gpus(),
            'epochs': 2,
            'saving_interval': 20,
            'classification_training_size': 256,
            'classification_batch_size': 16,
            'classification_epochs': 20,
        }
        # self.input_data_dir = None
        # self._training_test_data = {
        #     'training_root': None,
        #     'mask_dirs': None,
        #     'test_root': None,
        #     'image_mean': None,
        #     'mask_mean': None,
        #     'image_std': None,
        # }

        super(MainFrame, self).__init__(parent)
        self._setup_layout()
        self.setMinimumSize(frame_width, frame_height)
        self.setWindowTitle('AO-ICG Image Superresolution')
        self.framewidth = frame_width
        self.frameheight = frame_height

    def _setup_layout(self):
        self.setModal(True)
        view_layout = QtWidgets.QGridLayout()

        train_model_button = QtWidgets.QPushButton('Train', self)
        # train_model_button.clicked.connect(self.train_cyclegan)
        cycle_gan_action = QtWidgets.QAction('CycleGAN', train_model_button)
        fixed_label_cycle_gan_action = QtWidgets.QAction('Fixed-Label-CycleGAN', train_model_button)
        label_cycle_gan_action = QtWidgets.QAction('Label-CycleGAN', train_model_button)
        supervised_cycle_gan_action = QtWidgets.QAction('supervised-cycleGAN', train_model_button)

        cycle_gan_action.triggered.connect(self.train_cyclegan)
        label_cycle_gan_action.triggered.connect(self.train_label_cyclegan)
        fixed_label_cycle_gan_action.triggered.connect(self.train_fixed_label_cyclegan)
        supervised_cycle_gan_action.triggered.connect(self.train_supervised_cyclegan)

        train_model_menu = QtWidgets.QMenu()
        train_model_menu.addAction(cycle_gan_action)
        train_model_menu.addAction(label_cycle_gan_action)
        train_model_menu.addAction(fixed_label_cycle_gan_action)
        train_model_menu.addAction(supervised_cycle_gan_action)
        train_model_menu.setDefaultAction(label_cycle_gan_action)
        train_model_button.setMenu(train_model_menu)

        test_model_button = QtWidgets.QPushButton('Test', self)
        test_model_button.clicked.connect(self.test_gan)

        self._textbox = QtWidgets.QPlainTextEdit()
        self._textbox.setReadOnly(True)

        self._progressbar = QtWidgets.QProgressBar()

        view_layout.addWidget(train_model_button, 0, 0, 1, 2)
        view_layout.addWidget(test_model_button, 0, 2, 1, 2)
        view_layout.addWidget(self._textbox, 1, 0, 1, 4)
        view_layout.addWidget(self._progressbar, 2, 0, 1, 4)

        self.setLayout(view_layout)

    def train_image_labels(self, backbone, training_dir, ao_dir, test_dir, train_image_size,
                           batch_size, epochs, training_flag):

        classification_input_dict, duplicate_list = load_image_quality_data(
            labeled_train_dir=training_dir, unlabeled_train_dir=ao_dir,
            test_dir=test_dir, reduced_image_size=train_image_size,
            nr_of_channels=1)
        write_csv_file('overlap_image_indices.csv', duplicate_list)

        if training_flag:
            pseudo_label.train(backbone=backbone, input_shape=(train_image_size, train_image_size, 1),
                               x_train_labeled=classification_input_dict["training images"],
                               y_train_labeled=classification_input_dict["training labels"],
                               x_train_unlabeled=classification_input_dict["unlabeled training images"],
                               x_test=classification_input_dict["test images"],
                               y_test=classification_input_dict["test labels"],
                               num_of_classes=int(np.amax(classification_input_dict["training labels"]) + 1),
                               batch_size=self.training_settings['classification_batch_size'],
                               epochs=self.training_settings['classification_epochs'])

        quality_scores = pseudo_label.test(backbone=backbone, model_path='image_quality_test.hdf5',
                                           input_shape=(train_image_size, train_image_size, 1),
                                           num_of_classes=int(np.amax(classification_input_dict["training labels"])+1),
                                           test_images=classification_input_dict["unlabeled training images"])
        quality_scores = np.squeeze(quality_scores)
        write_csv_file1('image_scores.csv', classification_input_dict["unlabeled image names"], quality_scores)

    def train_label_cyclegan(self):
        training_input_dlg = LabelCycleTrainingInputDialog(self.framewidth, self.frameheight,
                                                           self.training_settings['training_size'],
                                                           self.training_settings['batch_size'],
                                                           self.training_settings['saving_interval'],
                                                           self.training_settings['epochs'],
                                                           self.training_settings['classification_training_size'],
                                                           self.training_settings['classification_batch_size'],
                                                           self.training_settings['classification_epochs'], self)
        if training_input_dlg.exec():
            if training_input_dlg.get_spectralis_dir() is None or training_input_dlg.get_AO_dir() is None \
                    or training_input_dlg.get_classification_training_input_dir() is None \
                    or training_input_dlg.get_classification_test_input_dir() is None:
                display_error('Input error', 'Either spectralis or AO directories or classification '
                                             'data are not selected')
                return

            self.write_text('Sucessfully load training directories: {}\n{}\n'.format(
                training_input_dlg.get_spectralis_dir(), training_input_dlg.get_AO_dir()))

            if not is_file_existed('overlap_image_indices.csv') or not is_file_existed('image_scores.csv'):
                self.train_image_labels(backbone='res',
                                        training_dir=training_input_dlg.get_classification_training_input_dir(),
                                        ao_dir=training_input_dlg.get_AO_dir(),
                                        test_dir=training_input_dlg.get_classification_test_input_dir(),
                                        train_image_size=self.training_settings['classification_training_size'],
                                        batch_size=self.training_settings['classification_batch_size'],
                                        epochs=self.training_settings['classification_epochs'],
                                        training_flag=True)
            else:
                para_option_dlg = display_option('Existing parameters',
                                                 'Do you want to use exising image indices and scores?')
                if para_option_dlg.clickedButton() == para_option_dlg.button(QtWidgets.QMessageBox.Cancel):
                    return
                elif para_option_dlg.clickedButton() == para_option_dlg.button(QtWidgets.QMessageBox.Discard):
                    if is_file_existed('image_quality_test.hdf5'):
                        weight_option_dlg = display_option('Classification weights option',
                                                'Do you want to use or overwrite classification weight')
                        if weight_option_dlg.clickedButton() == weight_option_dlg.button(QtWidgets.QMessageBox.Discard):
                            self.train_image_labels(backbone='res',
                                                    training_dir=training_input_dlg.get_classification_training_input_dir(),
                                                    ao_dir=training_input_dlg.get_AO_dir(),
                                                    test_dir=training_input_dlg.get_classification_test_input_dir(),
                                                    train_image_size=self.training_settings['classification_training_size'],
                                                    batch_size=self.training_settings['classification_batch_size'],
                                                    epochs=self.training_settings['classification_epochs'],
                                                    training_flag=True)
                        elif weight_option_dlg.clickedButton() == weight_option_dlg.button(QtWidgets.QMessageBox.Ok):
                            self.train_image_labels(backbone='res',
                                                    training_dir=training_input_dlg.get_classification_training_input_dir(),
                                                    ao_dir=training_input_dlg.get_AO_dir(),
                                                    test_dir=training_input_dlg.get_classification_test_input_dir(),
                                                    train_image_size=self.training_settings['classification_training_size'],
                                                    batch_size=self.training_settings['classification_batch_size'],
                                                    epochs=self.training_settings['classification_epochs'],
                                                    training_flag=False)


            ### start to load training data for label cycle GAN
            duplicate_list = read_csv_file('overlap_image_indices.csv', 0)
            small_training_labels = read_csv_file(os.path.join(training_input_dlg.get_classification_training_input_dir(),
                                                               'train_labels.csv'), 1)
            quality_scores = read_csv_file('image_scores.csv', 1)

            ### start to train cycle GAN
            self.training_settings['training_size'] = training_input_dlg.get_training_size()
            self.training_settings['batch_size'] = training_input_dlg.get_batch_size()
            self.training_settings['saving_interval'] = training_input_dlg.get_saving_interval()
            self.training_settings['epochs'] = training_input_dlg.get_epochs()

            gan_data = load_label_training_data(small_AO_dir=os.path.join(
                training_input_dlg.get_classification_training_input_dir(), 'AO_Images'),
                small_spectralis_dir=os.path.join(training_input_dlg.get_classification_training_input_dir(),
                                              'Spectralis30_Images'), large_AO_dir=training_input_dlg.get_AO_dir(),
                large_spectralis_dir=training_input_dlg.get_spectralis_dir(),
                trained_image_size=self.training_settings['training_size'],
                nr_of_channels = 1, duplicate_list=duplicate_list)

            AO_train_images = gan_data["AO images"]
            AO_train_labels = np.concatenate((small_training_labels, quality_scores))
            spectralis_train_images = gan_data["spectralis images"]

            classification_cycle_gan = LabelCycleGAN(image_shape=(self.training_settings['training_size'],
                                                                  self.training_settings['training_size'], 1),
                                                     num_classes=3,
                                                     epochs=self.training_settings['epochs'],
                                                     saving_interval=self.training_settings['saving_interval'],
                                                     batch_size=self.training_settings['batch_size'],
                                                     cycle_loss_type= training_input_dlg.get_cycle_loss())
            classification_cycle_gan.load_training_data(spectralis_images=spectralis_train_images,
                                                        ao_images=AO_train_images, ao_labels=AO_train_labels)
            classification_cycle_gan.train_model(self)

    def train_fixed_label_cyclegan(self):
        training_input_dlg = FixedLabelCycleTrainingInputDialog(self.framewidth, self.frameheight,
                                                           self.training_settings['training_size'],
                                                           self.training_settings['batch_size'],
                                                           self.training_settings['saving_interval'],
                                                           self.training_settings['epochs'],
                                                           self.training_settings['classification_training_size'],
                                                           self.training_settings['classification_batch_size'],
                                                           self.training_settings['classification_epochs'], self)

        if training_input_dlg.exec():
            if training_input_dlg.get_spectralis_dir() is None or training_input_dlg.get_AO_dir() is None \
                    or training_input_dlg.get_classification_training_input_dir() is None \
                    or training_input_dlg.get_classification_test_input_dir() is None:
                display_error('Input error', 'Either spectralis or AO directories or classification '
                                             'data are not selected')
                return

            self.write_text('Sucessfully load training directories: {}\n{}\n'.format(
                training_input_dlg.get_spectralis_dir(), training_input_dlg.get_AO_dir()))

            if not is_file_existed('overlap_image_indices.csv') or not is_file_existed('image_scores.csv'):
                self.train_image_labels(backbone='res',
                                        training_dir=training_input_dlg.get_classification_training_input_dir(),
                                        ao_dir=training_input_dlg.get_AO_dir(),
                                        test_dir=training_input_dlg.get_classification_test_input_dir(),
                                        train_image_size=self.training_settings['classification_training_size'],
                                        batch_size=self.training_settings['classification_batch_size'],
                                        epochs=self.training_settings['classification_epochs'],
                                        training_flag=True)
            else:
                para_option_dlg = display_option('Existing parameters',
                                                 'Do you want to use exising image indices and scores?')
                if para_option_dlg.clickedButton() == para_option_dlg.button(QtWidgets.QMessageBox.Cancel):
                    return
                elif para_option_dlg.clickedButton() == para_option_dlg.button(QtWidgets.QMessageBox.Discard):
                    if is_file_existed('image_quality_test.hdf5'):
                        weight_option_dlg = display_option('Classification weights option',
                                                           'Do you want to use or overwrite classification weight')
                        if weight_option_dlg.clickedButton() == weight_option_dlg.button(QtWidgets.QMessageBox.Discard):
                            self.train_image_labels(backbone='res',
                                                    training_dir=training_input_dlg.get_classification_training_input_dir(),
                                                    ao_dir=training_input_dlg.get_AO_dir(),
                                                    test_dir=training_input_dlg.get_classification_test_input_dir(),
                                                    train_image_size=self.training_settings[
                                                        'classification_training_size'],
                                                    batch_size=self.training_settings['classification_batch_size'],
                                                    epochs=self.training_settings['classification_epochs'],
                                                    training_flag=True)
                        elif weight_option_dlg.clickedButton() == weight_option_dlg.button(QtWidgets.QMessageBox.Ok):
                            self.train_image_labels(backbone='res',
                                                    training_dir=training_input_dlg.get_classification_training_input_dir(),
                                                    ao_dir=training_input_dlg.get_AO_dir(),
                                                    test_dir=training_input_dlg.get_classification_test_input_dir(),
                                                    train_image_size=self.training_settings[
                                                        'classification_training_size'],
                                                    batch_size=self.training_settings['classification_batch_size'],
                                                    epochs=self.training_settings['classification_epochs'],
                                                    training_flag=False)

            ### start to load training data for label cycle GAN
            duplicate_list = read_csv_file('overlap_image_indices.csv', 0)
            small_training_labels = read_csv_file(
                os.path.join(training_input_dlg.get_classification_training_input_dir(),
                             'train_labels.csv'), 1)
            quality_scores = read_csv_file('image_scores.csv', 1)

            ### start to train cycle GAN
            self.training_settings['training_size'] = training_input_dlg.get_training_size()
            self.training_settings['batch_size'] = training_input_dlg.get_batch_size()
            self.training_settings['saving_interval'] = training_input_dlg.get_saving_interval()
            self.training_settings['epochs'] = training_input_dlg.get_epochs()

            gan_data = load_label_training_data(small_AO_dir=os.path.join(
                training_input_dlg.get_classification_training_input_dir(), 'AO_Images'),
                small_spectralis_dir=os.path.join(training_input_dlg.get_classification_training_input_dir(),
                                                  'Spectralis30_Images'), large_AO_dir=training_input_dlg.get_AO_dir(),
                large_spectralis_dir=training_input_dlg.get_spectralis_dir(),
                trained_image_size=self.training_settings['training_size'],
                nr_of_channels=1, duplicate_list=duplicate_list)

            AO_train_images = gan_data["AO images"]
            AO_train_labels = np.concatenate((small_training_labels, quality_scores))
            spectralis_train_images = gan_data["spectralis images"]

            AO_train_images = AO_train_images[AO_train_labels == training_input_dlg.get_fixed_label()]
            spectralis_train_images = spectralis_train_images[AO_train_labels == training_input_dlg.get_fixed_label()]

            ### start to train cycle GAN
            self.training_settings['training_size'] = training_input_dlg.get_training_size()
            self.training_settings['batch_size'] = training_input_dlg.get_batch_size()
            self.training_settings['saving_interval'] = training_input_dlg.get_saving_interval()
            self.training_settings['epochs'] = training_input_dlg.get_epochs()

            cycle_gan = CycleGAN(image_shape=(self.training_settings['training_size'],
                                              self.training_settings['training_size'], 1),
                                 epochs=self.training_settings['epochs'],
                                 saving_interval=self.training_settings['saving_interval'],
                                 batch_size=self.training_settings['batch_size'],
                                 cycle_loss_type=training_input_dlg.get_cycle_loss(),
                                 supervised_learning=True)
            cycle_gan.load_training_data1(spectralis_images=spectralis_train_images,
                                          ao_images=AO_train_images)
            cycle_gan.train_model(self)

    def train_supervised_cyclegan(self):
        training_input_dlg = SupervisedCycleTrainingInputDialog(self.framewidth, self.frameheight,
                                                      self.training_settings['training_size'],
                                                      self.training_settings['batch_size'],
                                                      self.training_settings['saving_interval'],
                                                      self.training_settings['epochs'], self)
        if training_input_dlg.exec():
            if training_input_dlg.get_spectralis_dir() is None or training_input_dlg.get_AO_dir() is None:
                display_error('Input error', 'Either spectralis or AO directories are not selected')
                return

            if training_input_dlg.get_training_label_dir() is None or \
                   not is_file_existed(os.path.join(training_input_dlg.get_training_label_dir(), 'train_labels.csv')):
                return

            AO_train_labels = read_csv_file(os.path.join(training_input_dlg.get_training_label_dir(),
                             'train_labels.csv'), 1)
            AO_train_labels = np.asarray(AO_train_labels)

            ### start to train cycle GAN
            self.training_settings['training_size'] = training_input_dlg.get_training_size()
            self.training_settings['batch_size'] = training_input_dlg.get_batch_size()
            self.training_settings['saving_interval'] = training_input_dlg.get_saving_interval()
            self.training_settings['epochs'] = training_input_dlg.get_epochs()

            gan_data = load_training_data(training_input_dlg.get_AO_dir(), training_input_dlg.get_spectralis_dir(),
                                          (self.training_settings['training_size'], self.training_settings['training_size']),
                                          1, self.training_settings['batch_size'])

            AO_train_images = gan_data["trainA_images"]
            spectralis_train_images = gan_data["trainB_images"]

            classification_cycle_gan = LabelCycleGAN(image_shape=(self.training_settings['training_size'],
                                                                  self.training_settings['training_size'], 1),
                                                     num_classes=3,
                                                     epochs=self.training_settings['epochs'],
                                                     saving_interval=self.training_settings['saving_interval'],
                                                     batch_size=self.training_settings['batch_size'],
                                                     cycle_loss_type=training_input_dlg.get_cycle_loss())
            classification_cycle_gan.load_training_data(spectralis_images=spectralis_train_images,
                                                        ao_images=AO_train_images, ao_labels=AO_train_labels)
            classification_cycle_gan.train_model(self)


    def train_cyclegan(self):

        training_input_dlg = CycleTrainingInputDialog(self.framewidth, self.frameheight,
                                                 self.training_settings['training_size'],
                                                 self.training_settings['batch_size'],
                                                 self.training_settings['saving_interval'],
                                                 self.training_settings['epochs'], self)

        if training_input_dlg.exec():
            if training_input_dlg.get_spectralis_dir() is None or training_input_dlg.get_AO_dir() is None:
                display_error('Input error', 'Either spectralis or AO directories are not selected')
                return

            ### start to train cycle GAN
            self.training_settings['training_size'] = training_input_dlg.get_training_size()
            self.training_settings['batch_size'] = training_input_dlg.get_batch_size()
            self.training_settings['saving_interval'] = training_input_dlg.get_saving_interval()
            self.training_settings['epochs'] = training_input_dlg.get_epochs()

            cycle_gan = CycleGAN(image_shape=(self.training_settings['training_size'],
                                              self.training_settings['training_size'], 1),
                                 epochs=self.training_settings['epochs'],
                                 saving_interval=self.training_settings['saving_interval'],
                                 batch_size=self.training_settings['batch_size'],
                                 cycle_loss_type= training_input_dlg.get_cycle_loss(),
                                 supervised_learning=True)
            cycle_gan.load_training_data(input_dir1=training_input_dlg.get_spectralis_dir(),
                                         input_dir2=training_input_dlg.get_AO_dir())
            cycle_gan.train_model(self)

    def test_gan(self):
        test_input_dlg = TestInputDialog(self.framewidth, self.frameheight)
        if test_input_dlg.exec():
            if test_input_dlg.get_AO_dir()[0] is None or test_input_dlg.get_spectralis_dir()[0] is None\
                    or test_input_dlg.get_AO_to_spectralis_weight() is None\
                    or test_input_dlg.get_spectralis_to_AO_weight() is None:
                display_error('Input error', 'Either spectralis or AO directories or training weights are not selected')
                return

            self._textbox.clear()
            self.write_text('Spectralis to AO weight path: {}\n'
                            'AO to spectralis weight path: {}\n'
                            'Input spectralis data directory: {}\n'
                            'Input AO data directory: {}\n'.format(test_input_dlg.get_spectralis_to_AO_weight(),
                                                                   test_input_dlg.get_AO_to_spectralis_weight(),
                                                                   test_input_dlg.get_spectralis_dir()[0],
                                                                   test_input_dlg.get_AO_dir()[0]))

            if test_input_dlg.get_method_type() == 0:
                cycle_gan = CycleGAN(image_shape=(self.training_settings['training_size'],
                                                  self.training_settings['training_size'], 1),
                                     saving_interval=self.training_settings['saving_interval'],
                                     epochs=self.training_settings['epochs'],
                                     batch_size=self.training_settings['batch_size'])

                cycle_gan.load_test_data(input_dir1=test_input_dlg.get_spectralis_dir()[0],
                                         input_dir2=test_input_dlg.get_AO_dir()[0])

                cycle_gan.test_model(test_input_dlg.get_spectralis_to_AO_weight(),
                                     test_input_dlg.get_AO_to_spectralis_weight(),
                                     test_input_dlg.get_spectralis_dir()[1],
                                     test_input_dlg.get_AO_dir()[1], self)
            elif test_input_dlg.get_method_type() == 1: ### label cycleGAN
                label_cycle_gan = LabelCycleGAN(image_shape=(self.training_settings['training_size'],
                                                             self.training_settings['training_size'], 1),
                                                num_classes=3,
                                                epochs=self.training_settings['epochs'],
                                                saving_interval=self.training_settings['saving_interval'],
                                                batch_size=self.training_settings['batch_size'])
                label_cycle_gan.load_test_data(input_dir1=test_input_dlg.get_spectralis_dir()[0],
                                         input_dir2=test_input_dlg.get_AO_dir()[0])
                label_cycle_gan.test_model(test_input_dlg.get_spectralis_to_AO_weight(),
                                           test_input_dlg.get_AO_to_spectralis_weight(),
                                           test_input_dlg.get_spectralis_dir()[1],
                                           test_input_dlg.get_AO_dir()[1], self)


    def write_text(self, str):
        self._textbox.insertPlainText(str)
        self._textbox.ensureCursorVisible()
        QtWidgets.QApplication.processEvents()

    def show_progress(self, val):
        self._progressbar.setValue(val)
        QtWidgets.QApplication.processEvents()