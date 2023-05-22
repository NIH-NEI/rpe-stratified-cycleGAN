import numpy as np
import os
from utils import is_file_existed

try:
    from PyQt5 import QtCore, QtWidgets, Qt, QtGui
except ImportError:
    pass

def display_error(msg, msg2):
    b = QtWidgets.QMessageBox()
    b.setIcon(QtWidgets.QMessageBox.Critical)

    b.setText(msg)
    b.setInformativeText(msg2)
    b.setWindowTitle("Error")
    b.setStandardButtons(QtWidgets.QMessageBox.Ok)

    b.exec_()

def display_option(msg, msg2):
    b = QtWidgets.QMessageBox()
    b.setIcon(QtWidgets.QMessageBox.Question)

    b.setText(msg)
    b.setInformativeText(msg2)
    b.setWindowTitle("Question")
    b.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
    buttonY = b.button(QtWidgets.QMessageBox.Ok)
    buttonY.setText('Use')
    buttonD = b.button(QtWidgets.QMessageBox.Discard)
    buttonD.setText('Discard')

    b.exec_()
    return b

class CycleTrainingInputDialog(QtWidgets.QDialog):
    def __init__(self, frame_width, frame_height, training_size, batch_size, saving_interval,
                 epochs, parent=None):
        super(CycleTrainingInputDialog, self).__init__(parent)
        self._training_input_dir_list = []
        self._setup_cycle_layout(training_size, batch_size, saving_interval, epochs)
        self.setMinimumSize(frame_width * 5 // 4, frame_height // 5)
        self.setWindowTitle('Please set cycle training input')

    def _setup_cycle_layout(self, training_size, batch_size, saving_interval, epochs):
        self._cycle_parameter_group = QtWidgets.QGroupBox('Cycle training parameters')
        cycle_input_layout = QtWidgets.QGridLayout()
        open_training_dir_button = QtWidgets.QPushButton('Open training directory')
        open_training_dir_button.clicked.connect(self._open_training_data)
        self._training_dir_editor = QtWidgets.QLineEdit('Training directory')
        self._training_dir_editor.setReadOnly(True)

        spectralis_training_input_label = QtWidgets.QLabel('Spectralis: ')
        self._spectralis_input_dir_box = QtWidgets.QComboBox()
        # self._trainingA_input_dir_box.currentIndexChanged.connect(self._select_trainingA)

        AO_input_label = QtWidgets.QLabel('AO: ')
        self._AO_input_dir_box = QtWidgets.QComboBox()
        # self._trainingB_input_dir_box.currentIndexChanged.connect(self._select_trainingB)

        cycle_loss_label = QtWidgets.QLabel('Cycle loss: ')
        self._cycle_loss_box = QtWidgets.QComboBox()
        self._cycle_loss_box.addItem('Image difference')
        self._cycle_loss_box.addItem('Image difference + gradients')
        self._cycle_loss_box.addItem('Image difference + gradients + Laplace')
        self._cycle_loss_box.setCurrentIndex(0)

        training_size_label = QtWidgets.QLabel('Training size: ')
        self._training_size_input = QtWidgets.QSpinBox()
        self._training_size_input.setRange(1, 2000)
        self._training_size_input.setSingleStep(1)
        self._training_size_input.setValue(training_size)

        batch_size_label = QtWidgets.QLabel('Batch size: ')
        self._batch_size_input = QtWidgets.QSpinBox()
        self._batch_size_input.setRange(1, 500)
        self._batch_size_input.setSingleStep(1)
        self._batch_size_input.setValue(batch_size)

        epochs_label = QtWidgets.QLabel('Epochs: ')
        self._epochs_input = QtWidgets.QSpinBox()
        self._epochs_input.setRange(1, 50000)
        self._epochs_input.setSingleStep(1)
        self._epochs_input.setValue(epochs)

        saving_interval_label = QtWidgets.QLabel('Saving interval: ')
        self._saving_interval_input = QtWidgets.QSpinBox()
        self._saving_interval_input.setRange(1, 5000)
        self._saving_interval_input.setSingleStep(1)
        self._saving_interval_input.setValue(saving_interval)

        cycle_input_layout.addWidget(open_training_dir_button, 0, 0)
        cycle_input_layout.addWidget(self._training_dir_editor, 0, 1)
        cycle_input_layout.addWidget(spectralis_training_input_label, 1, 0)
        cycle_input_layout.addWidget(self._spectralis_input_dir_box, 1, 1)
        cycle_input_layout.addWidget(AO_input_label, 2, 0)
        cycle_input_layout.addWidget(self._AO_input_dir_box, 2, 1)
        cycle_input_layout.addWidget(cycle_loss_label, 3, 0)
        cycle_input_layout.addWidget(self._cycle_loss_box, 3, 1)
        cycle_input_layout.addWidget(training_size_label, 4, 0)
        cycle_input_layout.addWidget(self._training_size_input, 4, 1)
        cycle_input_layout.addWidget(batch_size_label, 5, 0)
        cycle_input_layout.addWidget(self._batch_size_input, 5, 1)
        cycle_input_layout.addWidget(epochs_label, 6, 0)
        cycle_input_layout.addWidget(self._epochs_input, 6, 1)
        cycle_input_layout.addWidget(saving_interval_label, 7, 0)
        cycle_input_layout.addWidget(self._saving_interval_input, 7, 1)
        self._cycle_parameter_group.setLayout(cycle_input_layout)

        self._response_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self._response_buttons.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(self.accept)
        self._response_buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(self.reject)

        view_layout = QtWidgets.QGridLayout()
        view_layout.addWidget(self._cycle_parameter_group, 0, 0)
        view_layout.addWidget(self._response_buttons, 1, 0)
        self.setLayout(view_layout)

    def _open_training_data(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(Qt.QFileDialog.Directory)
        file_dialog.setOption(Qt.QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            training_root_dir = file_dialog.directory().absolutePath()
            self._training_dir_editor.setText(training_root_dir)
            sub_dirs = [f for f in os.listdir(training_root_dir) if os.path.isdir(os.path.join(training_root_dir, f))]

            #add file list to combobox
            self._spectralis_input_dir_box.clear()
            self._AO_input_dir_box.clear()
            self._training_input_dir_list.clear()
            for sub_dir in sub_dirs:
                self._spectralis_input_dir_box.addItem(sub_dir)
                self._AO_input_dir_box.addItem(sub_dir)
                self._training_input_dir_list.append(os.path.join(training_root_dir, sub_dir))

            spectralis_index = self._spectralis_input_dir_box.findText('Spectralis', QtCore.Qt.MatchContains)
            if spectralis_index >= 0:
                self._spectralis_input_dir_box.setCurrentIndex(spectralis_index)

            AO_index = self._AO_input_dir_box.findText('AO', QtCore.Qt.MatchContains)
            if AO_index >= 0:
                self._AO_input_dir_box.setCurrentIndex(AO_index)


    def get_spectralis_dir(self):
        if len(self._training_input_dir_list) != self._spectralis_input_dir_box.count() \
                or len(self._training_input_dir_list) == 0:
            return None
        else:
            return self._training_input_dir_list[self._spectralis_input_dir_box.currentIndex()]

    def get_AO_dir(self):
        if len(self._training_input_dir_list) != self._AO_input_dir_box.count() \
                or len(self._training_input_dir_list) == 0:
            return None
        else:
            return self._training_input_dir_list[self._AO_input_dir_box.currentIndex()]

    def get_training_size(self):
        return self._training_size_input.value()

    def get_batch_size(self):
        return self._batch_size_input.value()

    def get_saving_interval(self):
        return self._saving_interval_input.value()

    def get_epochs(self):
        return self._epochs_input.value()

    def get_cycle_loss(self):
        return self._cycle_loss_box.currentIndex()

class SupervisedCycleTrainingInputDialog(CycleTrainingInputDialog):
    def __init__(self, frame_width, frame_height, training_size, batch_size, saving_interval,
                 epochs, parent=None):
        super(SupervisedCycleTrainingInputDialog, self).__init__(frame_width, frame_height,
                                                                 training_size, batch_size, saving_interval,
                                                                 epochs, parent)
        self._traning_label_dir = None

        self._setup_supervised_cycle_layout()
        self.setWindowTitle('Please set supervised cycleGAN training input')

    def _setup_supervised_cycle_layout(self):
        training_label_input = QtWidgets.QLabel('Training labels: ')
        self._training_label_input_box = QtWidgets.QLineEdit('Image quality score file')
        self._training_label_input_box.setReadOnly(True)
        self._cycle_parameter_group.layout().addWidget(training_label_input, 8, 0, 1, 1)
        self._cycle_parameter_group.layout().addWidget(self._training_label_input_box, 8, 1, 1, 1)

    def _open_training_data(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(Qt.QFileDialog.Directory)
        file_dialog.setOption(Qt.QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            # training label dir is also root dir..
            self._traning_label_dir = file_dialog.directory().absolutePath()
            self._training_dir_editor.setText(self._traning_label_dir)
            sub_dirs = [f for f in os.listdir(self._traning_label_dir) if
                        os.path.isdir(os.path.join(self._traning_label_dir, f))]

            #add file list to combobox
            self._spectralis_input_dir_box.clear()
            self._AO_input_dir_box.clear()
            self._training_input_dir_list.clear()
            for sub_dir in sub_dirs:
                self._spectralis_input_dir_box.addItem(sub_dir)
                self._AO_input_dir_box.addItem(sub_dir)
                self._training_input_dir_list.append(os.path.join(self._traning_label_dir, sub_dir))

            spectralis_index = self._spectralis_input_dir_box.findText('Spectralis', QtCore.Qt.MatchContains)
            if spectralis_index >= 0:
                self._spectralis_input_dir_box.setCurrentIndex(spectralis_index)

            AO_index = self._AO_input_dir_box.findText('AO', QtCore.Qt.MatchContains)
            if AO_index >= 0:
                self._AO_input_dir_box.setCurrentIndex(AO_index)

            # load training label
            if is_file_existed(os.path.join(self._traning_label_dir, 'train_labels.csv')):
                self._training_label_input_box.setText(os.path.join(self._traning_label_dir, 'traning_label.csv'))

    def get_training_label_dir(self):
        return self._traning_label_dir

class LabelCycleTrainingInputDialog(CycleTrainingInputDialog):
    def __init__(self, frame_width, frame_height, training_size, batch_size, saving_interval,
                 epochs, classification_training_size, classification_batch_size,
                 classification_epochs, parent=None):
        super(LabelCycleTrainingInputDialog, self).__init__(frame_width, frame_height, training_size, batch_size,
                         saving_interval, epochs, parent)

        self._clasification_training_input_dir = None
        self._classification_test_input_dir = None

        self._setup_label_cycle_layout(classification_training_size, classification_batch_size,
                                       classification_epochs)
        self.setWindowTitle('Please set label cycleGAN training input')

    def _setup_label_cycle_layout(self, classification_training_size, classification_batch_size,
                                  classification_epochs):

        ##load the directory with reduced number of training images
        self._classification_parameter_group = QtWidgets.QGroupBox('Image classification training parameters')
        classification_input_layout = QtWidgets.QGridLayout()
        open_clasification_training_dir_button = QtWidgets.QPushButton('Open classification training directory')
        open_clasification_training_dir_button.clicked.connect(self._open_classification_training_data)
        self._classifcation_training_dir_editor = QtWidgets.QLineEdit('Image quality training directory')
        self._classifcation_training_dir_editor.setReadOnly(True)
        self._classification_test_dir_editor = QtWidgets.QLineEdit('Image quality test directory')
        self._classification_test_dir_editor.setReadOnly(True)

        classification_training_size_label = QtWidgets.QLabel('Training size: ')
        self._classification_training_size_input = QtWidgets.QSpinBox()
        self._classification_training_size_input.setRange(1, 2000)
        self._classification_training_size_input.setSingleStep(1)
        self._classification_training_size_input.setValue(classification_training_size)

        classification_batch_size_label = QtWidgets.QLabel('Batch size: ')
        self._classification_batch_size_input = QtWidgets.QSpinBox()
        self._classification_batch_size_input.setRange(1, 500)
        self._classification_batch_size_input.setSingleStep(1)
        self._classification_batch_size_input.setValue(classification_batch_size)

        classification_epochs_label = QtWidgets.QLabel('Epochs: ')
        self._classification_epochs_input = QtWidgets.QSpinBox()
        self._classification_epochs_input.setRange(1, 50000)
        self._classification_epochs_input.setSingleStep(1)
        self._classification_epochs_input.setValue(classification_epochs)

        classification_input_layout = QtWidgets.QGridLayout()
        classification_input_layout.addWidget(open_clasification_training_dir_button, 0, 0)
        classification_input_layout.addWidget(self._classifcation_training_dir_editor, 0, 1)
        classification_input_layout.addWidget(self._classification_test_dir_editor, 1, 1)
        classification_input_layout.addWidget(classification_batch_size_label, 2, 0)
        classification_input_layout.addWidget(self._classification_training_size_input, 2, 1)
        classification_input_layout.addWidget(classification_batch_size_label, 3, 0)
        classification_input_layout.addWidget(self._classification_batch_size_input, 3, 1)
        classification_input_layout.addWidget(classification_epochs_label, 4, 0)
        classification_input_layout.addWidget(self._classification_epochs_input, 4, 1)
        self._classification_parameter_group.setLayout(classification_input_layout)

        self.layout().removeWidget(self._response_buttons)
        self.layout().removeWidget(self._cycle_parameter_group)

        self.layout().addWidget(self._classification_parameter_group, 0, 0)
        self.layout().addWidget(self._cycle_parameter_group, 1, 0)
        self.layout().addWidget(self._response_buttons, 2, 0)

    def is_classification_training_dir(self, dir):
        sub_dirs = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
        sub_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        if 'AO_Images' not in sub_dirs or 'Spectralis30_Images' not in sub_dirs \
                or 'train_labels.csv' not in sub_files:
            return False
        else:
            return True

    def _open_classification_training_data(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(Qt.QFileDialog.Directory)
        file_dialog.setOption(Qt.QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            dir = file_dialog.directory().absolutePath()
            if not self.is_classification_training_dir(dir):
                display_error('Input wrong', 'This is not a training directory with traning labels for classification')
                return

            self._clasification_training_input_dir = dir
            self._classifcation_training_dir_editor.setText(dir)

            parent_dir = os.path.dirname(dir)
            self._classification_test_input_dir = os.path.join(parent_dir, 'test')
            self._classification_test_dir_editor.setText(self._classification_test_input_dir)


    def get_classification_training_size(self):
        return self._classification_training_size_input.value()

    def get_classification_batch_size(self):
        return self._classification_batch_size_input.value()

    def get_classification_epochs(self):
        return self._classification_epochs_input.value()

    def get_classification_training_input_dir(self):
        return self._clasification_training_input_dir

    def get_classification_test_input_dir(self):
        return self._classification_test_input_dir

class FixedLabelCycleTrainingInputDialog(LabelCycleTrainingInputDialog):
    def __init__(self, frame_width, frame_height, training_size, batch_size, saving_interval,
                 epochs, classification_training_size, classification_batch_size,
                 classification_epochs, parent=None):
        super(FixedLabelCycleTrainingInputDialog, self).__init__(frame_width, frame_height, training_size,
                                                                 batch_size, saving_interval, epochs,
                                                                 classification_training_size,
                                                                 classification_batch_size,
                                                                 classification_epochs, parent)

        self._setup_fixed_label_cycle_layout()
        self.setWindowTitle('Please set fixed label cycle training input')

    def _setup_fixed_label_cycle_layout(self):

        ##load the directory with reduced number of training images
        fixed_label_input = QtWidgets.QLabel('Image quality: ')
        self._fixed_label_combobox = QtWidgets.QComboBox()
        self._fixed_label_combobox.addItem('0')
        self._fixed_label_combobox.addItem('1')
        self._fixed_label_combobox.addItem('2')
        self._fixed_label_combobox.setCurrentIndex(2)

        self._classification_parameter_group.layout().addWidget(fixed_label_input, 5, 0)
        self._classification_parameter_group.layout().addWidget(self._fixed_label_combobox, 5, 1)

    def get_fixed_label(self):
        return self._fixed_label_combobox.currentIndex()

class TestInputDialog(QtWidgets.QDialog):
    def __init__(self, frame_width, frame_height, parent=None):
        super(TestInputDialog, self).__init__(parent)
        self._test_input_dir_list = []
        self._model_spectralis_to_AO_list = []
        self._model_AO_to_spectralis_list = []

        self._setup_layout()
        self.setMinimumSize(frame_width*5//4, frame_height*2//5)
        self.setWindowTitle('Please set test data and learning model paths')

    def _setup_layout(self):
        method_input_label = QtWidgets.QLabel('Methods: ')
        self._method_combobox = QtWidgets.QComboBox()
        self._method_combobox.addItem('CycleGAN')
        self._method_combobox.addItem('LabelCycleGAN')
        self._method_combobox.addItem('FixedLabelCycleGAN')
        # self._method_combobox.addItem('BicycleGAN')
        # self._method_combobox.addItem('MUNIT')
        self._method_combobox.setCurrentIndex(1)

        open_test_dir_button = QtWidgets.QPushButton('Open test directory')
        open_test_dir_button.setToolTip('Open a directory contains all AO and spectralis subdirectories')
        open_test_dir_button.clicked.connect(self._open_test_data)
        self._test_dir_editor = QtWidgets.QLineEdit('Test directory')
        self._test_dir_editor.setReadOnly(True)

        Spectralis_input_label = QtWidgets.QLabel('Spectralis: ')
        self._spectralis_input_dir_box = QtWidgets.QComboBox()

        AO_input_label = QtWidgets.QLabel('AO: ')
        self._AO_input_dir_box = QtWidgets.QComboBox()

        open_spectralis_to_AO_button = QtWidgets.QPushButton('Training weights')
        open_spectralis_to_AO_button.clicked.connect(self._open_training_weights)
        self._spectrails_to_AO_box = QtWidgets.QComboBox()
        self._AO_to_spectralis_box = QtWidgets.QComboBox()

        input_layout = QtWidgets.QGridLayout()
        input_layout.addWidget(method_input_label, 0, 0)
        input_layout.addWidget(self._method_combobox, 0, 1)
        input_layout.addWidget(open_test_dir_button, 1, 0)
        input_layout.addWidget(self._test_dir_editor, 1, 1)
        input_layout.addWidget(Spectralis_input_label, 2, 0)
        input_layout.addWidget(self._spectralis_input_dir_box, 2, 1)
        input_layout.addWidget(AO_input_label, 3, 0)
        input_layout.addWidget(self._AO_input_dir_box, 3, 1)
        input_layout.addWidget(open_spectralis_to_AO_button, 4, 0)
        input_layout.addWidget(self._spectrails_to_AO_box, 4, 1)
        input_layout.addWidget(self._AO_to_spectralis_box, 5, 1)


        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(self.accept)
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(self.reject)

        view_layout = QtWidgets.QGridLayout()
        view_layout.addLayout(input_layout, 0, 0)
        view_layout.addWidget(buttons, 1, 0)
        self.setLayout(view_layout)

    def _open_test_data(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(Qt.QFileDialog.Directory)
        file_dialog.setOption(Qt.QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            test_root_dir = file_dialog.directory().absolutePath()

            self._test_dir_editor.setText(test_root_dir)
            sub_dirs = [f for f in os.listdir(test_root_dir) if os.path.isdir(os.path.join(test_root_dir, f))]

            # add file list to combobox
            self._spectralis_input_dir_box.clear()
            self._AO_input_dir_box.clear()
            self._test_input_dir_list.clear()
            for sub_dir in sub_dirs:
                self._spectralis_input_dir_box.addItem(sub_dir)
                self._AO_input_dir_box.addItem(sub_dir)
                self._test_input_dir_list.append(os.path.join(test_root_dir, sub_dir))

            spectralis_index = self._spectralis_input_dir_box.findText('Spectralis', QtCore.Qt.MatchContains)
            if spectralis_index >= 0:
                self._spectralis_input_dir_box.setCurrentIndex(spectralis_index)

            AO_index = self._AO_input_dir_box.findText('AO', QtCore.Qt.MatchContains)
            if AO_index >= 0:
                self._AO_input_dir_box.setCurrentIndex(AO_index)

    def _assign_combobox_items(self, weight_dir, contained_str):
        weight_files = [f for f in os.listdir(weight_dir) \
                        if os.path.isfile(os.path.join(weight_dir, f)) \
                        and os.path.splitext(f)[1] == '.hdf5' \
                        and contained_str in f]

        if len(weight_files) == 0:
            display_error('Input wrong', 'No weight files are found in the current directory\n')
            return None, -1

        epoch_list = []
        for f in weight_files:
            f_base = os.path.basename(f)
            f_base_no_extension = os.path.splitext(f_base)[0]
            f_components = f_base_no_extension.split('_')
            epoch_list.append(int(f_components[-1]))

        return weight_files, epoch_list.index(max(epoch_list))


    def _open_training_weights(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(Qt.QFileDialog.Directory)
        file_dialog.setOption(Qt.QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            weight_dir = file_dialog.directory().absolutePath()
            weight_files, max_index = self._assign_combobox_items(weight_dir, 'G_A2B_model_weights')
            if weight_files is None or max_index == -1:
                return

            self._spectrails_to_AO_box.clear()
            for f in weight_files:
                self._spectrails_to_AO_box.addItem(f)
                self._model_spectralis_to_AO_list.append(os.path.join(weight_dir, f))
            self._spectrails_to_AO_box.setCurrentIndex(max_index)

            weight_files, max_index = self._assign_combobox_items(weight_dir, 'G_B2A_model_weights')
            if weight_files is None or max_index == -1:
                return

            self._AO_to_spectralis_box.clear()
            for f in weight_files:
                self._AO_to_spectralis_box.addItem(f)
                self._model_AO_to_spectralis_list.append(os.path.join(weight_dir, f))
            self._AO_to_spectralis_box.setCurrentIndex(max_index)

    def get_spectralis_dir(self):
        if len(self._test_input_dir_list) != self._spectralis_input_dir_box.count() \
                or len(self._test_input_dir_list) == 0:
            return None, None
        else:
            return self._test_input_dir_list[self._spectralis_input_dir_box.currentIndex()], \
                   self._spectralis_input_dir_box.currentText()

    def get_AO_dir(self):
        if len(self._test_input_dir_list) != self._AO_input_dir_box.count() \
            or len(self._test_input_dir_list) == 0:
            return None, None
        else:
            return self._test_input_dir_list[self._AO_input_dir_box.currentIndex()], \
                   self._AO_input_dir_box.currentText()

    def get_spectralis_to_AO_weight(self):
        if len(self._model_spectralis_to_AO_list) != self._spectrails_to_AO_box.count() \
            or len(self._model_spectralis_to_AO_list) == 0:
            return None
        else:
            return self._model_spectralis_to_AO_list[self._spectrails_to_AO_box.currentIndex()]

    def get_AO_to_spectralis_weight(self):
        if len(self._model_AO_to_spectralis_list) != self._AO_to_spectralis_box.count() \
            or len(self._model_AO_to_spectralis_list) == 0:
            return None
        else:
            return self._model_AO_to_spectralis_list[self._AO_to_spectralis_box.currentIndex()]

    def get_method_type(self):
        return self._method_combobox.currentIndex()