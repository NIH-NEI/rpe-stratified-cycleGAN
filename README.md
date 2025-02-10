# Description

This is an implementation of the paper "Graded Image Generation Using Stratified CycleGAN".

# System Requirements

### Prerequisites

- Windows 10
- NVIDIA GPU +CUDA (tested on NVIDIA TITAN V, CUDA 11.7)

# Demo

### Test

- Run `python main.py` to test the model.

  ![gui] (assets/step 1_choose test.PNG)

- To test the model, Click on the  `Test` button.

- In  `Open test directory ` tab, select the folder  `./data/` which contains two subfolders:  `AO_images`  and  `spectralis30_images` . 

  ![gui] (assets/step 1_choose test.PNG)

- In  `Training weights ` tab, select the folder  `./saved_models/20200225-180644_labelcyclegan` .

- Click OK.

- The generated images are automatically saved in `./generate_images` .

  



