# Description

This is a software implementation for the paper "**Artificial intelligence assisted clinical fluorescence imaging achieves in vivo cellular resolution comparable to adaptive optics ophthalmoscopy**".

# System Requirements

### Prerequisites

- Windows 10
- NVIDIA GPU +CUDA (tested on NVIDIA TITAN V, CUDA 11.7)

# Demo 

### Test

- Run `python main.py` to test the model.

  

  <img src="assets/step 1_choose test.png" width="500" height="500" />

- To test the model, Click on the  `Test` button.

- In  `Open test directory ` tab, select the folder  `./data/` which contains two subfolders:  `AO_images`  and  `spectralis30_images` . 

  

  <img src="assets/step2-0.png" width="3000" height="500" />

- In  `Training weights ` tab, select the folder  `./saved_models/20200225-180644_labelcyclegan` .

- Click OK.

- The generated images are automatically saved in `./generate_images` .

# Example Images

- Conventional indocyanine green (ICG) image of the RPE cells 

  <img src="assets/conventional.png" width="512" height="512" />

- Stratified CycleGAN enhanced RPE cells (AI-ICG)

  <img src="assets/cycleGAN enhanced.png" width="512" height="512" />
  
- Adaptive optics image of RPE cells (AO-ICG, Ground truth)

  <img src="assets/ao.png" width="512" height="512" />

