# A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras
This is an official implementation of Area of Continuous Contrast Curve (AOCC) method for event cameras.

# Abstract
Event cameras are renowned for their high efficiency due to outputting a sparse, asynchronous stream of events. However, they are plagued by noisy events, especially in low light conditions. Denoising is an essential task for event cameras, but evaluating denoising performance is challenging. Label-dependent denoising metrics involve artificially adding noise to clean sequences, complicating evaluations. Moreover, the majority of these metrics are monotonic, which can inflate scores by removing substantial noise and valid events. To overcome these limitations, we propose the first label-free and non-monotonic evaluation metric, the area of the continuous contrast curve (AOCC), which utilizes the area enclosed by event frame contrast curves across different time intervals. This metric is inspired by how events capture the edge contours of scenes or objects with high temporal resolution. An effective denoising method removes noise without eliminating these edge-contour events, thus preserving the contrast of event frames. Consequently, contrast across various time ranges serves as a metric to assess denoising effectiveness. As the time interval lengthens, the curve will initially rise and then fall. The proposed metric is validated through both theoretical and experimental evidence.


# 👉Citation   

Citations are welcome, and if you use all or part of our codes in a published article or project, please cite: 

C. Shi et al., "A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2025.3598329.

BibTeX of the paper:  
```
@ARTICLE{10804847,
  author={Shi, Chenyang and Guo, Sha Sha and Wei, Boyi and Liu, Hanxiao and Zhang, Yibo and Song, Ningfang and Jin, Jing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras}, 
  year={},
  volume={35},
  number={},
  pages={},
  doi={10.1109/TCSVT.2025.3598329}}
```

### Prerequisites

Ensure you have Python 3.7 or higher installed on your system.

### Install Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/shicy17/AOCC
   cd AOCC
   ```

2. Install the required packages using pip:
   ```bash
   pip install -r requirements_txt.txt
   ```

   Alternatively, you can install the dependencies manually:
   ```bash
   pip install numpy>=1.21.0 opencv-python>=4.5.0 tqdm>=4.62.0 matplotlib>=3.5.0 pandas>=1.3.0 scipy>=1.7.0
   ```

## Usage

### Download Test Data

Before running the code, download the required test data file:

1. Download the test data from: [https://bhpan.buaa.edu.cn/link/AA710008ED8DAD4C39BE571181B9D4DBE8](https://bhpan.buaa.edu.cn/link/AA710008ED8DAD4C39BE571181B9D4DBE8)
2. Extract and place the file `f171hz_fla.txt` in the project root directory

### Run the Code

Run the main AOCC implementation:
```bash
python AOCC.py
```

**Note:** Ensure the test data file `f171hz_fla.txt` is in the same directory as `AOCC.py` before running the code.

## Requirements

The code requires the following Python packages:
- numpy (≥1.21.0)
- opencv-python (≥4.5.0)
- tqdm (≥4.62.0)
- matplotlib (≥3.5.0)
- pandas (≥1.3.0)
- scipy (≥1.7.0)

****
