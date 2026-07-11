# AOCC: Label-Free, Non-Monotonic Metrics for Event-Camera Denoising Evaluation 

Official implementation of the **Area of Continuous Contrast Curve (AOCC)** family of label-free, non-monotonic evaluation metrics for event-camera denoising.

This repository contains:
- **AOCC** — general denoising evaluation (IEEE TCSVT 2026)

---

## :star2: What's New
- **[2026/01]** AOCC published in IEEE TCSVT, vol. 36, no. 1, pp. 669–684.

---

## :scroll: Abstract

Event cameras are renowned for their high efficiency due to outputting a sparse, asynchronous stream of events. However, they are plagued by noisy events, especially in low light conditions. Denoising is an essential task for event cameras, but evaluating denoising performance is challenging. Label-dependent denoising metrics involve artificially adding noise to clean sequences, complicating evaluations. Moreover, the majority of these metrics are monotonic, which can inflate scores by removing substantial noise and valid events. To overcome these limitations, we propose the first label-free and non-monotonic evaluation metric, the **area of the continuous contrast curve (AOCC)**, which utilizes the area enclosed by event frame contrast curves across different time intervals. This metric is inspired by how events capture the edge contours of scenes or objects with high temporal resolution. An effective denoising method removes noise without eliminating these edge-contour events, thus preserving the contrast of event frames. Consequently, contrast across various time ranges serves as a metric to assess denoising effectiveness. As the time interval lengthens, the curve will initially rise and then fall. The proposed metric is validated through both theoretical and experimental evidence.

---

## :eyes: Demonstration

Experimental results of AOCC for label-free event-camera denoising evaluation.

| CCC | AOCC Value Curves | Label-Dependent Parameters |
|:---:|:---:|:---:|
| <img width="300" alt="qmlpf_3hz_legendfree_page_1" src="https://github.com/user-attachments/assets/a9ddc857-eee6-43b6-b24a-1d59aa5deb40" /> | <img width="300" alt="qmlpf_3hz_aocc_page_1" src="https://github.com/user-attachments/assets/44dd82a7-77ef-4efb-a80d-08730e5d4ca8" /> | <img width="300" alt="other_3hz_page_1" src="https://github.com/user-attachments/assets/f3f1492d-e19d-43eb-ab86-c7cdb02c213e" /> |

- **Figure 1:** CCC obtained by scanning the QMLPF denoising method under various parameter conditions.
- **Figure 2:** AOCC value curves corresponding to the CCC curves in Figure 1.
- **Figure 3:** Label-dependent denoising parameters under different parameter conditions.

<!-- TODO: optionally add an AOCC-flicker demo figure row here -->

---

## :bulb: Key Contributions

- **AOCC** achieves non-monotonicity by construction, making it the **first truly effective label-free non-monotonic evaluation metric for event-camera denoising**. It removes the dependency on ground-truth labels while remaining robust to over-aggressive denoisers that delete valid events along with noise — a failure mode that monotone count-based metrics cannot detect.
- **AOCC-flicker** extends this framework to the flicker-suppression setting, combining a structurally normalized spatial contrast term with a dB peak-over-background frequency residual that is provably invariant to uniform event thinning at leading order.

---

## :sparkles: Installation

Python 3.7 or higher is required.

1. Clone this repository:
   ```bash
   git clone https://github.com/shicy17/AOCC
   cd AOCC
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_txt.txt
   ```

   Or manually:
   ```bash
   pip install numpy>=1.21.0 opencv-python>=4.5.0 tqdm>=4.62.0 \
               matplotlib>=3.5.0 pandas>=1.3.0 scipy>=1.7.0
   ```

---

## :relaxed: Usage

### AOCC — General Denoising Evaluation

#### Download Test Data

1. Download the test data from: [https://bhpan.buaa.edu.cn/link/AA710008ED8DAD4C39BE571181B9D4DBE8](https://bhpan.buaa.edu.cn/link/AA710008ED8DAD4C39BE571181B9D4DBE8)
2. Extract and place `f171hz_fla.txt` in the project root directory.

#### Run

```bash
python AOCC.py
```

> **Note:** Ensure `f171hz_fla.txt` is in the same directory as `AOCC.py` before running.


---

## :package: Requirements

- `numpy` ≥ 1.21.0
- `opencv-python` ≥ 4.5.0
- `tqdm` ≥ 4.62.0
- `matplotlib` ≥ 3.5.0
- `pandas` ≥ 1.3.0
- `scipy` ≥ 1.7.0

---

## :rocket: C++ Implementation (Faster)

For large-scale batch evaluation, a standalone C++ implementation of **AOCC** is provided. It reproduces the same continuous contrast curve (CCC) and AOCC computation as `AOCC.py`, while running substantially faster — useful when sweeping many sequences or using a fine interval grid. It depends only on OpenCV and carries no Python runtime overhead.

> This C++ build covers **AOCC** (general denoising evaluation) only. It is not the AOCC-flicker variant.

### Build

Requirements:
- A C++17 compiler (for `std::filesystem`)
- OpenCV 4

```bash
g++ -Wall -Wextra -O3 AOCC.cpp -o AOCC $(pkg-config --cflags --libs opencv4)
```

> Use `-O3` for best throughput; switch to `-g3` only while debugging.

### Configure

Run parameters are set at the top of `main()` — edit them before building:

| Parameter | Meaning | Default |
|---|---|---|
| `input_folder` | Directory of input `.txt` event files | — |
| `results_csv_path` | Output directory for per-file CCC csv files | — |
| `save_directory` | Output path for the AOCC summary csv | — |
| `image_output_dir` | Output directory for accumulation frames and CCC plots | — |
| `width`, `height` | Sensor resolution | `1280`, `720` |
| `min_interval`, `max_interval`, `step` | Time-interval sweep in microseconds | `4000`, `50001`, `1000` |
| `min_value`, `max_value` | AOCC integration bounds in microseconds | `0`, `max_interval - 1` |

### Input Format

Each input is a plain-text file with one event per line, space-separated as:

```
x y p t
```

An optional fifth column is treated as a label; when present, only events with label `1` are kept. A commented `t x y p` ordering is available in `read_events_from_txt` if your data uses that layout.

### Run

```bash
./AOCC
```

All `.txt` files in `input_folder` are processed in sorted order.

### Outputs

- **Per-file CCC** — `<results_csv_path>/<name>_ccc.csv` with columns `Interval (us)`, `Mean Contrast`, `Median Contrast`, `RMS Contrast`.
- **AOCC summary** — `<save_directory>` with columns `Filename`, `Area Under Curve`, one row per sequence.
- **Diagnostics** — accumulation frames and a CCC curve plot with the AOCC region shaded, under `<image_output_dir>`.

## :books: Citation

If you use **AOCC**, please cite:

```bibtex
@ARTICLE{shi2025label,
  author={Shi, Chenyang and Guo, Shasha and Wei, Boyi and Liu, Hanxiao and Zhang, Yibo and Song, Ningfang and Jin, Jing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras},
  year={2026},
  volume={36},
  number={1},
  pages={669-684},
  keywords={Noise reduction;Noise;Cameras;Measurement;Videos;Hardware;Voltage control;Retina;Vision sensors;Streaming media;Event cameras;denoising evaluation metric;label-free;non-monotonic},
  doi={10.1109/TCSVT.2025.3598329}
}
```

---

## :handshake: Contact

For questions, collaboration on flicker-aware evaluation, or access to additional benchmark sequences, please open an issue or contact the first author.
