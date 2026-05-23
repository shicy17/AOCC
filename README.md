# AOCC: Label-Free, Non-Monotonic Metrics for Event-Camera Denoising Evaluation 
# AOCC-flicker: Label-Free Evaluation of Flicker Noise Denoising for Event Cameras

Official implementation of the **Area of Continuous Contrast Curve (AOCC)** family of label-free, non-monotonic evaluation metrics for event-camera denoising.

This repository contains:
- **AOCC** — general denoising evaluation (IEEE TCSVT 2026)
- **AOCC-flicker** — extension for flicker-noise suppression evaluation (Under Review) <!-- TODO: update venue/status once accepted -->

---

## :star2: What's New

- **[2026/05]** Released **AOCC-flicker** — *"AOCC-flicker: Label-Free Evaluation of Flicker Noise Denoising for Event Cameras"*, an extension for evaluating flicker-noise suppression in event cameras. <!-- TODO: fill date when ready --> Highlights:
  - Frequency-domain residual based on dB peak-over-background prominence aggregated across spatial tiles.
  - Decoupled structure-preservation and flicker-suppression terms with a closed-form trade-off interval.
  - Provable invariance to uniform event thinning at leading order, closing a common loophole of count-based proxies.
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

### AOCC-flicker — Flicker-Suppression Evaluation

`aocc_flicker.py` implements the **global (whole-image) canonical configuration** used in the paper: tile-aggregated DFT, dB peak-over-background prominence, top-N raw-detected peaks, with no ROI annotation. The ROI-annotated variant is reserved for the ablation in the supplementary material and is not part of this script.

#### Input Format

For each evaluated sequence the script expects:
- **One** raw event file (no denoising applied).
- **One or more** denoised event files produced by the method(s) under evaluation (e.g. PFD variants).
- Optionally, an event file produced by a baseline such as EFR.

Each event file is a plain-text file with four numeric tokens per line; the script auto-detects `xypt` vs. `txyp` ordering.

<!-- TODO: add download link for flicker test data, e.g. one raw + several PFD variants of a single sequence -->

#### Run

Minimal invocation (1 raw + 4 PFD variants of a single sequence):

```bash
python aocc_flicker.py \
  --raw_file       path/to/sequence_raw.txt \
  --pfd_files      path/to/sequence_pfd_v1.txt \
                   path/to/sequence_pfd_v2.txt \
                   path/to/sequence_pfd_v3.txt \
                   path/to/sequence_pfd_v4.txt \
  --pfd_labels     1 2 3 4 \
  --output_csv     ./flicker_out.csv \
  --ccc_dir        ./flicker_diag \
  --width 1280 --height 720 \
  --microbin_us 500 \
  --n_residual_peaks 3 \
  --residual_gain 3.0 \
  --residual_zero_threshold 0.05 \
  --global_dft_plots --dft_fmax_hz 500 --dft_ref_freq_hz 100
```

To include an EFR baseline alongside the PFD variants, add `--efr_file path/to/sequence_efr.txt`.

#### Outputs

- `--output_csv`: per-method metrics (`structural_aocc`, `residual_ratio`, `db_peak_prominence`, `score`, etc.).
- `--ccc_dir`: diagnostic plots — per-tile flicker-purity heatmaps, global DFT panels, and tile-aggregate spectrum panels with the top-N detected peaks marked.

A full annotated example is included in the docstring at the top of `aocc_flicker.py`.

---

## :package: Requirements

- `numpy` ≥ 1.21.0
- `opencv-python` ≥ 4.5.0
- `tqdm` ≥ 4.62.0
- `matplotlib` ≥ 3.5.0
- `pandas` ≥ 1.3.0
- `scipy` ≥ 1.7.0

---

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
