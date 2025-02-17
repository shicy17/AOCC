# A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras
This is an official implementation of Area of Continuous Contrast Curve (AOCC) method for event cameras.

Event cameras are renowned for their high efficiency due to outputting a sparse, asynchronous stream of events. However, they are plagued by noisy events, especially in low light conditions. Denoising is an essential task for event cameras, but evaluating denoising performance is challenging. Label-dependent denoising metrics involve artificially adding noise to clean sequences, complicating evaluations. Moreover, the majority of these metrics are monotonic, which can inflate scores by removing substantial noise and valid events. To overcome these limitations, we propose the first label-free and non-monotonic evaluation metric, the area of the continuous contrast curve (AOCC), which utilizes the area enclosed by event frame contrast curves across different time intervals. This metric is inspired by how events capture the edge contours of scenes or objects with high temporal resolution. An effective denoising method removes noise without eliminating these edge-contour events, thus preserving the contrast of event frames. Consequently, contrast across various time ranges serves as a metric to assess denoising effectiveness. As the time interval lengthens, the curve will initially rise and then fall. The proposed metric is validated through both theoretical and experimental evidence.

****
# 👉Citation   

Citations are welcome, and if you use all or part of our codes in a published article or project, please cite: 

C. Shi et al., "A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 
BibTeX of the paper:  
```
@ARTICLE{shi2024labelfree,  
  author={Shi,Chenyang, Guo, Shasha, Wei, Boyi, Liu, Hanxiao, Zhang, Yibo, Song, Ningfang and Jin, Jing},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},   
  title={A Label-Free and Non-Monotonic Metric for Evaluating Denoising in Event Cameras},   
  year={2025},  
  volume={},  
  number={},  
  pages={},  
  doi={}}  
```

****

# Installation
Run the AOCC.py

****
