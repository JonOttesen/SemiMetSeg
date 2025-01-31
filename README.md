# SemiMetSeg

The GitHub repo for [*Semi-Supervised Learning Allows for Improved Segmentation With Reduced Annotations of Brain Metastases Using Multicenter MRI Data*]( https://doi.org/10.1002/jmri.29686). This repository provides the source code for the semi supervised methods, model weight and training schematic.

## Dataset

The models were trained on the "BrainMetShare" dataset available [here](https://aimi.stanford.edu/brainmetshare). The UCSF data is available [here](https://doi.org/10.1148/ryai.230126) and the Yale data is available [here](https://www.nature.com/articles/s41597-024-03021-9).

## Model Weights

The model weights can be downloaded [here](https://uio-my.sharepoint.com/:f:/g/personal/jonakri_uio_no/EuhkkormbL5DivWxpeQv5YoBPWtfPJfsuQ-AIZSyDpxdiw?e=FpVYWD)

### General Information

All models were trained on four sequences with the following input order: BRAVO, T1 post gd, T1 pre gd, and FLAIR. Note, the 3D model is trained to handle missing input sequences. In those cases, just have the channels corresponding to the missing sequences be zeroes and multiply the input with:
$$
\begin{align}
inp = inp\cdot\frac{1}{1-p},
\end{align}
$$
where p is the fraction of included sequences (p=0.75 when 3 sequences are included, this happens automatically).