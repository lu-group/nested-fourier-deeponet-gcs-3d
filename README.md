> Under Construction

# Nested Fourier-DeepONet for 3D geological carbon sequestration (GCS)

The data and code for the paper [J. E. Lee, M. Zhu, Z. Xi, K. Wang, Y. O. Yuan, & L. Lu. Efficient and generalizable nested Fourier-DeepONet for three-dimensional geological carbon sequestration. *Engineering Applications of Computational Fluid Mechanics*, 18 (1), 2024.](https://doi.org/10.1080/19942060.2024.2435457)

## Data
The full dataset for Nested Fourier-DeepONet is available on [OneDrive](https://yaleedu-my.sharepoint.com/:f:/r/personal/lu_lu_yale_edu/Documents/datasets/2024_Lee?csf=1&web=1&e=Rk14Bn). Download and put all files into `datasets` folder.

Steps to generate the full dataset for Nested Fourier-DeepONet:

- step 1: download [raw data](https://github.com/gegewen/nested-fno)
- step 2: run [file_config.sh](https://github.com/gegewen/nested-fno/blob/main/data_config/file_config.sh) to convert `.npy` file into `.pt` files
- step 3: create a new folder `datasets_nested_fno` under the main folder and put the files generated in step 2 into the folder `datasets_nested_fno`
- step 4: run [data_generation.py](https://github.com/MinZhu123/nested-fourier-deeponet-gcs-3d/blob/main/datasets/data_generation.py) to covert `.pt` files into our `.npz` files

## Code
to be uploaded

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{lee2024efficient},
author = {Jonathan E. Lee, Min Zhu, Ziqiao Xi, Kun Wang, Yanhua O. Yuan and Lu Lu},
title = {Efficient and generalizable nested Fourier-DeepONet for three-dimensional geological carbon sequestration},
journal = {Engineering Applications of Computational Fluid Mechanics},
volume = {18},
number = {1},
pages = {2435457},
year = {2024},
doi = {https://doi.org/10.1080/19942060.2024.2435457}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
