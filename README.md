#Contour Tracking Algorithms

This repository contains contour tracking algorithms designed for fitting line profiles from the center to the edge of either membrane-bound (nuclear membrane) or membraneless (nucleolus) organelles with a Gaussian shape. These algorithm classes, along with pre/post image processing functions, are encapsulated in the utils.py file.

The contour dynamics are analyzed by converting them to spatial frequency and fitting them to a mathematical model to extract bending rigidity and surface tension.

The input folder contains time-series data of the nuclear membrane and nucleolus, with a frame rate of approximately 17 frames per second. For demonstration purposes, each .tif stack in this repository contains 180 frames. The nuclear membrane is tagged with mRuby-LaminAC, while the nucleolus is tagged with mRuby-NPM1. All experiments were recorded at the equatorial plane of the nuclear membrane or the nucleolus.

These codes were developed by An T. Pham as part of his postdoctoral work at Northwestern University.

If you use these codes in your research, please cite:

Pham, An T., et al. "Multiscale biophysical analysis of nucleolus disassembly during mitosis." Proceedings of the National Academy of Sciences 121.6 (2024): e2312250121. https://doi.org/10.1073/pnas.2312250121
