#Contour Tracking Algorithms
This repository contains contour tracking algorithms designed for fitting line profiles from the center to the edge of either membrane-bound (nuclear membrane) or membraneless (nucleolus) organelles with a Gaussian shape. These algorithm classes, along with pre/post image processing functions, are encapsulated in the utils.py file.

The contour dynamics are analyzed by converting them to spatial frequency and fitting them to a mathematical model to extract bending rigidity and surface tension.

These codes were developed by An T. Pham as part of his postdoctoral work at Northwestern University.

If you use these codes in your research, please cite:

Pham, An T., et al. "Multiscale biophysical analysis of nucleolus disassembly during mitosis." Proceedings of the National Academy of Sciences 121.6 (2024): e2312250121. https://doi.org/10.1073/pnas.2312250121
