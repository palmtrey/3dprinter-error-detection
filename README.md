# 3D Printer Error Detection Experiment

Computer vision system for detecting layer shifts in desktop fused filament fabrication (D-FFF) 3D printers.

## Project Overview

This repository contains the code for my Honors capstone project at Clarkson University (2023) that implements an automated system for detecting layer shifts in 3D prints using computer vision and deep learning.

⚠️ **Note:** This repository is no longer actively maintained and is provided primarily for documentation and educational purposes. Running this code will require additional effort and likely modifications to work with current libraries.

## Background

Desktop 3D printers frequently encounter print failures that waste time and materials. One common failure is "layer shifting," which occurs when the printer's rubber belts slip over the stepper motor's drive gears during fast movements. Since these printers use open-loop control systems, they don't recognize when this happens, resulting in misaligned prints that are often unusable.

This project demonstrates how computer vision and artificial intelligence can be used to detect these failures automatically, making 3D printing more accessible by reducing the need for constant human monitoring.

## Key Features

- Custom dataset creation using a Raspberry Pi camera system and an Ender 3 V2
- G-code injection method to programmatically create layer shifts for training data
- ResNet18 convolutional neural network for image classification
- Real-time detection system for monitoring ongoing prints

## Stack

- **Python 3.10+**
- **Poetry** for dependency management
- **PyTorch** for neural network implementation
- **PyTorch Lightning** for training framework
- **OpenCV** for image processing
- **Raspberry Pi** for image capture
- **Weights & Biases** for experiment tracking

## Repository Structure

```
.
├── gcode_injection/             # G-code modification for simulating layer shifts
│   ├── inject.py                # Tool for G-code injection
│   └── sample_ender_correct_phil_28m.gcode  # Sample G-code file
├── model/                       # Neural network and associated code
│   ├── benchmark.py             # Performance benchmarking
│   ├── client_interface.py      # Client-side interface for detection
│   ├── create_split_file.py     # Dataset splitting utility
│   ├── dataloader.py            # Dataset loading and preprocessing
│   ├── eval.py                  # Model evaluation
│   ├── eval_delay.py            # Evaluating detection delay
│   ├── eval_visual.py           # Visual evaluation of results
│   ├── grid_imgs.py             # Grid visualization of images
│   ├── network.py               # Neural network architecture
│   ├── server_interface.py      # Server-side interface for detection
│   ├── test_crop.py             # Image cropping test
│   ├── train.py                 # Model training script
│   └── utils.py                 # Utility functions
├── sample_images/               # Sample images and metadata
│   └── ender_91/                # Sample print directory
│       ├── ender_91_*.jpg       # Sample images
│       ├── _center.json         # Centering information
│       ├── _meta.json           # Metadata
│       └── _shift.json          # Shift label information
├── poetry.lock                  # Poetry lock file
├── pyproject.toml               # Poetry dependency management
├── LICENSE                      # License file
└── README.md                    # This file
```

## Setup (Historical Reference)

This project used Poetry for dependency management. The original dependencies are listed in `pyproject.toml`.

For reference, the main dependencies were:
- torch
- torchvision
- pytorch-lightning
- opencv-python
- wandb (Weights & Biases)
- numpy
- pillow

## Results

- Validation accuracy: 91.4% on individual images
- Real-world detection accuracy: 83.3% (5 out of 6 test prints)
- Detection delay: 60-70 seconds

## Contact

For any questions about this project, feel free to reach out:

- **Author**: Cameron Palmer, [https://cameronmpalmer.com/](https://cameronmpalmer.com/)
- **Email**: [cameron@cameronmpalmer.com](mailto:cameron@cameronmpalmer.com)
- **LinkedIn**: [https://www.linkedin.com/in/cameronmpalmer/](https://www.linkedin.com/in/cameronmpalmer/)
- **GitHub**: [https://github.com/palmtrey](https://github.com/palmtrey)

## Resources
- This project's blog post [here](https://medium.com/p/f573c0025f0e/edit)
- Full report [here](https://cameronmpalmer.wordpress.com/wp-content/uploads/2025/04/honors-capstone-report.docx.pdf)
  
## License
This project is released under the MIT License - see the LICENSE file for details.
