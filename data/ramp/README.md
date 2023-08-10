# Ramp - draft
This file mocks the mnih README file (Massachusetts Roads Dataset), tentatively for now

## Usage
1. Download [pre-splitted training, validation and test sets](https://www.cs.toronto.edu/~vmnih/data/)
using instead the pre-splitted material fro fair-utils
2. Make sure the directory structure is:
we try to make the same here
```
mnih
│
└── train
│   │
│   └── map
│   │   │   ***.tif  <--- target
│   │   │   ...
│   │
│   └── sat
│	│   ***.tiff  <--- input
│	│   ...
│
└── valid
│   │
│   └── map
│   │   │   ***.tif
│   │   │   ...
│   │
│   └── sat
│	│   ***.tiff
│	│   ...
│
└── test
    │
    └── map
    │   │   ***.tif
    │   │   ...
    │
    └── sat
	│   ***.tiff
	│   ...
```
3. Edit `DATA_DIR` and `save_dir` in `preprocess.py`
4. Run `preprocess.py`

## Citation
@phdthesis{MnihThesis,
    author = {Volodymyr Mnih},
    title = {Machine Learning for Aerial Image Labeling},
    school = {University of Toronto},
    year = {2013}
}
