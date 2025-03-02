# MSCSCNet
## Usage
Please follow the official [MMsegmentation](https://mmsegmentation.readthedocs.io/en/latest/overview.html)

## Dataset
### iSAID
The data images could be download from DOTA-v1.0 (train/val/test)

The data annotations could be download from iSAID (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have semantic segmentation) in Aerial Images.

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.
```
├── data
│   ├── iSAID
│   │   ├── train
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
│   │   │   │   ├── part3.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── val
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── test
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
```
python tools/dataset_converters/isaid.py /path/to/iSAID
