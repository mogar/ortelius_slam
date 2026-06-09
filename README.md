# Ortelius SLAM

SLAM experiments using factor graphs (eventually with iSAM2).

## TODO

* fix graph manager update_graph method

## Setup

```bash
conda create --name ortelius
pip install -r requirements.txt
```

You'll also need to install gtsam and the Python wrappers (see [instructions](https://github.com/borglab/gtsam/)).

## Datasets

KAIST - assumes you have downloaded one of the [Urban datasets](https://sites.google.com/view/complex-urban-dataset/download-lidar-stereo?authuser=0).

