# YOLOv3 Coding
Building YOLOv3 networks model **from scratch** using `Pytorch`
It contains...
- Dataloader
- Model
- Train/Eval Logic
- Loss

## 1. Prepare the Dataset
### 1.1 Download Object Datasets
It requires **KITTI** object datasets, you can download [HERE](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
And download 2 items, which are,
- Basic object dataset([Download](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip))
- Training labels of object dataset([Download](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip))
### 1.2 Project Structure
The directory structures like below.
```
$YOUR_PROJECT_ROOT_DIR
├── dataset
│   ├── testing
│   │   ├── image_sets
│   │   ├── png_images
│   └── training
│       ├── annotations
│       ├── image_sets
│       ├── png_images
├── READ.md
├── src
└── yolov3_kitti.cfg
```
### 1.3 Unzip And Move Dataset
Unzip the files and move all images to project dataset folder like this,
```bash
$ mv YOUR_DATASET_LOCATION/data_object_image_2/training/image_2 \ YOUR_PROJECT_ROOT_DIR/dataset/training/image_sets

$ mv YOUR_DATASET_LOCATION/data_object_image_2/testing/image_2 \  YOUR_PROJECT_ROOT_DIR/dataset/testing/image_sets
```

### 1.4 Test DataLoader
```bash
python src/main.py --modein --cfg yolov3_kitti.cfg
```