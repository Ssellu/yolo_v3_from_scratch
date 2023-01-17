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
$ python src/main.py --mode train --cfg yolov3_kitti.cfg
```

## Image Augmentation
**Image Augmentation** is converting images into a new, much larger amount of images slighty altered.
This model uses [imgaug](https://www.github.com/aleju/imgaug) opensource library. To download the library,
```bash
pip install imgaug
```

```python
if is_train:
    data_transform = tf.Compose(
        [AbsoluteLabels(),
         DefaultAug(),
         RelativeLabels(),
         ResizeImage(new_size=(cfg_param['width'], cfg_param['height'])),
         ToTensor(),
        ])
```

## Buildling the Model
![img](https://miro.medium.com/max/720/1*d4Eg17IVJ0L41e7CTWLLSg.webp)
