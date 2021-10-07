# RSLO
The code for our work **"Robust Self-supervised LiDAR Odometry via Representative Structure Discovery and 3D Inherent Error Distribution Modeling"** 
<!-- will be released in this repo. -->

## News
- **2021-10-7** The code will be ready in few days.  

## Installation 
As the dependencies is complex, a dockerfile has been provide. You need to install [docker](https://docs.docker.com/get-docker/) and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) first and then set up the docker image and start up a container with the following commands: 

```
cd RSLO
sudo docker build -t rslo .    
sudo docker run  -it  --runtime=nvidia --ipc=host  --volume="HOST_VOLUME_YOU_WANT_TO_MAP:DOCKER_VOLUME"  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1  rslo bash

```

## Data Preparation
You need to download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and unzip them into the below directory structures. 
```
./kitti/dataset
|──sequences
|    ├── 00/           
|    |   ├── calib.txt	
|    │   ├── velodyne/	
|    |   |	├── 000000.bin
|    |   |	├── 000001.bin
|    |   |	└── ...
|    ├── 01/ 
|    |   ...
|    └── 21/
└──poses
    |──00.txt
    |──01.txt
    |    ...
    └──10.txt

```
Then, create hdf5 data with 
```
python script create_hdf5.py ./kitti/dataset ./kitti/dataset/all.h5
```

## Test with the Pretrained Models
The trained models on the KITTI dataset have been uploaded to the [OneDrive](). You can download them and put them into the directory "weights" for testing. 

```
export PYTHON_PATH="$PROJECT_ROOT_PATH/RSLO:$PYTHON_PATH"
export PYTHON_PATH="$PROJECT_ROOT_PATH/RSLO:$PYTHON_PATH"

python -u  $PROJECT_ROOT_PATH/evaluate.py multi_proc_eval \
        --config_path $PROJECT_ROOT_PATH/config/kitti_eval_ours.prototxt \
        --model_dir ./outputs/ \
        --use_dist True \
        --gpus_per_node 1 \
        --use_apex True \
        --world_size 1 \
        --dist_port 20000 \
        --pretrained_path $PROJECT_ROOT_PATH/weights/ours.tckpt \
        --refine False \
```
Note that you need to specify the PROJECT_ROOT_PATH, i.e. the absolute directory of the project folder "RSLO" lies in, before running the above commands.  

