# RSLO
The code for our work **"Robust Self-supervised LiDAR Odometry via Representative Structure Discovery and 3D Inherent Error Distribution Modeling"** 
<!-- will be released in this repo. -->
![demo_vid](demo/output.gif)
<!-- ## News
- **2021-10-7** The code will be ready in few days.   -->


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
The trained models on the KITTI dataset have been uploaded to the [OneDrive](https://1drv.ms/u/s!AgP7bY0L6pvta-AeCK1tFxJrn-8?e=1hYWzy). You can download them and put them into the directory "weights" for testing. 

```
export PYTHONPATH="$PROJECT_ROOT_PATH:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT_PATH/rslo:$PYTHONPATH"
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
Note that you need to specify the PROJECT_ROOT_PATH, i.e. the absolute directory of the project folder "RSLO" and modify the path to the created data, i.e. all.h5, in the configuration file kitti_eval_ours.prototxt before running the above commands. A bash script "script/eval_ours.sh" is provided for reference. 

## Training from Scratch
Please see [training](./doc/train.md) for more details.

## TODO List and ETA
- [x] Inference code and pretrained models (9/10/2022)
- [ ] Training code (expected after CVPR 2022)
- [ ] Code cleaning and refactor (expected after CVPR 2022)


## Acknowledgments
We thank for the open-sourced codebases [spconv](https://github.com/traveller59/spconv) and [second](https://github.com/traveller59/second.pytorch) 
## Copyright
```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


