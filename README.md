# Yolo2Bolo

This application is used for exporting Yolo V5, V6 and V7 object detection models for OAKs.

```shell
usage: main.py [-h] [-imgsz IMGSZ] -f FILE [-v {v5,v6,v7}]

options:
  -h, --help            show this help message and exit
  -imgsz IMGSZ, --imgsz IMGSZ
                        Integer for square shape, or width and height separated by space. Must be divisible by 32.
  -f FILE, --file FILE  Weights of a pre-trained model (.pt file), size needs to be smaller than 100Mb.
  -v {v5,v6,v7}, --version {v5,v6,v7}
                        Yolo Version
```

## Clone source

```shell
git clone --recursive https://github.com/richard-xx/yolo2blob.git
```

or 

```shell
git clone https://github.com/richard-xx/yolo2blob.git
cd yolo2blob
git submodule update --init --recursive
```

## Install Requirements

```shell
python -m pip install -r requirements.txt
```

## Usage

```shell
python main.py -f <path/to/model.pt> -imgsz <size> -v <version>
```

