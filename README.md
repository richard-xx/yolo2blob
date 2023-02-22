# Yolo2Bolo

This application is used for exporting Yolo V5, V6 and V7 object detection models for OAKs.

```shell
usage: main.py [-h] [-imgsz IMGSZ] -f FILE [-v {v5,v5lite,v6,v7,v8}] [-sh {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}] [-ov {2021.2,2021.3,2021.4,2022.1}] [-dt {U8,FP16}]

optional arguments:
  -h, --help            show this help message and exit
  -imgsz IMGSZ, --imgsz IMGSZ
                        Integer for square shape, or width and height separated by space. Must be divisible by 32.
  -f FILE, --file FILE  Weights of a pre-trained model (.pt file), size needs to be smaller than 100Mb.
  -v {v5,v5lite,v6,v7,v8}, --version {v5,v5lite,v6,v7,v8}
                        Yolo Version
  -sh {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, --shaves {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
                        Specifies number of SHAVE cores that converted model will use
  -ov {2021.2,2021.3,2021.4,2022.1}, --openvino_version {2021.2,2021.3,2021.4,2022.1}
                        OpenVINO version to use for conversion
  -dt {U8,FP16}, --data_type {U8,FP16}
                        Specifies precision for all input layers of the network
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

