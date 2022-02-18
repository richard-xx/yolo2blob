import sys
sys.path.append("yolov5")
sys.path.append("./yolo/yolov5")

import torch
import json
import warnings
from yolov5.models.experimental import attempt_load
from yolov5.models.common import Conv
from yolov5.models.yolo import Detect
import torch.nn as nn
import onnx
import onnxsim
import mo.main as model_optimizer
import subprocess
import blobconverter
import numpy as np
import openvino.inference_engine as ie
from zipfile import ZipFile
import os


DIR_TMP = "./tmp"

class YoloV5Exporter:

    def __init__(self, weights_path, imgsz):

        # set up variables
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.model_name = weights_path.split(".")[0]

        # load the model
        self.load_model()

        # set up file paths
        self.f_onnx = None
        self.f_simplified = None
        self.f_bin = None
        self.f_xml = None
        self.f_mapping = None
        self.f_blob = None
        self.f_json = None
        self.f_zip = None

    
    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.weights_path)  # load FP32 model

        # check num classes and labels
        assert model.nc == len(model.names), f'Model class count {model.nc} != len(names) {len(model.names)}'

        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")
        
        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, 'forward_export'):
                    m.forward = m.forward_export  # assign custom forward (optional)

        self.model = model           

    def export_onnx(self):
        # export onnx model
        self.f_onnx = f"{DIR_TMP}/{self.model_name}.onnx"
        im = torch.zeros(1, 3, *self.imgsz)#.to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(self.model, im, self.f_onnx, verbose=False, opset_version=12,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=['output'],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # simplify the moodel
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'

        # add named sigmoids for prunning in OpenVINO
        conv_indices = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Conv" in n.name:
                conv_indices.append(i)

        input1, input2, input3 = conv_indices[-3:]

        sigmoid1 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input1].output[0]],
            outputs=['output1_yolov5'],
        )

        sigmoid2 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input2].output[0]],
            outputs=['output2_yolov5'],
        )

        sigmoid3 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input3].output[0]],
            outputs=['output3_yolov5'],
        )

        onnx_model.graph.node.append(sigmoid1)
        onnx_model.graph.node.append(sigmoid2)
        onnx_model.graph.node.append(sigmoid3)

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = f"{DIR_TMP}/{self.model_name}-simplified.onnx"
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_openvino(self):

        if self.f_simplified is None:
            self.export_onnx()

        # export to OpenVINO and prune the model in the process
        cmd = f"mo --input_model {self.f_simplified} " \
        f"--output_dir {DIR_TMP} " \
        f"--model_name {self.model_name} " \
        '--data_type FP16 ' \
        '--reverse_input_channel ' \
        '--scale 255 ' \
        '--output "output1_yolov5,output2_yolov5,output3_yolov5"'

        subprocess.check_output(cmd, shell=True)

        # set paths
        self.f_xml = f"{DIR_TMP}/{self.model_name}.xml"
        self.f_bin = f"{DIR_TMP}/{self.model_name}.bin"
        self.f_mapping = f"{DIR_TMP}/{self.model_name}.mapping"

        return self.f_xml, self.f_mapping, self.bin

    def export_blob(self):

        if self.f_xml is None or self.f_bin is None:
            self.export_openvino()
        
        # export blob from generate bin and xml
        blob_path = blobconverter.from_openvino(
            xml=self.f_xml,
            bin=self.f_bin,
            data_type="FP16",
            shaves=6,
            version="2021.4",
            use_cache=False,
            output_dir=DIR_TMP
        )

        self.f_blob = blob_path

        return blob_path

    def export_json(self):

        # load json template
        f = open("./yolo/json/yolov5.json")
        content = json.load(f)

        # generate anchors and sides
        anchors, sides = [], []
        m = self.model.module.model[-1] if hasattr(self.model, 'module') else self.model.model[-1]
        for i in range(3):
            sides.append(m.anchor_grid[i].size()[2])
            for j in range(3):
                anchors.extend(m.anchor_grid[i][0, j, 0, 0].numpy())
        anchors = [float(x) for x in anchors]
        sides.sort()

        # generate masks
        masks = dict()
        for i, num in enumerate(sides[::-1]):
            masks[f"side{num}"] = list(range(i*3, i*3+3))

        # set parameters
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        content["nn_config"]["NN_specific_metadata"]["classes"] = self.model.nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        content["mappings"]["labels"] = self.model.names

        # save json
        f_json = f"{DIR_TMP}/{self.model_name}.json"
        with open(f_json, 'w') as outfile:
            json.dump(content, outfile)

        self.f_json = f_json

        return self.f_json

    def make_zip(self):
        # create a ZIP folder
        if self.f_simplified is None:
            self.export_onnx()
        
        if self.f_xml is None:
            self.export_openvino()

        if self.f_blob is None:
            self.export_blob()
        
        if self.f_json is None:
            self.export_json()

        f_zip = f"{DIR_TMP}/{self.model_name}.zip"
        
        zip_obj = ZipFile(f_zip, 'w')
        zip_obj.write(self.f_simplified)
        zip_obj.write(self.f_xml)
        zip_obj.write(self.f_bin)
        zip_obj.write(self.f_blob)
        zip_obj.write(self.f_json)
        zip_obj.close()

        self.f_zip = f_zip
        return f_zip

    def clear(self):
        
        # remove all files except zip
        if self.f_json is not None:
            os.remove(self.f_json)
        if self.f_blob is not None:
            os.remove(self.f_blob)
        if self.f_bin is not None:
            os.remove(self.f_bin)
        if self.f_xml is not None:
            os.remove(self.f_xml)
        if self.f_mapping is not None:
            os.remove(self.f_mapping)
        if self.f_onnx is not None:
            os.remove(self.f_onnx)
        if self.f_simplified is not None:
            os.remove(self.f_simplified)