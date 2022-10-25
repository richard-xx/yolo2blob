# coding=utf-8

import sys

sys.path.append("./yolo/YOLOv6")

import torch
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import onnx
from exporter import Exporter

import numpy as np
import onnxsim
import subprocess

from yolo.detect_head import DetectV2, DetectV1
import blobconverter

DIR_TMP = "./tmp"
R1_VERSION = 1
R2_VERSION = 2


class YoloV6Exporter(Exporter):
    def __init__(self, conv_path, weights_filename, imgsz, version):
        super().__init__(conv_path, weights_filename, imgsz, version)
        self.load_model()

    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(
            str(self.weights_path.resolve()),
            map_location="cpu",
            inplace=True,
            fuse=True,
        )  # load FP32 model

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        if not hasattr(model.detect, "obj_preds"):
            self.selected_release = R2_VERSION
            model.detect = DetectV2(model.detect)
        else:
            self.selected_release = R1_VERSION
            model.detect = DetectV1(model.detect)

        self.num_branches = len(model.detect.grid)

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")

        model.eval()
        self.model = model

    def get_onnx_r2(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(
            1, 3, *self.imgsz[::-1]
        )  # .to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
            dynamic_axes=None,
        )

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel
        return onnxsim.simplify(model_onnx)

    def export_onnx(self):
        if self.selected_release == R1_VERSION:
            onnx_model, check = self.get_onnx()
            assert check, "assert check failed"
            # get contacts ready for parsing
            conc_idx = []
            for i, n in enumerate(onnx_model.graph.node):
                if "Concat" in n.name:
                    conc_idx.append(i)

            outputs = conc_idx[-(self.num_branches + 1) : -1]

            for i, idx in enumerate(outputs):
                onnx_model.graph.node[idx].name = f"output{i+1}_yolov6"
        else:
            onnx_model, check = self.get_onnx_r2()
            assert check, "assert check failed"

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (
            self.conv_path / f"{self.model_name}-simplified.onnx"
        ).resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_blob(self):
        if self.f_simplified is None:
            self.export_onnx()

        if self.selected_release == R1_VERSION:
            output_list = [
                f"output{i+1}_yolo{self.version}" for i in range(self.num_branches)
            ]
        else:
            output_list = [
                f"output{i+1}_yolo{self.version}r2" for i in range(self.num_branches)
            ]
        output_list = ",".join(output_list)

        # export blob from generate onnx
        blob_path = blobconverter.from_onnx(
            model=str(self.f_simplified.resolve()),  # as_posix(),
            data_type="FP16",
            shaves=6,
            version="2021.4",
            use_cache=False,
            output_dir=self.conv_path.resolve(),
            optimizer_params=[
                "--scale=255",
                "--reverse_input_channel",
                f"--output={output_list}",
            ],
            download_ir=True,
        )

        self.f_blob = blob_path
        return blob_path

    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
