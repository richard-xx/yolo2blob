# coding=utf-8
import sys

import onnxsim

from yolo.detect_head import DetectV5

sys.path.append("./yolo/yolov5lite")

import onnx
import torch
import torch.nn as nn
from yolo.yolov5lite.models.common import Conv
from yolo.yolov5lite.models.experimental import attempt_load
from yolo.yolov5lite.models.yolo import Detect
from yolo.yolov5lite.utils.activations import Hardswish, SiLU

from yolo.exporter import Exporter

DIR_TMP = "./tmp"


class YoloV5LiteExporter(Exporter):
    def __init__(self, conv_path, weights_filename, **kwargs):
        super().__init__(conv_path, weights_filename, **kwargs)
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(
            self.weights_path.resolve(), map_location=torch.device("cpu")
        )  # load FP32 model

        # check num classes and labels
        assert model.nc == len(
            model.names
        ), f"Model class count {model.nc} != len(names) {len(model.names)}"

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

        inplace = True

        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # if isinstance(model.model[-1], Detect):
        model.model[-1] = DetectV5(model.model[-1])

        self.model = model

        m = model.model[-1]
        self.num_branches = len(m.anchor_grid)

    def get_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}-origin.onnx").resolve()
        im = torch.zeros(
            1, 3, *self.imgsz[::-1]
        )  # .to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=17,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=[
                f"output{i+1}_yolo{self.version}" for i in range(self.num_branches)
            ],
            dynamic_axes=None,
        )

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel
        return onnxsim.simplify(model_onnx)

    def export_onnx(self):
        onnx_model, check = self.get_onnx()
        assert check, "assert check failed"

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_json(self):
        # generate anchors and sides
        anchors, sides = [], []
        m = (
            self.model.module.model[-1]
            if hasattr(self.model, "module")
            else self.model.model[-1]
        )
        for i in range(self.num_branches):
            sides.append(int(self.imgsz[0] // m.stride[i]))
            for j in range(m.anchor_grid[i].size()[1]):
                anchors.extend(m.anchor_grid[i][0, j, 0, 0].numpy())
        anchors = [float(x) for x in anchors]
        # sides.sort()

        # generate masks
        masks = dict()
        # for i, num in enumerate(sides[::-1]):
        for i, num in enumerate(sides):
            masks[f"side{num}"] = list(range(i * 3, i * 3 + 3))

        return self.write_json(anchors, masks)
