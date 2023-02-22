# coding=utf-8
import json
from pathlib import Path
from zipfile import ZIP_LZMA, ZipFile

import blobconverter
import onnx
import onnxsim
import torch

import docker

# client = docker.from_env()
# image = client.images.pull("openvino/ubuntu20_dev", tag="2022.1.0")
# docker_output = client.containers.run(
#     image=image.tags[0],
#     command="echo 'MYRIAD_ENABLE_MX_BOOT ON' | tee /tmp/myriad.conf "
#     + "&& /opt/intel/openvino/tools/compile_tool/compile_tool -m {} -ip {} "
#     + "-VPU_NUMBER_OF_SHAVES {} -VPU_NUMBER_OF_CMX_SLICES {} -d MYRIAD -c /tmp/myriad.conf",
#     remove=True,
#     volumes=[
#         ":/io",
#     ],
#     working_dir="/io",
# )
ov_dict = {
    "2021.2": "2021.2",
    "2021.3": "2021.3.2",
    "2021.4": "2021.4.3",
    "2022.1": "2022.1.0",
}


class Exporter:
    def __init__(
        self,
        conv_path,
        weights_filename,
        imgsz=640,
        version="v5",
        shaves=6,
        openvino_version="2202.1",
        data_type="U8",
        convert="network",
        **kwargs,
    ):

        # set up variables
        self.conv_path = conv_path
        self.weights_path = self.conv_path / weights_filename
        self.imgsz = imgsz
        self.model_name = weights_filename.split(".")[0]  # "result"
        self.version = version
        self.shaves = shaves
        self.openvino_version = openvino_version
        self.data_type = data_type

        # set up file paths
        self.f_onnx = None
        self.f_simplified = None
        # self.f_bin = None
        # self.f_xml = None
        # self.f_mapping = None
        self.f_blob = None
        self.f_json = None
        self.f_zip = None

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
            output_names=["output"],
            dynamic_axes=None,
        )

        # check if the architecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # simplify the model
        return onnxsim.simplify(model_onnx)

    def export_blob(self):

        output_list = [
            f"output{i + 1}_yolo{self.version}" for i in range(self.num_branches)
        ]
        output_list = ",".join(output_list)

        if self.f_simplified is None:
            self.export_onnx()
        # export blob from generate onnx
        blob_path = blobconverter.from_onnx(
            model=str(self.f_simplified.resolve()),  # as_posix(),
            data_type="FP16",
            shaves=self.shaves,
            version=self.openvino_version,
            use_cache=False,
            output_dir=self.conv_path.resolve(),
            optimizer_params=[
                "--scale=255",
                "--reverse_input_channel",
                f"--output={output_list}",
                "--use_new_frontend" if self.openvino_version >= "2022.1" else "",
            ],
            compile_params=[f"-ip {self.data_type}"],
            download_ir=True,
        )

        self.f_blob = blob_path
        return blob_path

    def write_json(self, anchors, masks, nc=None, names=None):
        # set parameters
        f = open((Path(__file__).parent / "json" / "yolo.json").resolve())
        content = json.load(f)

        content["model"]["xml"] = f"{self.model_name}.xml"
        content["model"]["bin"] = f"{self.model_name}.bin"
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        if nc:
            content["nn_config"]["NN_specific_metadata"]["classes"] = nc
        else:
            content["nn_config"]["NN_specific_metadata"]["classes"] = self.model.nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        if names:
            # use COCO labels if 80 classes, else use a placeholder
            content["mappings"]["labels"] = (
                content["mappings"]["labels"] if nc == 80 else names
            )
        else:
            content["mappings"]["labels"] = (
                self.model.names
                if isinstance(self.model.names, list)
                else list(self.model.names.values())
            )
        content["version"] = 1

        # save json
        f_json = (self.conv_path / f"{self.model_name}.json").resolve()
        with open(f_json, "w") as outfile:
            json.dump(content, outfile, ensure_ascii=False, indent=4)

        self.f_json = f_json

        return self.f_json

    def make_zip(
        self,
    ):
        # create a ZIP folder
        if self.f_simplified is None:
            self.export_onnx()

        if self.f_blob is None:
            self.export_blob()

        if self.f_json is None:
            self.export_json()

        # f_zip = f"{DIR_TMP}/{self.model_name}.zip"
        # f_zip = (self.conv_path / f"{self.model_name}.zip").resolve()
        f_zip = self.f_blob

        with ZipFile(f_zip, "a", ZIP_LZMA) as zip_obj:
            zip_obj.extract(
                self.f_simplified.with_suffix(".bin").name,
                self.conv_path.resolve(),
            )
            zip_obj.extract(
                self.f_simplified.with_suffix(".xml").name,
                self.conv_path.resolve(),
            )
            zip_obj.extract(
                self.f_simplified.with_suffix(".blob").name,
                self.conv_path.resolve(),
            )
            # zip_obj.write(self.f_simplified, self.f_simplified.name)
            # zip_obj.write(self.f_json, self.f_json.name)

        self.f_zip = f_zip
        return f_zip
