# coding=utf-8
import argparse
from pathlib import Path
import shutil


def export_file(imgsz, file, version):

    if "x" in imgsz:
        imgsz = imgsz.split("x")
        input_shape = [int(imgsz[0]), int(imgsz[1])]
    else:
        input_shape = int(imgsz)
    file = Path(file)
    filename = file.name

    conv_path = Path("export") / file.stem
    conv_path.mkdir(exist_ok=True, parents=True)

    shutil.copy(file, conv_path / filename)

    if version == "v5":
        from yolo.export_yolov5 import YoloV5Exporter

        exporter = YoloV5Exporter(conv_path, filename, input_shape, version)
    elif version == "v6":
        from yolo.export_yolov6 import YoloV6Exporter

        exporter = YoloV6Exporter(conv_path, filename, input_shape, version)
    elif version == "v7":
        from yolo.export_yolov7 import YoloV7Exporter

        exporter = YoloV7Exporter(conv_path, filename, input_shape, version)
    else:
        raise ValueError(f"Yolo version {version} is not supported.")

    exporter.export_onnx()
    exporter.export_json()
    exporter.export_blob()
    exporter.make_zip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-imgsz",
        "--imgsz",
        type=str,
        default="640",
        help="Integer for square shape, or width and height separated by space. Must be divisible by 32.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Weights of a pre-trained model (.pt file), size needs to be smaller than 100Mb.",
    )
    parser.add_argument(
        "-v", "--version", type=str, help="Yolo Version", choices=["v5", "v6", "v7"]
    )

    args = parser.parse_args()

    export_file(imgsz=args.imgsz, file=args.file, version=args.version)
