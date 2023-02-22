# coding=utf-8
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

parents = Path(__file__).parent


def export_file(args):
    if "x" in args.imgsz:
        imgsz = args.imgsz.split("x")
        input_shape = [int(imgsz[0]), int(imgsz[1])]
    else:
        input_shape = int(args.imgsz)
    args.imgsz = input_shape
    file: Path = args.file
    filename = file.name

    conv_path = (parents / "export").resolve().absolute() / file.stem
    conv_path.mkdir(exist_ok=True, parents=True)

    # shutil.copy(file, conv_path / filename)
    # (conv_path / filename).link_to(file)
    if (conv_path / filename).exists():
        (conv_path / filename).unlink()
    file.link_to(conv_path / filename)

    if args.version == "v5":
        from yolo.export_yolov5 import YoloV5Exporter as Exporter
    elif args.version == "v5lite":
        from yolo.export_yolov5lite import YoloV5LiteExporter as Exporter

        args.version = "v5"
    elif args.version == "v6":
        from yolo.export_yolov6 import YoloV6Exporter as Exporter
    elif args.version == "v7":
        from yolo.export_yolov7 import YoloV7Exporter as Exporter
    elif args.version == "v8":
        from yolo.export_yolov8 import YoloV8Exporter as Exporter
    else:
        raise ValueError(f"Yolo version {args.version} is not supported.")

    exporter = Exporter(conv_path, filename, **vars(args))

    exporter.make_zip()


if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
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
            type=Path,
            required=True,
            help="Weights of a pre-trained model (.pt file), size needs to be smaller than 100Mb.",
        )
        parser.add_argument(
            "-v",
            "--version",
            type=str,
            help="Yolo Version",
            choices=["v5", "v5lite", "v6", "v7", "v8"],
        )

        parser.add_argument(
            "-sh",
            "--shaves",
            type=int,
            default=6,
            help="Specifies number of SHAVE cores that converted model will use",
            choices=range(1, 16),
        )

        parser.add_argument(
            "-ov",
            "--openvino_version",
            type=str,
            help="OpenVINO version to use for conversion",
            choices=["2021.2", "2021.3", "2021.4", "2022.1"],
            default="2021.4",
        )

        parser.add_argument(
            "-dt",
            "--data_type",
            type=str,
            help="Specifies precision for all input layers of the network",
            choices=["U8", "FP16"],
            default="U8",
        )

        export_file(parser.parse_args())
