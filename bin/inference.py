import os
import cv2
import argparse

from ocr.inference import InferenceOCR


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', type=str, default='data/inference_images'
    )

    args = parser.parse_args()
    return args


def main():
    """Before run inference set parameters in the configs."""
    args = parse_arguments()

    inference = InferenceOCR()

    image_names = os.listdir(args.datapath)
    image_paths = [os.path.join(args.datapath, image_name) for
                   image_name in image_names]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        output = inference.predict(image=image)

        cv2.imshow('0', output)
        cv2.waitKey()


if __name__ == "__main__":
    main()
