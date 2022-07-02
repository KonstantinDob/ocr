import os
import cv2

from ocr.inference import InferenceOCR


def main():
    """Before run inference set parameters in the inference.yaml."""
    inference = InferenceOCR()

    image_names = os.listdir('data/inference_images')
    image_paths = [os.path.join('data/inference_images', image_name) for
                   image_name in image_names]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        output = inference.predict(image=image)

        cv2.imshow('0', output)
        cv2.waitKey()


if __name__ == "__main__":
    main()
