"""Script for cropping celebA adopted from: https://github.com/ningyu1991/GANFingerprints/"""
import argparse
import os

from PIL import Image
import numpy as np

from concurrent.futures import ProcessPoolExecutor

def crop_image(stupid):
    i, directory, file_path, output = stupid
    if file_path.endswith("png") or file_path.endswith("jpeg") or file_path.endswith("jpg"):
        image = np.asarray(Image.open(f"{directory}/{file_path}"))

        if image.shape[0] != 128 or image.shape[1] != 128:
            x,y,_= image.shape
            image = np.copy(image)
            x_upper = min(121+64, x)
            y_upper = min(89+64, y)
            image = image[x_upper-128:x_upper, y_upper-128:y_upper]
            image = np.clip(image, 0, 255.).astype(np.uint8)

        if not (image.shape[0] == 128 and image.shape[1] == 128):
            print("Aborting")
            return i

        Image.fromarray(image).save(f"{output}/cropped_{file_path}")
        return i

def resize_image(stupid):
    i, directory, file_path, output = stupid
    if file_path.endswith("png") or file_path.endswith("jpeg") or file_path.endswith("jpg"):
        image = Image.open(f"{directory}/{file_path}")

        # Resize the image to 128x128
        resized_image = image.resize((128, 128))

        # Save the resized image
        resized_image.save(f"{output}/resized_{file_path}")

        return i

def main(args):
    os.makedirs(args.OUTPUT, exist_ok=True)
    paths = os.listdir(args.DIRECTORY)
    packed = map(lambda x: (x[0], args.DIRECTORY, x[1], args.OUTPUT) , enumerate(paths))

    with ProcessPoolExecutor() as pool:
        jobs = pool.map(resize_image, packed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--DIRECTORY", help="Source directory.", type=str, default=r'C:\Users\dmpoo\OneDrive\Desktop\Yamini\realvsfake_merged\fake')
    parser.add_argument("--OUTPUT", help="Output directory.", type=str, default=r'C:\Users\dmpoo\OneDrive\Desktop\Yamini\realvsfake_merged\fake_cropped')
    parser.add_argument("--SIZE", help="Amount of data to convert.", type=int, default=1000)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

