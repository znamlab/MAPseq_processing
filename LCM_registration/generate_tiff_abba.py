import iss_preprocess as iss
import cv2
from iss_preprocess.io.save import save_ome_tiff_pyramid
import matplotlib.pyplot as plt
from flexiznam.config import PARAMETERS
import numpy as np
from pathlib import Path
from iss_preprocess.image import correction
import yaml

processed = Path(PARAMETERS["data_root"]["processed"])
data_path = 'turnerb_MAPseq/LCM/sections/FIAA32.6a/ara_registration'

data_path = processed / data_path

assert data_path.is_dir()
metadata_file = data_path / "metadata.yml"
assert metadata_file.exists()

with open(metadata_file, 'r') as fhandle:
    metadata = yaml.safe_load(fhandle)

abba_folder = data_path / "abba"
abba_folder.mkdir(exist_ok=True)
for image_file in (data_path / "raw_sections").glob('*.jpg'):
    image = cv2.imread(str(image_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.bitwise_not(image)
    target = abba_folder / f"{image_file.stem}.ome.tif"
    save_ome_tiff_pyramid(
        target,
        image,
        pixel_size=metadata['pixel_size'],
        subresolutions=3,
        dtype="uint16",
        verbose=True,
        save_thumbnail=False
    )