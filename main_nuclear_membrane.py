import numpy as np
import matplotlib 
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
import time
import yaml
from tqdm import tqdm
from utils import *

if __name__ == "__main__":
    start_time = time.time()
    config = Config('config_membrane.yaml')
    AngleStepSize = np.float64((config.end_angle - config.start_angle) / config.nr_of_angle)
    membraneCoordinatesX = np.zeros((config.nr_of_angle, config.NrofFrames), dtype=np.float64)
    membraneCoordinatesY = np.zeros((config.nr_of_angle, config.NrofFrames), dtype=np.float64)
    filtered_imgs = []

    for i in tqdm(range(config.NrofFrames)):
        image_processor = ImageProcessing(config, image_index=i)
        filtered_img = image_processor.filtered_image
        filtered_imgs.append(filtered_img)  # Collect filtered images
        for j in range(config.nr_of_angle):
            contour_tracker = ContourTracker_Membrane(filtered_img, config.center, AngleStepSize)
            membraneCoordinatesX[j, i], membraneCoordinatesY[j, i] = contour_tracker.search_coordinates(j)

    # Convert filtered_imgs list to numpy array
    filtered_imgs = np.array(filtered_imgs)
    post_processing = PostProcessing(filtered_imgs, membraneCoordinatesX, membraneCoordinatesY, config.save_path, config.file_name)
    membraneCoordinatesX = post_processing.fill_nan_interpolation(membraneCoordinatesX)
    membraneCoordinatesY = post_processing.fill_nan_interpolation(membraneCoordinatesY)
    post_processing.save_coordinates()
    print(f"---process_time {time.time() - start_time} seconds ---")
    post_processing.create_animation()    