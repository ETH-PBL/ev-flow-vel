import cv2
import numpy as np
from src.of_estimator import OpticalFlowEstimator
import time

class Farneback(OpticalFlowEstimator):

    def __init__(self, file, start_offset, roi, saved_bag=False, bottom_crop=0,
                 use_car_state=False, pyr_scale=0.5, levels=3, winsize=120,
                 iterations=5, poly_n=5, poly_sigma=1.2, flags=1):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.file = file
        self.start_offset = start_offset
        self.roi = roi
        self.saved_bag = saved_bag
        self.bottom_crop = bottom_crop
        self.use_car_state = use_car_state
        super().__init__(file, start_offset, roi, saved_bag=saved_bag,
                         bottom_crop=bottom_crop, use_car_state=use_car_state)

    def estimate_flow(self, image1, image2, input_flow=None):
        mask = np.ones_like(image1)
        if input_flow is None:
            input_flow = np.zeros((image1.shape[0],
                                   image1.shape[1], 2)).astype(np.float32)
        
        start = time.time()
        flow_with_initial_guess = cv2.calcOpticalFlowFarneback(image1, image2,
                                                input_flow,
                                                0.5, 3, 45, 3, 7, 1.5,
                                                cv2.OPTFLOW_USE_INITIAL_FLOW)
        latency = time.time() - start
        return flow_with_initial_guess, mask, latency
