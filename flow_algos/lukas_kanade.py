import cv2
import time
import numpy as np
from src.of_estimator import OpticalFlowEstimator

class LukasKanade(OpticalFlowEstimator):

    def __init__(self, file, start_offset, roi, saved_bag=False, bottom_crop=0,
                 use_car_state=False):
        
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners = 1000,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )
        
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize = (15, 15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | 
                                          cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                              flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
        self.file = file
        self.start_offset = start_offset
        self.roi = roi
        self.saved_bag = saved_bag
        self.bottom_crop = bottom_crop
        self.use_car_state = use_car_state
        super().__init__(file, start_offset, roi, saved_bag=saved_bag,
                         bottom_crop=bottom_crop, use_car_state=use_car_state)

    def estimate_flow(self, image1, image2, prev_flow=None):
        start = time.time()
        p0 = self.extract_keypoints(image1)
        if prev_flow is None:
            prev_flow = np.zeros((image1.shape[0], image1.shape[1], 2)).astype(np.float32)
        p0_int = p0.astype(int)
        
        flow_vectors = prev_flow[p0_int[:, :, 1], p0_int[:, :, 0]]
        
        p1_est = p0 + flow_vectors
        p1_est = p1_est.astype(np.float32)
        
        p1, st, _ = cv2.calcOpticalFlowPyrLK(image1, image2,
                                               p0, p1_est, **self.lk_params)
        
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            good_old = np.round(good_old).astype(int)
            flow = np.zeros((image1.shape[0], image1.shape[1], 2))
            sparse_flow = good_new - good_old
            flow[good_old[:, 1], good_old[:, 0]] = sparse_flow
            mask = np.zeros_like(image1)
            mask[good_old[:, 1], good_old[:, 0]] = 1
            latency = time.time() - start
            return flow, mask, latency
        else:
            return None
    
    def extract_keypoints(self, image):
        return cv2.goodFeaturesToTrack(image, mask = None,
                                       **self.feature_params)