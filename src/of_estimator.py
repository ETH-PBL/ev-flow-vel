import cv2
import os

import numpy as np

from abc import ABC, abstractmethod


class OpticalFlowEstimator(ABC):
    def __init__(self, file, start_offset, roi, bottom_crop=None,
                 saved_bag=False, use_car_state=False):
        self.file = file
        print(self.file)
        self.flow_generator = self.check_file_extension()
        self.start_offset = start_offset
        if roi is None:
            self.crop = False
        else:
            self.crop = True
            self.roi_top_left = (roi[0], roi[1])
            self.roi_bottom_right = (roi[2], roi[3])

        self.bottom_crop = bottom_crop
        self.saved_bag = saved_bag
        self.use_car_state = use_car_state

    @abstractmethod
    def estimate_flow(self, image1, image2):
        pass

    def check_file_extension(self):
        if self.file.endswith('.mp4') or self.file.endswith('.avi'):
            return self.video_flow_generator
        elif self.saved_bag:
            return self.img_files_with_gt_generator
        elif self.file.endswith('.bag'):
            return self.rosbag_flow_generator
        elif os.path.isdir(self.file):
            return self.from_files_flow_generator
        else:
            raise ValueError(f"Unknown file extension: \
                             {self.file.split('.')[-1]}")
    
    def get_topic_files(self, topic):
        dir = self.file + topic
        files = os.listdir(dir)
        files.sort()
        files = [dir + file for file in files]
        return files

    def video_flow_generator(self):
        cap = cv2.VideoCapture(self.file)
        ret, prev_frame = cap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        if self.crop:
            prev_frame = prev_frame[self.roi_top_left[1]:self.roi_bottom_right[1],
                                    self.roi_top_left[0]:self.roi_bottom_right[0]]

        while ret:
            ret, curr_frame_c = cap.read()
            prev_flow = None
            if ret:
                if self.crop:
                    curr_frame_c = curr_frame_c[self.roi_top_left[1]:self.roi_bottom_right[1],
                                            self.roi_top_left[0]:self.roi_bottom_right[0]]
                curr_frame = cv2.cvtColor(curr_frame_c, cv2.COLOR_BGR2GRAY)
                flow, mask, latency = self.estimate_flow(prev_frame, curr_frame,
                                                prev_flow)
                other_topics = {"latency": latency}
                yield flow, mask, curr_frame_c, other_topics
                prev_frame = curr_frame
                prev_flow = flow
        cap.release()

            
