import cv2
import argparse
import numpy as np

import utils.plotting as plotting
import src.motion_estimation as me
import matplotlib.pyplot as plt

from flow_algos.farneback import Farneback
from flow_algos.lukas_kanade import LukasKanade
from src.ransac import ransac

if __name__=="__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--flow-algo", type=str,
                           default="farneback",)
    argparser.add_argument("--video-path", type=str,
                           default="data/event_faster.avi")
    argparser.add_argument("--start-offset", type=float, default=0.0)
    argparser.add_argument("--save-gif", type=str, default=None)
    argparser.add_argument("--use-ransac", type=bool, default=False)
    argparser.add_argument("--camera-height", type=float, default=0.25)
    argparser.add_argument("--camera-focal-length", type=float, default=1110)
    argparser.add_argument("--roi", type=int, nargs=4, default=None)
    argparser.add_argument("--delta-t", type=float, default=0.005)
    argparser.add_argument("--end-frame", type=int, default=20)
    
    
    args = argparser.parse_args()

    video_path = args.video_path
    if args.flow_algo == "farneback":
        flow_algo = Farneback(video_path, args.start_offset, args.roi)
        plot_fraction = 0.01
    elif args.flow_algo == "lukas-kanade":
        flow_algo = LukasKanade(video_path, args.start_offset, args.roi)
        plot_fraction = 0.5
    else:
        raise ValueError(f"Unknown flow algorithm: {args.flow_algo}.\
                           Choose between 'farneback' and 'lukas-kanade'")

    velocities = []
    frame_num = 0
    for (flow, mask, img, other_topics) in flow_algo.flow_generator():
        print(f"Frame: {frame_num}", end='\r')
        if args.use_ransac:
            t, r = ransac(img, flow, mask, plot_inliers=False,
                          fraction=plot_fraction)
        else:
            t, r = me.rigid_body_motion_estimation(flow, mask)
        
        t_orientation = np.arctan2(t[1], t[0])
        img_with_flow = plotting.plot_of_and_estimate(img, flow, r, t,
                                                      fraction=plot_fraction,
                                                      arrow_scale=3)

        if other_topics is not None or args.delta_t is not None:
            if args.delta_t is not None:
                img_delta_t = args.delta_t
            else:
                img_delta_t = other_topics["img_delta_t"]
            v_camera = me.compute_velocity_in_m_per_s(t,
                                                      f=args.camera_focal_length,
                                                      z=args.camera_height,
                                                      fps=1/img_delta_t)
            velocities.append(v_camera)

        cv2.imshow("Optical flow", img_with_flow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if args.end_frame != None:
            if frame_num > args.end_frame:
                break
        frame_num += 1
        
        if args.save_gif is not None:
            img_with_flow = cv2.cvtColor(img_with_flow, cv2.COLOR_BGR2RGB)
    velocities = -np.array(velocities)
    plotting.velocity_graph(velocities)
    plt.show()
