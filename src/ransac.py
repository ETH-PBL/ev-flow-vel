import cv2
import numpy as np
import src.motion_estimation as me
import utils.plotting as plotting


def create_vector_field(r, t, shape):
    h, w = shape
    x = np.arange(-w//2, w//2)
    y = np.arange(-h//2, h//2)
    xv, yv = np.meshgrid(x, y)
    p = np.stack((xv, yv), axis=-1)
    p_prime = np.dot(p, r.T) + t
    vec_field = p_prime - p

    return vec_field

def ransac(img, flow, mask, n_iters=16, inlier_thresh=0.1, plot_inliers=False,
           fraction=0.1):
    max_inliers = 0
    best_inliers = None
    
    for i in range(n_iters):
        valid_coords = np.argwhere(mask)
        rand_indices = np.random.choice(valid_coords.shape[0], 2, replace=False)
        rand_y, rand_x = valid_coords[rand_indices].T
        sample_coords = np.stack([rand_y, rand_x], axis=1)

        flow_1 = flow[rand_y[0], rand_x[0]]
        flow_2 = flow[rand_y[1], rand_x[1]]
        min_sample = np.stack([flow_1, flow_2], axis=0)
        t, r = me.rigid_body_motion_estimation(min_sample, mask=np.ones_like(flow[:, :, 0]),
                                               sparse_coords=sample_coords)
        
        vec_field = create_vector_field(r, t, flow.shape[:2])
        diff = np.full_like(flow, fill_value=np.iinfo(np.int32).max)
        diff[mask != 0] = flow[mask != 0] - vec_field[mask != 0]
        epe = np.sqrt(np.sum(diff ** 2, axis=2))

        inliers = epe < inlier_thresh
        n_inliers = np.sum(inliers)
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
        if plot_inliers:
            vis = plotting.plot_ransac_inliers(img.copy(), flow, vec_field, t,
                                               inliers, fraction=min(1, 4*fraction))
            cv2.imshow("RANSAC", vis)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    best_inlier_mask = np.zeros_like(flow)
    best_inlier_mask[best_inliers] = 1
    t_final, r_final = me.rigid_body_motion_estimation(flow, best_inlier_mask)
    return t_final, r_final