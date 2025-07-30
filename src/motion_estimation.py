import numpy as np

def transfer_velocity_to_new_frame(vel, omega, displacement):
    """
    Given the velocity at point A on a rigid body, the angular velocity of the
    rigid body and the displacement of point A to point B, compute the velocity
    at point B.
    :param vel: The velocity at point A [m/s]
    :param omega: The angular velocity of the rigid body [Hz]
    :param displacement: The displacement of point A to point B [m]
    :return: The velocity at point B [m/s]
    """

    vel_new = np.array([vel[0], vel[1], 0]) + np.cross(omega, displacement)
    return vel_new[:2]

def compute_velocity_in_m_per_s(vel_px, f=640, z=1.5, fps=100):
    """
    Compute the velocity in m/s from the pixel velocity. Because optical flow
    is in the opposite direction of the velocity, we need to multiply by -1.
    :param vel_px: The velocity in pixels [px/frame]
    :param f: The focal length of the camera. [px]
              e.g. focal length = 1280 /(2 * tan(90 * π / 360))
    :param z: The distance of the camera to the ground [m]
    :param fps: The frame rate with which the velocity was computed
    :return: The velocity [m/s]
    """

    vel = -vel_px * z / f * fps
    return vel

def compute_angular_velocity(r, fps=100):
    """
    Compute the angular velocity from the rotation matrix
    :param r: The rotation matrix
    :return: The angular velocity [rad/sec]
    """
    theta = np.arctan2(r[1, 0], r[0, 0])
    omega = np.array([0, 0, theta*fps])

    return omega

def create_vector_field(shape):
    """"
    Create a vactor field assuming planar motion for a random translation and
    rotation.
    :param shape: tuple of the shape of the image
    :return: flow: numpy array of shape (h, w, 2) containing the flow vectors
             grid: numpy array of shape (h, w, 2) containing the original pixel
                   coordinates
             new_grid: numpy array of shape (h, w, 2) containing the new pixel
                       coordinates after applying the transformation
    """
    h, w = shape
    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Create a flow field assuming planar motion
    grid = np.stack([x - w / 2, y - h / 2], axis=-1)
    # Create a random rotation matrix
    theta = 0.1
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
                                                      np.cos(theta)]])
    print(rot)
    # Create a random translation
    t = np.array([0, 0])
    # Apply the rotation and translation to the flow field
    new_grid = np.dot(grid, rot) + t
    flow = new_grid - grid

    return flow, grid, new_grid

def rigid_body_motion_estimation(flow: np.ndarray, mask=None, sparse_coords=None) -> tuple:
    """
    Estimate the translation and rotation of a rigid body given the flow.
    Implementation is based on https://igl.ethz.ch/projects/ARAP/svd_rot.pdf.

    :param flow: The flow field as a numpy array of shape (h, w, 2).
    :param mask: A binary mask of the same shape as flow, where 1 indicates
                 valid pixels and 0 indicates invalid pixels.
    :param sparse_coords: Optional, a numpy array of shape (n, 2) containing
                          sparse coordinates for the flow estimation.
    :return: translation: A numpy array of shape (2,) representing the
                          translation of the rigid body.
             rotation: A numpy array of shape (2, 2) representing the rotation
                       matrix of the rigid body.
    """

    
    if sparse_coords is None:
        h, w = flow.shape[:2]
        x, y = np.meshgrid(np.arange(flow.shape[1]), np.arange(flow.shape[0]))
        points_initial = np.stack([x - w / 2, y - h / 2], axis=-1)#.reshape(-1, 2)
        if mask is None:
            mask = np.ones_like(flow)

        points_final = points_initial + flow

        points_initial = points_initial[mask==1].reshape(-1, 2)
        points_final = points_final[mask==1].reshape(-1, 2)
    else:
        h, w = mask.shape
        points_initial = sparse_coords - np.array([w/2, h/2])
        points_final = points_initial + flow

    centroid_initial = np.mean(points_initial, axis=0)
    centroid_final = np.mean(points_final, axis=0)

    centered_initial = points_initial - centroid_initial
    centered_final = points_final - centroid_final

    H = np.dot(centered_initial.T, centered_final)
    U, _, Vt = np.linalg.svd(H)

    det_diag = np.diag([1, np.linalg.det(np.dot(Vt.T, U.T))])
    rot = np.linalg.multi_dot([Vt.T, det_diag, U.T])

    translation = centroid_final - np.dot(rot, centroid_initial)

    return translation, rot