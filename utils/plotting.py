import cv2
import numpy as np
import matplotlib.pyplot as plt


def velocity_graph(vel):
    """
    Plot the velocity of the camera
    :param vel: The estimated velocity
    :return: None
    """
    plt.plot(vel[:, 0], label="Longitudinal Velocity")
    plt.plot(vel[:, 1], label="Lateral Velocity")
    plt.xlabel("Frame")
    plt.ylabel("Velocity [m/s]")
    plt.title("Estimated Velocity")
    plt.legend()

def plot_ransac_inliers(vis, flow, vec_field, t, inliers, fraction=0.01):
    """
    Given the flow, the vector field and the inliers, plot the inliers green
    and the outliers red.
    :param vis: The image
    :param flow: The flow
    :param vec_field: The vector field
    :param t: The translation vector
    :param inliers: The inliers
    :param fraction: The fraction of flows to plot
    """
    h, w, _ = flow.shape

    step = int(1 / fraction)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if flow[y, x, 0] == 0 and flow[y, x, 1] == 0:
                continue
            else:
                vis = cv2.arrowedLine(vis, (x, y),
                        (int(x + flow[y, x, 0]), int(y + flow[y, x, 1])),
                        (255, 255, 0), 2)
            if inliers[y, x]:
                vis = cv2.arrowedLine(vis, (x, y),
                    (int(x + vec_field[y, x, 0]), int(y + vec_field[y, x, 1])),
                    (0, 255, 0), 2)
            else:
                vis = cv2.arrowedLine(vis, (x, y),
                    (int(x + vec_field[y, x, 0]), int(y + vec_field[y, x, 1])),
                    (0, 0, 255), 2)

    h, w, _ = vis.shape
    vis = cv2.arrowedLine(vis, (w//2, h//2),
                          (int(w//2 + t[0]), int(h//2 + t[1])),
                          (0, 255, 255), 2)
    add_legend(vis, ["Optical Flow", "RANSAC Inliers", "RANSAC Outliers",
                     "Translation of Camera Center"],
               [(255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)],
               location="top-left")
    return vis

def plot_of_and_estimate(image, of, r, t, fraction, arrow_scale):
    """
    Plot the optical flow as well as the estimated rigid-body motion.

    :param image: The image
    :param of: The optical flow
    :param r: The estimation rotation
    :param t: The estimated translation
    :param fraction: The fraction of vectors that shall be plotted
    :param arrow_scale: Rescaling of the arrows for better visualization
    :return: The image with the arrows drawn in it
    """
    vis = image.copy()
    vis = plot_of_arrows(vis, of, fraction=fraction, scaler=arrow_scale)
    vis = plot_velocities_from_r_and_t(r, t, vis, fraction=min(fraction, 0.01),
                                       scaler=arrow_scale)
    legend = ["Optical Flow", "Rigid-Body Motion", "Translation of Camera Center"]
    legend_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
    vis = add_legend(vis, legend, legend_colors)
    return vis


def plot_velocities(image, of_vel, gt_vel, cs_vel=None):
    """
    Plot a vector for each of the velocities in the center of the image.
    
    :param image: The image
    :param of_vel: The optical flow velocity
    :param gt_vel: The ground truth velocity
    :param cs_vel: The car state velocity
    :return: The image with the velocities
    """
    of_vel *= 10
    # of_vel = np.array([of_vel[0], -of_vel[1]])
    gt_vel *= 10
    if cs_vel is not None:
        cs_vel *= 10

    h, w, _ = image.shape
    vis = image.copy()
    # Draw the optical flow velocity
    vis = cv2.arrowedLine(vis, (w//2, h//2),
                          (int(w//2 + of_vel[0]), int(h//2 + of_vel[1])),
                          (0, 255, 0), 2)
    # Draw the ground truth velocity
    vis = cv2.arrowedLine(vis, (w//2, h//2),
                          (int(w//2 + gt_vel[0]), int(h//2 + gt_vel[1])),
                          (0, 0, 255), 2)
    # Draw the car state velocity
    if cs_vel is not None:
        vis = cv2.arrowedLine(vis, (w//2, h//2),
                              (int(w//2 + cs_vel[0]), int(h//2 + cs_vel[1])),
                              (255, 0, 0), 2)
    return vis


def plot_of_arrows(image, flow, mask=None, fraction=0.01, color=(0, 255, 0),
                   scaler=1, thic=2):
    """
    Plot the optical flow as arrows on the image
    :param image: The image
    :param flow: The optical flow
    :param mask: Optional plotting mask
    :param fraction: The fraction of flows to plot
    :param color: The color of the arrows
    :param scaler: The scaling factor ofthe of arrows
    :param thic: The thickness of the arrows
    :return: The image with the optical flow
    """
    color = (0, 255, 0)
    h, w, c = image.shape
    if c == 3:
        vis = image.copy()
    else:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    step = int(1 / fraction)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask is not None and not mask[y, x]:
                continue
            fx, fy = flow[y, x] * scaler
            
            if fx != 0 or fy != 0:
                vis = cv2.arrowedLine(vis, (x, y), (int(x + fx), int(y + fy)),
                                    color, thic)
            
    return vis

def plot_velocities_from_r_and_t(r, t, image, fraction=0.01, scaler=1):
    """
    Plot the optical flow as arrows on the image
    :param r: The rotation matrix
    :param t: The translation vector
    :param image: The image on which to plot the optical flow
    :param fraction: The fraction of pixels to plot
    :param scaler: The scaling factor for the arrows
    :return: The image with the optical flow
    """
    h, w, c = image.shape
    if c == 3:
        vis = image.copy()
    else:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    step = int(1 / fraction)
    for y in range(-h//2, h//2, step):
        for x in range(-w//2, w//2, step):
            p = np.array([x, y])
            p_prime = np.dot(r, p) + t * scaler

            p_prime += np.array([w//2, h//2])
            flow_start_x = x + w//2
            flow_start_y = y + h//2
            vis = cv2.arrowedLine(vis, (flow_start_x, flow_start_y),
                                  (int(p_prime[0]), int(p_prime[1])),
                                  (0, 0, 255), 2)
    # Plot the translation vector
    vis = cv2.arrowedLine(vis, (w//2, h//2),
                          (int(w//2 + t[0]*scaler), int(h//2 + t[1]*scaler)),
                          (255, 255, 0), 2)
    return vis


def compute_orientation(flow):
    """
    Compute the orientation of the optical flow
    :param flow: The optical flow
    :return: The orientation
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    return np.arctan2(v, u) * -1

def compute_magnitude(flow):
    """
    Compute the magnitude of the optical flow
    :param flow: The optical flow
    :return: The magnitude
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    return np.sqrt(u ** 2 + v ** 2)

def create_hist_of_magnitude(flow, bins=50):
    """
    Create a histogram of the magnitude of the optical flow
    :param flow: The optical flow
    :param bins: The number of bins
    :return: The histogram
    """
    magnitude = compute_magnitude(flow)
    return np.histogram(magnitude, bins=bins, range=(0, 100))

def create_hist_of_orientation(flow, bins=36):
    """
    Create a histogram of the orientation of the optical flow
    :param flow: The optical flow
    :param bins: The number of bins
    :return: The histogram
    """
    orientation = compute_orientation(flow)
    return np.histogram(orientation, bins=bins, range=(-np.pi, np.pi))

def plot_hist(hist):
    """
    Plot the histogram
    :param hist: The histogram
    :return: None
    """
    plt.bar(hist[1][:-1], hist[0], width=2 * np.pi / len(hist[0]))
    plt.show()

def plot_orientation_hist(flow):
    """
    Plot the orientation histogram with a polar plot
    """
    orientation = compute_orientation(flow).flatten()
    magnitude = compute_magnitude(flow).flatten()

    rbins = np.linspace(0, 2, 2)
    abins = np.linspace(-np.pi, np.pi, 30)

    hist, _, _ = np.histogram2d(orientation, magnitude, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r")
    fig.colorbar(pc)
    ax.grid(True)
    plt.show()

def draw_text(img,
              text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)):
    """
    Draw text on an image with a background rectangle.
    :param img: The image on which to draw the text
    :param text: The text to draw
    :param font: The font to use for the text
    :param pos: The position of the text
    :param font_scale: The scale of the font
    :param font_thickness: The thickness of the font
    :param text_color: The color of the text
    :param text_color_bg: The color of the background rectangle
    :return: The image with the text drawn on it
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h + 10), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale,
                text_color, font_thickness)
    return img

def add_legend(img, items, colors, location="bottom-left",
               font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1, font_thickness=2):
    """
    Add a legend to the image with the specified items and colors.
    :param img: The image to which the legend will be added
    :param items: A list of items to display in the legend
    :param colors: A list of colors corresponding to the items
    :param location: The location of the legend on the image
    :param font: The font to use for the legend text
    :param font_scale: The scale of the font
    :param font_thickness: The thickness of the font
    :return: The image with the legend added
    """
    h, w, _ = img.shape
    if location == "top-left":
        x, y = 10, 10
    elif location == "top-right":
        x, y = w - 100, 10
    elif location == "bottom-left":
        x, y = 10, h - 100
    elif location == "bottom-right":
        x, y = w - 350, h - 100
    else:
        raise ValueError("Invalid location")

    for i, item in enumerate(items):
        draw_text(img, item, pos=(x, y + i * 30), font=font,
                  font_scale=font_scale, font_thickness=font_thickness,
                  text_color=colors[i])

    return img