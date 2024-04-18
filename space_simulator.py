import logging
import os
import re
from datetime import datetime
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

G = 6.6743 * 10 ** (-11)  # Gravitational constant, units: m^3 kg^-1 s^-2 / N m^2 kg^-2
SPACE_SIZE = 1000

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def r(p: np.ndarray, q: np.ndarray) -> float:
    """
    Parameters
    ----------
    p : Location of point p
    q : Location of point q
    Returns
    -------
    float
        Euclidean distance between the points
    """
    d = np.sqrt(np.dot(p - q, (p - q).T))
    return d if d > 50 else 50  # to avoid division by very small numbers


def ru(p, q):
    """
    Parameters
    ----------
    p : Location of point p
    q : Location of point q
    Returns
    -------
    ndarray of floats
        unit vector pointing from p to q
    """
    x = q[0] - p[0]
    y = q[1] - p[1]

    d = r(p, q)

    return np.array([x / d, y / d])


class ObjectSimulator:

    def __init__(
        self,
        mp: float,
        mq: float,
        up: np.ndarray,
        uq: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        frames_dir: str,
    ) -> None:
        """
        Parameters
        ----------
        mp : float
            Mass of point p
        mq : float
            Mass of point q
        up : ndarray of floats
            Initial velocity of point p
        uq : ndarray of floats
            Initial velocity of point q
        p : ndarray of floats
            Initial location of point p
        q : ndarray of floats
            Initial location of point q
        frames_dir : str
            Directory to save the frames

        """
        self.mp = mp
        self.mq = mq
        self.up = up
        self.uq = uq
        self.p = p
        self.q = q
        self.space = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        self.frames_dir = frames_dir
        os.makedirs(self.frames_dir, exist_ok=True)

        logger.info("ObjectSimulator initialized with parameters...")
        logger.info(f"mp: {self.mp:,} kg, mq: {self.mq:,} kg")
        logger.info(f"up: {self.up}, uq: {self.uq}")
        logger.info(f"p: {self.p}, q: {self.q}")
        logger.info(f"frames_dir: {self.frames_dir}")

    def mass_circle(self, x: int, y: int, circle: str) -> np.ndarray:
        """
        Parameters
        ----------
        x : int
            x-coordinate of the point
        y : int
            y-coordinate of the point
        Returns
        -------
        np.ndarray
            Boolean array of the points within the circle

        """
        return (
            (
                ((x - self.p[0]) ** 2) + ((y - self.p[1]) ** 2)
                <= ((self.mp / 10**14) * 100)
            )
            if circle == "p"
            else (
                (x - self.q[0]) ** 2 + (y - self.q[1]) ** 2
                <= ((self.mq / 10**14) * 100)
            )
        )

    def get_new_space_state(self) -> np.ndarray:
        """
        Get the new state of the space by adding the mass circles of the points
        """
        new_space = (
            self.space
            + np.fromfunction(
                lambda x, y: self.mass_circle(x, y, "p"), (SPACE_SIZE, SPACE_SIZE)
            )
            + np.fromfunction(
                lambda x, y: self.mass_circle(x, y, "q"), (SPACE_SIZE, SPACE_SIZE)
            )
        )
        return new_space

    def update_velocity(self, del_t: int) -> None:
        """
        Update the velocity of the points by calculating the acceleration using Newton's law of gravitation
        """
        ap = G * self.mq * ru(self.p, self.q) / (r(self.p, self.q) ** 2)
        aq = G * self.mp * ru(self.q, self.p) / (r(self.p, self.q) ** 2)

        self.up += ap * del_t
        self.uq += aq * del_t

    def update_position(self, del_t) -> None:
        """
        Update the position of the points by adding the velocity
        Parameters
        ----------
        del_t : int
            Time step
        """
        self.p += self.up * del_t
        self.q += self.uq * del_t

        # Periodic boundary
        self.p %= SPACE_SIZE
        self.q %= SPACE_SIZE

    def plot_trace(self):
        """
        Plot the trace of the points
        """
        t = 2  # size of the trace points
        px, py = self.p.astype(np.uint16)
        qx, qy = self.q.astype(np.uint16)

        self.space[px - t : px + t, py - t : py + t] = 1
        self.space[qx - t : qx + t, qy - t : qy + t] = 1

    def simulate(self, del_t: int, n_iter: int) -> List[str]:
        """
        Simulate the movement of the points
        Parameters
        ----------
        del_t : int
            Time step
        n_iter : int
            Number of iterations
        Returns
        -------
        List[str]
            List of paths of the saved frames
        """
        saved_frames_paths = []
        for i in tqdm(range(n_iter), desc="Simulating..."):
            self.update_velocity(del_t)
            self.update_position(del_t)
            self.plot_trace()
            space_state = self.get_new_space_state()

            fig = plt.figure(figsize=(20, 15))

            plt.imshow(space_state, cmap="gray")
            plt.xticks([])
            plt.yticks([])
            savepath = self.frames_dir + "/img.{0:05d}.png".format(i)
            plt.savefig(savepath)
            plt.close(fig)
            saved_frames_paths.append(savepath)

        logger.info("Simulation completed")
        logger.info(f"Position of p: {self.p}, Position of q: {self.q}")
        logger.info(f"Velocity of p: {self.up}, Velocity of q: {self.uq}")

        return saved_frames_paths

    def __call__(self, del_t: int, n_iter: int) -> List[str]:
        return self.simulate(del_t, n_iter)


def write_frames_to_video(frames_dir: str, vid_dir: str) -> None:
    """
    Write the frames to a video
    Parameters
    ----------
    frames_dir : str
        Directory containing the frames
    vid_dir : str
        Path to save the video
    """
    list_files = os.listdir(frames_dir)
    list_files = sorted([f for f in list_files if re.match(r"img\.\d{5}\.png$", f)])
    imshape = cv2.imread(os.path.join(frames_dir, list_files[0])).shape
    imshape = (imshape[1], imshape[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    result = cv2.VideoWriter(vid_dir, fourcc, 20, imshape)
    logger.info(f"Writing frames to video: {vid_dir}")
    for file in tqdm(list_files, desc="Saving ..."):
        image = cv2.imread(os.path.join(frames_dir, file))
        result.write(image)
    result.release()

    logger.info(f"\nSimulation saved as: {vid_dir}")


if __name__ == "__main__":

    mp = 4 * pow(10, 14)  # kg
    mq = 2 * pow(10, 14)  # kg

    up = np.array([2.0, -3.0])
    uq = np.array([-3, -2.5])

    p = np.array([400.0, 400.0])
    q = np.array([900.0, 200.0])

    del_t = 1
    n_iter = 1000

    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    frames_dir = "frames_dir_space_sim" + now
    logger.info(f"Creating frames directory: {frames_dir}")

    vid_dir = f"space_sim_t{del_t * n_iter}_{now}.mp4"

    obj_sim = ObjectSimulator(
        mp=mp, mq=mq, up=up, uq=uq, p=p, q=q, frames_dir=frames_dir
    )
    sim_frames = obj_sim(del_t, n_iter)
    write_frames_to_video(frames_dir, vid_dir)
