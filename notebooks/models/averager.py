import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from models.utils import make_template_mass


class Averager(object):

    def __init__(self):

        self.template = None
        self.pupil_location = None


    def train(self, train_patches, n_order=1):
        """
        Inputs:
        -------
        train_patches (array):
            Array of shape (num_patch, patch_size) with the training semples (i.e. positive patches)
        n_order (int):
            The order to perform smoothing (c.f. preprocessing.m_function)
        """

        m = np.mean(train_patches, axis=0)

        self.template = (m - np.mean(m)) / np.std(m)

        r = (self.template.shape[0] - 1) / 2

        temp = make_template_mass(r=r, n_order=n_order)

        self.template *= temp


    def predict_im(self, image, ax=None):
        """
        Inputs:
        -------
        image (array):
            Array in gray tone (2D)
        ax (plt.axis):
            Axis on which disply the result

        Outputs:
        --------
        """

        conv = correlate2d(image, self.template, mode='same')

        (y, x) = np.where(conv==np.amax(conv))

        self.pupil_location = (y, x)

        if ax is not None:

            ax.imshow(image, cmap='gray')
            ax.scatter(x, y, c='r')

        return conv, (y, x)