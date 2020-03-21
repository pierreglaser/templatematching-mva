import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class Averager(object):

    def __init__(self):

        self.template = None
        self.pupil_location = None


    def train(self, train_patches):
        """
        Inputs:
        -------
        train_patches (array): Array of shape (num_patch, patch_size) with the training semples (i.e. positive patches)
        """

        m = np.mean(train_patches, axis=0)

        self.template = (m - np.mean(m)) / np.std(m)


    def predict_im(self, image, ax=None):
        """
        Inputs:
        -------
        image (array): Array in gray tone (2D)
        ax (plt.axis): ax on whoch disply the result

        Outputs:
        --------
        """

        image = (image - np.mean(image)) / np.std(image)
        conv = convolve2d(image, self.template, mode='same')

        (y, x) = np.where(conv==np.amax(conv))

        self.pupil_location = (y, x)

        if ax is not None:

            ax.imshow(image, cmap='gray')
            ax.scatter(x, y, c='r')

        return conv, (y, x)