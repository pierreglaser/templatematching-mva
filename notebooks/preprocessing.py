import numpy as np
from scipy.signal import convolve2d


def m_function(x, y, r, n_order=1):
    """
    The windowing function a defined in the article

    Inputs:
    -------
    x (int): x-coordinate with (0, 0) in center of image
    y (int): y-coordinate with (0, 0) in center of image
    r (float): the disk radius
    n_order (int): the order of the sum (c.f. paper)
    """

    res = 0

    norm_sq = x * x + y * y
    s = 2 * r * r / (1 + 2 * n_order)

    for i in range(n_order):

        res += np.exp(- norm_sq / s) * (norm_sq / s) ** i / np.math.factorial(i)


    return res


def mask_img(image, c=(0, 0), r=10):
    """
    Create disk-like mask for selecytong region of interest

    Inputs:
    -------
    image (array): The input image
    c (tuple): The center
    r (int): the radius of the disk 
    """
    (dim_y, dim_x) = image.shape
    X = np.linspace(- dim_x / 2 + 1, dim_x / 2, dim_x)
    Y = np.linspace(- dim_y / 2 + 1, dim_y / 2, dim_y)
    x, y = np.meshgrid(X, Y)

    mask = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) <= r

    return mask


def normalize_img(image, window, mask=None):
    """
    Normalizing fonction based on Foracchia's Luminosity-Contrast 
    normalization scheme

    Inputs:
    -------

    image (array): The input image to normalize
    window (array): The window/filter function (i.e. m_function)
    mask (array): The mask to select the region of interest

    Outputs:
    --------
    normalized_img (array): The normalized image

    """

    eps = 1E-7

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]))

    mask = mask.astype(int)

    
    imMean = convolve2d(image * mask, window, mode='same') / (convolve2d(mask, window, mode='same') + eps)
    imMeanSq = convolve2d((image ** 2) * mask, window, mode='same') / (convolve2d(mask, window, mode='same') + eps)
    std = np.sqrt(np.abs(imMeanSq - imMean ** 2))

    background = (1 - np.abs(image - imMean) / (std + eps)) >= 0
    background = background.astype(int)

    imMean = convolve2d(image * mask * background, window, mode='same') / (convolve2d(mask * background, window, mode='same') + eps)
    imMeanSq = convolve2d((image ** 2) * mask * background, window, mode='same') / (convolve2d(mask * background, window, mode='same') + eps)
    std = np.sqrt(np.abs(imMeanSq - imMean ** 2))

  
    return np.tanh(8 * (image - imMean) / (std + eps))