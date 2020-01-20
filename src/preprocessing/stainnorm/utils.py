import numpy as np
import cv2
try:
    import spams
except:
    pass

def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True


def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def get_sign(x):
    """
    Returns the sign of x.

    :param x: A scalar x.
    :return: The sign of x.
    """

    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.

    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels.

    :param I: Image RGB uint8.
    :return:
    """
    assert is_uint8_image(I), "Should be a RGB uint8 image"
    I = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    I_float = I.astype(np.float32)
    I1, I2, I3 = cv2.split(I_float)
    I1 /= 2.55  # should now be in range [0,100]
    I2 -= 128.0  # should now be in range [-127,127]
    I3 -= 128.0  # should now be in range [-127,127]
    return I1, I2, I3

def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8.

    :param I1: L
    :param I2: A
    :param I3: B
    :return: Image RGB uint8.
    """
    I1 *= 2.55  # should now be in range [0,255]
    I2 += 128.0  # should now be in range [0,255]
    I3 += 128.0  # should now be in range [0,255]
    I = np.clip(cv2.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(I, cv2.COLOR_LAB2RGB)

def get_concentrations(I, stain_matrix, regularizer=0.01):
    """
    Estimate concentration matrix given an image and stain matrix.

    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


def luminosity_standardise(I, percentile=95):
    """
    Transform image I to standard brightness.
    Modifies the luminosity channel such that a fixed percentile is saturated.

    :param I: Image uint8 RGB.
    :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
    :return: Image uint8 RGB with standardized brightness.
    """
    assert is_uint8_image(I), "Image should be RGB uint8."
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L_float = I_LAB[:, :, 0].astype(float)
    p = np.percentile(L_float, percentile)
    I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
    I = cv2.cvtColor(I_LAB, cv2.COLOR_LAB2RGB)
    return I

def luminosity_get_tissue_mask(I, luminosity_threshold=0.8):
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically we use to identify tissue in the image and exclude the bright white background.
    :param I: RGB uint 8 image.
    :param luminosity_threshold: Luminosity threshold.
    :return: Binary mask.
    """
    assert is_uint8_image(I), "Image should be RGB uint8."
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
    if mask.sum() == 0:
        print("Empty tissue mask computed")

    return mask

