from main import extract_nifti_data, threshold_data, get_mean
import nibabel as nib
import numpy as np
import os
import logging

def test_extract_nifti_data(tmpdir):
    data = np.ones((32, 32, 15, 100), dtype=np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    path = os.path.join(tmpdir, "test_img.nii.gz")
    nib.save(img, path)
    loaded_data = extract_nifti_data(path)
    logging.warning(tmpdir)
    assert np.array_equal(data, loaded_data), "Loading is incorrect"


def test_threshold_data():
    data = np.random.randn(4,4)
    threshold = 0.1
    thresholded_data = threshold_data(data, threshold)

    assert np.all(thresholded_data > threshold), "thresholding incorrect"


def test_get_mean():
    data = np.ones((4,4))
    average = get_mean(data)

    assert average == 1, "mean incorrect"
