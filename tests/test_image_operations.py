"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import sys
import unittest

from arl.image.iterators import *
from arl.image.operations import *
from arl.util.testing_support import create_test_image
from arl.util.run_unittests import run_unittests


log = logging.getLogger(__name__)

class TestImage(unittest.TestCase):

    def setUp(self):
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi

    def test_create_image_from_array(self):
        m31model_by_array = create_image_from_array(self.m31image.data, wcs=None)
        
        m31model_by_array = create_image_from_array(self.m31image.data, self.m31image.wcs)
        m31modelsum = add_image(self.m31image, m31model_by_array)
        m31modelsum = add_image(self.m31image, m31model_by_array, checkwcs=True)
        assert m31model_by_array.shape == self.m31image.shape
        log.debug(export_image_to_fits(self.m31image, fitsfile='%s/test_model.fits' % (self.dir)))
        log.debug(qa_image(m31model_by_array, context='test_create_from_image'))

    def test_create_empty_image_like(self):
        emptyimage = create_empty_image_like(self.m31image)
        assert emptyimage.shape == self.m31image.shape
        assert numpy.max(numpy.abs(emptyimage.data)) == 0.0

    def test_checkwcs(self):
    
        cellsize = 1.5 * self.cellsize
        newwcs = self.m31image.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
        with self.assertRaises(AssertionError):
            checkwcs(self.m31image.wcs, newwcs)
    
    def test_reproject(self):
        # Reproject an image
    
        cellsize = 1.5 * self.cellsize
        newwcs = self.m31image.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
    
        newshape = numpy.array(self.m31image.data.shape)
        newshape[2] /= 1.5
        newshape[3] /= 1.5
        newimage, footprint = reproject_image(self.m31image, newwcs, shape=newshape)
        checkwcs(newimage.wcs, newwcs)
              
    def test_show_image(self):
        show_image(self.m31image)

if __name__ == '__main__':
    run_unittests()
