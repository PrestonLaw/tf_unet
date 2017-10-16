# TODO:
# - Make the files loaded configurable.
# - When selecting batch images, skip images that have no classes (somewhere in the middle)
# - Augment images with flip x, flip y
# Augment images with aarbitrary rotation
# Augment images with noise.
# Make option to turn augmentation on and off.


# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.




'''
Created on Sept 20, 2017

author: Preston Law
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import os
import scipy
from scipy import misc
from tf_unet.image_util import BaseDataProvider


class GrayscaleSlideDataProvider(BaseDataProvider):
    """
    Extends the BaseDataProvider to randomly select the next 
    image region
    """
    
    channels = 1
    n_class = 2
    
    # files is a list of file name pairs [(image1.png,mask1.png), ....]
    # There has to be at least one file pair and the images have to have
    # shape larger or equal to (nx,ny).
    # nx, ny is the shape/size of the arrays returned for training.
    # a_min, a_max is the clipping range for image pixel values.
    # Image and mask arrays returned have element ranges [0,1]
    def __init__(self, basepath, file_names, shape=(512,512), a_min=0, a_max=255):
        
        # Read the slide images using scipy, put them into a list.
        self.images = []
        self.masks = []

        for file_pair in file_names:
            self.images.append(scipy.misc.imread(os.path.join(basepath,file_pair[0])))
            self.masks.append(scipy.misc.imread(os.path.join(basepath,file_pair[1])))
            
        super(GrayscaleSlideDataProvider, self).__init__(a_min, a_max)
        self.file_names = file_names
        self.nx = shape[0]
        self.ny = shape[1]

        # Randomly select a current image to sample
        self._cylce_file()
    
    def _sample_region(self):
        (width, height) = self.images[self.file_idx].shape
        #(width, height, _) = self.images[self.file_idx].shape
        # TODO: fix potential infinite loop
        while width < self.nx or height < self.nx:
            print("file %s too small to sample"%self.file_names[self.file_idx][0])
            self._cylce_file()
            (width, height) = self.images[self.file_idx].shape
            #(width, height, _) = self.images[self.file_idx].shape
        
        x = np.random.randint(0, width - self.nx)
        y = np.random.randint(0, height - self.ny)

        # Crop sample regions out of the larger arrays
        data = self.images[self.file_idx][x:x+self.nx, y:y+self.ny, ...]
        rfi = self.masks[self.file_idx][x:x+self.nx, y:y+self.ny, ...]

        # Turn mask into an array of bools.
        rfi = rfi > 128

        return data, rfi
    
    def _next_data(self):
        # Randomly choose an image to sample
        self._cylce_file()
        # Extract regions.
        image, mask = self._sample_region()

        # Skip regions that are all background. 
        # TODO: Consider checking central sub region.
        while np.max(mask) == False:
            self._cylce_file()
            image, mask = self._sample_region()

        # Start augmenting with simple flips.
        # Augmentation has to be here (and not in _post_process)
        # because with rotation we will ahve to sample a larger region
        # to avoid clipping the corners.
        if bool(np.random.randint(0,2)):
            # Flip over the x axis.
            np.flip(image, 0)
            np.flip(mask, 0)
        if bool(np.random.randint(0,2)):
            # Flip over the y axis.
            np.flip(image, 1)
            np.flip(mask, 1)
        
        return image, mask

    
    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.images))
        
