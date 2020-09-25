import autograd.numpy as np
import glob
import matplotlib.pyplot as plt
import os
import sys
import time

class uavsar_slc_stack_1x1():
    """A class to store data corresponding to a SLC stack of UAVSAR data
        * path = path to a folder containing the files obtained from UAVSAR (.slc, .ann, .llh)"""
    def __init__(self, path):
        super(uavsar_slc_stack_1x1, self).__init__()
        self.path = path
        self.meta_data = dict()
        self.llh_grid = dict()
        self.slc_data = dict()
        self.dates = list()

    def read_data(self, polarisation=['HH', 'HV', 'VV'], segment=2, crop_indexes=[20000,23000, 3000,4500]):
        """ A method to read UAVSAR SLC 1x1 data stack
            Inputs:
                * polarisation = a list of polarisations to read
                * crop_indexes = if we want to read a portion of the image, a list of the form
                    [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1]"""
        
        # Iterate on annotation files and get all dates of the time series
        regex_annotations = os.path.join(self.path, '*.ann')
        for path in glob.glob(regex_annotations):
            filename = path.split('.')[0].split('/')[1]
            self.dates.append(filename.split('_')[4])
            self.meta_data[filename] = dict() # initialise dict for the currect file
            with open(path, 'r') as f:
                for line in f:
                    # Discard commented lines
                    line = line.strip().split(';')[0]
                    if not (line == ''):
                        category = ' '.join(line.split()[:line.split().index('=')-1])   
                        value = ' '.join(line.split()[line.split().index('=')+1:])
                        self.meta_data[filename][category] = value
 
        # remove duplicate and sort dates
        self.dates = list(set(self.dates))
        self.dates.sort()
        
        # Read slc file corresponding to the segment of interest and crop it
        self.data = np.empty((crop_indexes[1]-crop_indexes[0], crop_indexes[3]-crop_indexes[2], len(polarisation), len(self.dates)), dtype='complex64')
        for t, date in enumerate(self.dates):
            for i_pol, polar in enumerate(polarisation):
                regex = os.path.join(self.path, '*_' + str(date) + '_*' + polar)
                
                regex_slc = regex + '*_s' + str(segment) + '*.slc'
                path_slc = glob.glob(regex_slc)[0]
                
                regex_ann = regex + '*.ann'
                path_ann = glob.glob(regex_ann)[0]

                print('Reading:', path_slc) 

                with open(path_slc, 'rb') as f:
                    filename_ann = path_ann.split('/')[-1].split('.')[0]
                    shape = (int(self.meta_data[filename_ann]['slc_1_1x1 Rows']),
                             int(self.meta_data[filename_ann]['slc_1_1x1 Columns']))   
                    f.seek((crop_indexes[0]*shape[1]+crop_indexes[2])*8, os.SEEK_SET)
                    for row in range(crop_indexes[1]-crop_indexes[0]):
                             self.data[row, :, i_pol, t] = np.fromfile(f, dtype=np.complex64, count=crop_indexes[3]-crop_indexes[2])
                             f.seek(((crop_indexes[0]+row)*shape[1]+crop_indexes[2])*8, os.SEEK_SET)


if __name__ == '__main__':
    t_beginning = time.time()
    
    # Reading UAVSAR dataset
    print('READING UAVSAR DATASET')
    data_class = uavsar_slc_stack_1x1('data')
    crop_indexes = [20000,23000, 3000,4500]
    data_class.read_data(polarisation=['HH', 'HV', 'VV'], segment=2, crop_indexes=crop_indexes)
    images = data_class.data 
    print(images.shape)
    
    # Saving image time series UAVSAR as numpy array
    np.save('Snjoaq', images)
    print("Done in %f s." % (time.time()-t_beginning))
