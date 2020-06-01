import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# from generic_functions import *

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
        for date in self.dates:
            for polar in polarisation:
                print(date)
                temp = os.path.join(self.path, '*_' + str(date) + '_*' + polar + '_S' + str(segment) + '*')
                print(temp)
                filename = temp + '.slc'
                filename_ann = temp + '.ann'
                print(filename)
                print(glob.glob(filename))
                import sys
                sys.exit(0)

        self.unique_identifiers_time_list = list()
        for entry in list(self.meta_data.keys()):
            print(entry)
            unique_identifiers_time = '_'.join(entry.split('_')[:-2])[:-2] + "POL_" + \
                                      '_'.join(entry.split('_')[-2:]) + '_sSEGMENT'
            if unique_identifiers_time not in self.unique_identifiers_time_list:
                self.unique_identifiers_time_list.append(unique_identifiers_time)

        # Then we read the files one by one for each polarisation and time
            self.data = np.empty((crop_indexes[1]-crop_indexes[0], crop_indexes[3]-crop_indexes[2], len(polarisation), 
                        len(self.unique_identifiers_time_list)), dtype='complex64')

            for t, entry_time in enumerate(self.unique_identifiers_time_list):
                for i_pol, pol in enumerate(polarisation):
                    # Read slc file at the given crop indexes
                    filename = entry_time.replace('POL', pol).replace('SEGMENT', str(segment))
                    shape = (int(self.meta_data['_'.join(filename.split('_')[:-1])]['slc_1_1x1 Rows']),
                             int(self.meta_data['_'.join(filename.split('_')[:-1])]['slc_1_1x1 Columns']))   
                    print("Reading %s" % (self.path+filename))
                    with open(os.path.join(self.path, filename + '_1x1.slc'), 'rb') as f:
                        f.seek((crop_indexes[0]*shape[1]+crop_indexes[2])*8, os.SEEK_SET)
                        for row in range(crop_indexes[1]-crop_indexes[0]):
                            self.data[row, :, i_pol,t] = np.fromfile(f, dtype=np.complex64, count=crop_indexes[3]-crop_indexes[2])
                            f.seek(((crop_indexes[0]+row)*shape[1]+crop_indexes[2])*8, os.SEEK_SET)
            import sys
            sys.exit(0)


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------------------------------
    # Reading UAVSAR dataset
    # ---------------------------------------------------------------------------------------------------------------
    print('READING DATASET')
    # Reading data using the class
    import time
    t_beginning = time.time()
    data_class = uavsar_slc_stack_1x1('data')
    crop_indexes = [20000,23000, 3000,4500]
    data_class.read_data(polarisation=['HH', 'HV', 'VV'], segment=2, crop_indexes=crop_indexes)
    image = data_class.data 
    print(image.shape)
    fig = plt.figure(figsize=(23,13), dpi=80)
    dyn = 70
    plt.imshow(20*np.log10(np.abs(data_class.data[:,:,0,0])), aspect='auto', cmap='gray', vmin=20*np.log10(np.abs(data_class.data.max())) - dyn, vmax=20*np.log10(np.abs(data_class.data.max())))
    plt.axis('off')
    fig.tight_layout()
    plt.show()
    print("Done in %f s." % (time.time()-t_beginning))
