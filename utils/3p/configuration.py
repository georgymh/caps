from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
from time import localtime, strftime
import re


class Config(object):
    """ Configuration object."""
    def __init__(self, **kwargs):
        for key, value in sorted(kwargs.iteritems()):
            setattr(self, key, value)

    def save_config_to_file(self):
        params = sorted(self.__dict__.keys(), key=self._natural_keys)
        f_dump = ['%s = %s' % (k, self.__dict__[k]) for k in params]
        config_name = 'config___%s.txt' % strftime('%Y-%m-%d_%H-%M-%S', localtime())
        with open(os.path.join(self.log_path, config_name), 'w') as f:
            f.write('\r\n'.join(f_dump))
        # Save the dictionary instead of the object for maximum flexibility
        # Avoid this error:
        # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
        with open(os.path.join(self.log_path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def overwrite_safety_check(self, overwrite):
        """ Exits if log_path exists but `overwrite` is set to `False`."""
        path_exists = os.path.exists(self.log_path)
        if path_exists:
            if not overwrite:
                print('\nINFO: log_path already exists. '
                      'Set `overwrite` to True? Exiting now.')
                raise SystemExit
            else: print('\nINFO: log_path already exists. '
                        'The directory will be overwritten.')
        else:
            print('\nINFO: log_path does not exist. '
                  'The directory will be created.')
            #os.mkdir(self.log_path)
            os.makedirs(self.log_path)

    @staticmethod
    def _natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        def atoi(text):
            return int(text) if text.isdigit() else text
        return [atoi(c) for c in re.split('(\d+)', text)]


def load_config(config_filepath):
    with open(config_filepath, 'rb') as f:
        c_dict = pickle.load(f)
    config = Config(**c_dict)
    return config
