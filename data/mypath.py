class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            # foler that contains leftImg8bit/
            #return r'F:\dataset\cityspaces'
            return r'/workshop/user_data/hgh/data/cityspaces'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError("undefined dataset {}.".format(dataset))
