import h5py

try:
    import fsspec
    from petrel_client.client import Client
except: 
    Client=None
    pass


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class HDF5Singleton(type):
    _instances = {}
    _file_paths = [] 
    def __call__(cls, file_path, mode='r',libver="latest", swmr=True,  rdcc_nbytes=1024**2*15):
        # file_path = kwargs.get('file_path') 
        # if cls not in cls._instances:
        if file_path not in cls._file_paths:
            # cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[file_path] = super(HDF5Singleton, cls).__call__(file_path, mode, libver, swmr, rdcc_nbytes)
            cls._file_paths.append(file_path) 
        # else:
            # cls._instances[file_path].__init__(*args, **kwargs)

        return cls._instances[file_path]

class HDF5File(metaclass=HDF5Singleton):
# class HDF5File(metaclass=Singleton):

    def __init__(self, file_path, mode='r',libver="latest", swmr=True, rdcc_nbytes=1024**2*15 ):
        self.file_path = file_path


        if Client is not None:
            conf_path = '~/petreloss.conf'
            print("HDF5 is opening...", flush=True)
            client = Client(conf_path)
            url = client.generate_presigned_url('s3://Bucket/all.h5')
            with fsspec.open(url) as f:
                # hf = h5py.File(f)
                # self.file = h5py.File(f, mode=mode, libver=libver,swmr=swmr, rdcc_nbytes=rdcc_nbytes) 
                self.file = h5py.File(f, mode=mode, )#libver=libver,swmr=swmr, rdcc_nbytes=rdcc_nbytes) 
            print("HDF5 is opened!", flush=True)
        else:
            self.file = h5py.File(file_path, mode=mode, libver=libver,swmr=swmr, rdcc_nbytes=rdcc_nbytes) 

            
        # self.file = h5py.File(file_path, mode=mode, libver=libver,swmr=swmr, rdcc_nbytes=rdcc_nbytes) 

    def read(self):
        # print(self._instances,flush=True)
        return self.file


