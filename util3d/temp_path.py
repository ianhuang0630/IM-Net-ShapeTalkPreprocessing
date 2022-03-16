import os
import shutil
import uuid

shapenet_dir = os.path.dirname(__file__)
tmp_dir = '/tmp'

if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)


def get_temp_path(filename_fn=lambda x: x):
    path = os.path.join(tmp_dir, filename_fn(uuid.uuid4().hex))
    while os.path.isfile(path):
        path = filename_fn(uuid.uuid4().hex)
    return path


class TempDir(object):
    def __init__(self, path):
        self._path = path

    def __enter__(self):
        if os.path.isdir(self._path):
            raise IOError('Directory already exists at %s' % self._path)
        os.makedirs(self._path)
        return self._path

    def __exit__(self, *args, **kwargs):
        shutil.rmtree(self._path)
        # pass


def get_temp_dir(filename_fn=lambda x: x):
    return TempDir(get_temp_path(filename_fn))
