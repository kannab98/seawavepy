


import logging
import re, os
logger = logging.getLogger(__name__)

def get_files(file, **kwargs):
    """
    Рекурсивный поиск данных по регулярному выражению 
    """
    # file = file.replace('\\', os.path.sep)
    # file = file.replace('/', os.path.sep)
    path, file = os.path.split(file)

    path = os.path.abspath(path)

    # file = os.path.join(path, file)
    rx = re.compile(file)


    _files_ = []
    for root, dirs, files in os.walk(path, **kwargs):
        for file in files:
            tmpfile = os.path.join(root, file)
            _files_ += rx.findall(tmpfile)
    
    for file in _files_:
        logger.info("Found file: %s" % file)

    return _files_

from . import retracking_new

from . import retracking
retracking = retracking.__retracking__()