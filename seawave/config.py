from . import CONFIG
from json import load
import logging
logger = logging.getLogger(__name__)


class rcParams():
    logger.info('Load config  from %s' % CONFIG)
    with open(CONFIG) as f:
        __rc__ = load(f)

    def __init__(self, **kwargs):
        self.__json2object__("rc.json")


    def __json2object__(self, file):

#     """
#     Преобразование полей конфигурационного файла rc.json 
#     в объекты класса и присвоение им соответствующих значений


#     Файл из json вида:

#         >> { ... "swell": { ... "speed": [ 10, ... ] }, ... }

#     преобразуется в переменную __rc__ типа dict, поля выделенные под размерности и комментарии отбрасываются):

#         >> __rc__["swell"] = { ... , "speed": 10 } 

#     после словарь __rc__ становится объектом класса:

#         >> rc.swell.speed
#         >> out: 10
#     """
        with open(file) as f:
            __rc__ = load(f)



        for Key, Value in __rc__.items():
            setattr(self, Key, type('rc', (object,), {}))
            attr = getattr(self, Key)
            # setattr(attr, "call", {})
            for key, value in Value.items():
                setattr(attr, key, value[0])