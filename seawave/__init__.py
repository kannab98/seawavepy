from json import load
import logging
import sys, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('modeling.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.debug('Welcome to project repo: https://github.com/kannab98/seawavepy')

class rcParams():

    def __init__(self, file=None):



        defaultrc = os.path.join(os.path.dirname(__file__), 'rc.json')
        cwdrc = os.path.join(os.getcwd(), 'rc.json')

        if file != None and os.path.isfile(file):
            logger.info('Load config from %s' % file)

        elif os.path.isfile(cwdrc):
            logger.info('Load config from cwd')
            file = cwdrc

        elif os.path.isfile(defaultrc):
            logger.info('Load default config ')
            file = defaultrc

        self.__json2object__(file)


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

rc = rcParams()


