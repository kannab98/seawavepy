from json import load
import logging
import sys, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('modeling.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.info('Welcome to project repo: https://github.com/kannab98/modeling')

class rcParams():
    logger.info('Load config  from %s' % os.path.join(os.getcwd(), 'rc.json'))
    with open("rc.json") as f:
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







        # return rc

# Кастомный list
# class alist(collections.UserList):
#     def __init__(self, *args, **kwargs):
#         data = kwargs.pop('data')
#         super().__init__(self, *args, **kwargs)
#         self.data = data

#     def append(self, item):
# #         print('No appending allowed.')
#         return self.data.append(item)



# def getset(name, getting, setting):

#     return property(lambda self: getting(getattr(self, name)),
#                     lambda self, val: setattr(self, name, setting(val)))
# name = "wind"
# val = 10
# class Foo(object):                                   

#     def __init__(self):               
#         self._wind = 15



    
# value.call.append(lambda x: x)
# d = Foo()
# print(d.wind)
#     return None

# d.fset = fset
# d.value = 15
# 
# 

# config  = kwargs["config"] if "config" in kwargs else os.path.join(os.path.abspath(os.getcwd()), "rc.json")






rc = rcParams()
 
# from . import Spectrum
# spectrum = Spectrum.__spectrum__()

# from modeling import Surface
# surface = Surface.__surface__()


# from . import retracking
# retracking = retracking.retracking.__retracking__()
# brown = Retracking.brown()


