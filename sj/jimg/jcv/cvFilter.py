
class Filter:
    def __init__(self):
        self.subFilters = []

    def addFilter(self, flt):
        self.subFilters.append(flt)

    def onFrame(self,filter_frame, draw_frame):
        for flt in self.subFilters:
            flt.onFrame(filter_frame, draw_frame)

    def onEvent(self, event_name, parameters=None):
        if event_name == self.__class__.__name__:
            pass

    def onKey(self,key):
        for flt in self.subFilters:
            flt.onKey(key)

    def getName(self):
        subNames = []
        for flt in self.subFilters:
            subNames.append(flt.getName())
        if len(subNames)>0:
            return self.__class__.__name__+"-"+",".join(subNames)
        else:
            return self.__class__.__name__