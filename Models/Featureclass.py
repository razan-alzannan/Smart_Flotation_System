class Feature:
    def __init__(self, feature_name=None, output1=None, output2=None, output3=None):
        self._feature_name = feature_name
        self._output1 = output1
        self._output2 = output2
        self._output3 = output3

    @property
    def feature_name(self):
        return self._feature_name

    @feature_name.setter
    def feature_name(self, value):
        self._feature_name = value

    @property
    def output1(self):
        return self._output1

    @output1.setter
    def output1(self, value):
        self._output1 = value

    @property
    def output2(self):
        return self._output2

    @output2.setter
    def output2(self, value):
        self._output2 = value

    @property
    def output3(self):
        return self._output3

    @output3.setter
    def output3(self, value):
        self._output3 = value