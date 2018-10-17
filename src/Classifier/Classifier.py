import abc


class Classifier:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, data_in):
        """
        Apply the classifier and get the classification result
        :param data_in: input data, a 1-D numpy array
        :return: the class number and its probability, both scalar
        """
        pass

    @abc.abstractmethod
    def train(self, data_in, label):
        """
        Train the classifier if needed.
        Please handle the parameter update and the storage inside the class.
        :param data_in: input data, a 1-D numpy array
        :param label: the ground truth, a scalar
        :return: statistics data needed.
        """
        pass
