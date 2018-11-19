import abc
from src.Classifier.Naive_Bayes_Classifier import Naive_Bayes_Classifier
from src.Classifier.RandomForest_Classifier import RandomForest_Classifier
from src.Classifier.SVM_Classifier import SVM_Classifier
from src.Classifier.KNN_Classifier import KNN_Classifier
from src.Classifier.DecisionTree_Classifier import DecisionTree_Classifier


class Classifier:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, data_in):
        """
        Apply the classifier and get the classification result
        :param data_in: input data, a 2-D numpy array [feature, num of data]
        :return: the class number and its probability, both scalar
        """
        pass

    @abc.abstractmethod
    def train(self, data_in, label):
        """
        Train the classifier if needed.
        Please handle the parameter update and the storage inside the class.
        :param data_in: input data, a 2-D numpy array [feature, num of data]
        :param label: the ground truth, a scalar
        :return: statistics data needed.
        """
        pass


def classifier_dict():
    classifiers = {
        'naive_bayes': Naive_Bayes_Classifier,
        'random_forest': RandomForest_Classifier,
        'SVM': SVM_Classifier,
        'KNN': KNN_Classifier,
        'Decision_Tree': DecisionTree_Classifier
    }
    return classifiers

