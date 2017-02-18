#Two files from exercise were merged here
# 1. quiz.py which contains method definitions
# 2. sandbox.py which is nothing but a test case and uses methods from tensorFlow.py

#=====================Code from quiz.py goes here========================
import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    #pass
    batch_arr = [[features[x:x+batch_size], labels[x:x+batch_size]] for x in range(0, len(features), batch_size)]
    return batch_arr

#===========================Code from quiz.py ends======================

#=====================Code from sandbox.py goes here========================

#from quiz import batches
from pprint import pprint

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))

#for fea, lbl in batches(3, example_features, example_labels):
#    print fea
#    print lbl

#===========================Code from sandbox.py ends======================
