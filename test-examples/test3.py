import tensorflow
import itertools
import random

from time import time


class ClassifiedNumber:
    __number = 0
    __classifiedAs = 3


    def __init__(self, number):

        self.__number = number

        if number == 0:
            self.__classifiedAs = 0  # zero

        elif number > 0:
            self.__classifiedAs = 1  # positive

        elif number < 0:
            self.__classifiedAs = 2  # negative

    def number(self):
        return self.__number

    def classifiedAs(self):
        return self.__classifiedAs


def classifiedAsString(classifiedAs):
    if classifiedAs == 0:
        return "Zero"

    elif classifiedAs == 1:
        return "Positive"

    elif classifiedAs == 2:
        return "Negative"


def trainDatasetFunction():
    trainNumbers = []
    trainNumberLabels = []

    for i in range(-1000, 1001):
        number = ClassifiedNumber(i)
        trainNumbers.append(number.number())
        trainNumberLabels.append(number.classifiedAs())

    return ({"number": trainNumbers}, trainNumberLabels)


def inputDatasetFunction():
    global randomSeed
    random.seed(randomSeed)  # to get same result

    numbers = []

    for i in range(0, 4):
        numbers.append(random.randint(-9999999, 9999999))

    print(numbers)
    return {"number": numbers}


def main():
    print("TensorFlow Positive-Negative-Zero numbers classifier test by demensdeum 2017 (demensdeum@gmail.com)")

    maximalClassesCount = len(set(trainDatasetFunction()[1])) + 1

    numberFeature = tensorflow.feature_column.numeric_column("number")
    classifier = tensorflow.estimator.DNNClassifier(feature_columns=[numberFeature], hidden_units=[10, 20, 10],
                                                    n_classes=maximalClassesCount)


    generator = classifier.train(input_fn=trainDatasetFunction, steps=1000).predict(input_fn=inputDatasetFunction)

    inputDataset = inputDatasetFunction()

    results = list(itertools.islice(generator, len(inputDataset["number"])))

    i = 0
    for result in results:
        print("number: %d classified as %s" % (inputDataset["number"][i], classifiedAsString(result["class_ids"][0])))
        i += 1


randomSeed = time()

main()