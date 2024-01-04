import scipy
import numpy as np

def build_binary_classifier(num, trainX, trainY):
    binaryY = np.copy(trainY)
    binaryY[binaryY != num] = -1
    binaryY[binaryY == num] = 1
    
    trainX = trainX / np.max(trainX)
    # print(trainY)
    # print("max: " + str(np.min(trainX)) + " min: " + str(np.max(trainX)))
    n = (trainX.shape)[0]
    A = np.hstack((np.ones((n,1)), trainX))
    
    # print("A after adding 1's column: " + str(A))
    # print("Values of A: " + str(np.unique(A)))
    
    ATA = np.matmul(A.transpose(), A)
    ATA_pinv = np.linalg.pinv(ATA)
    ATy = np.matmul(A.transpose(), binaryY)
    return np.matmul(ATA_pinv, ATy)
    
# trainX, trainY, testX, testY

def load_train_data(filename, str_type):
    pixels = scipy.io.loadmat(filename)
    
    # trainX and trainY should be scipy 2D arrays, and should have the same number of rows

    pixels[str_type+"Y"] = pixels[str_type+"Y"].transpose()
    
    if(str_type == "train"):
        pixels[str_type+"Y"]=pixels[str_type+"Y"]
        pixels[str_type+"X"] = pixels[str_type+"X"]

    return pixels[str_type+"X"].astype(float), pixels[str_type+"Y"]

def build_one_vs_all_classifier(labels, trainX, trainY):
    # the k-th column of classifier is the one vs all classifier for the label k
    classifier = np.empty([trainX.shape[1]+1, labels.shape[0]])
    for label in labels:
        classifier[:,label:(label+1)] = build_binary_classifier(label, trainX, trainY)
    return classifier

def test_one_vs_all_classifier(classifier, labels):
    testX, testY = load_train_data("/Users/pranavreddy/Desktop/ECE 174/Project1/mnist.mat", "test")
    n = (testX.shape)[0]

    testX = testX / np.max(testX)

    testX = np.hstack((np.ones((n,1)), testX))
    predictions = np.matmul(testX, classifier)

    # print(predictions)

    print("Shape of testX " + str(testX.shape))
    print("Max of testX " + str(np.max(testX)))
    print("Min of testX " + str(np.min(testX)))

    predictions = np.argmax(predictions, axis=1)

    print("Shape of predictions " + str(predictions.shape))
    print("Max of predictions " + str(np.max(predictions)))
    print("Min of predictions " + str(np.min(predictions)))

    return predictions, build_confusion(predictions, testY, labels)

def build_confusion(predictions, test_data, labels):
    confusion_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    num_wrong = 0
    for i in range(len(test_data)):
        if(predictions[i] != test_data[i]):
            num_wrong = num_wrong + 1
        confusion_matrix[test_data[i]][predictions[i]] += 1
    print(num_wrong / len(test_data))
    return confusion_matrix

trainX, trainY = load_train_data("/Users/pranavreddy/Desktop/ECE 174/Project1/mnist.mat", "train")
print("trainY: " + str(trainY.shape))
labels = np.arange(0, 10)
binary_test = build_binary_classifier(5, trainX, trainY)
# theta = build_one_vs_all_classifier(labels, trainX, trainY)
# print("Binary classifier: " + str(binary_test))
# print("Binary classifier shape: " + str(binary_test.shape))
# print(theta)
predictions, confusion_matrix = test_one_vs_all_classifier(binary_test, labels)
# print(predictions.shape)
print(predictions)
# print(confusion_matrix)