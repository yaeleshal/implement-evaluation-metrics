from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LR

# A
def adjust_labels_to_binary (y_train, target_class_value):
    types_dict = {'Setosa': 0 , 'Versicolour': 1 ,'Virgincacv': 2}
    return np.where(y_train == types_dict[target_class_value],1, -1)


# B
def one_vs_rest( x_train, y_train, target_class_value):
    y_train_binary = adjust_labels_to_binary(y_train, target_class_value)
    lr = LR()
    return lr.fit(x_train, y_train_binary)

# C
def binarized_confusion_matrix (X, y_binarized, one_vs_rest_model, prob_threshold):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    y_pred = one_vs_rest_model.predict_proba(X)[:,1]
    y_pred_binary = np.zeros(np.size(y_binarized))
    for i in range(len(y_pred)):
        if y_pred[i] >= prob_threshold:
            y_pred_binary[i] = 1
        else:
            y_pred_binary[i] = -1

        if y_pred_binary[i] == 1 and y_binarized[i] == 1:
            TP += 1
        if y_pred_binary[i] == 1 and y_binarized[i] == -1:
            FP += 1
        if y_pred_binary[i] == -1 and y_binarized[i] == 1:
            FN += 1
        if y_pred_binary[i] == -1 and y_binarized[i] == -1:
            TN += 1
    return [[TP, FP], [FN, TN]]

#D
X, y = load_iris(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7, random_state=98)
for type in ['Setosa', 'Versicolour', 'Virgincacv']:
    model = one_vs_rest(X_train, Y_train, type)

    y_train_binary = adjust_labels_to_binary(Y_train, type)
    cm_train = binarized_confusion_matrix(X_train, y_train_binary, model, 0.5)
    print("train CM for: ", type, "is: ", cm_train)

    y_test_binary = adjust_labels_to_binary(Y_test, type)
    cm_test = binarized_confusion_matrix(X_test, y_test_binary, model, 0.5)
    print("test CM for: ", type, "is: ", cm_test)


# E
def micro_avg_precision(X, y, all_targed_class_dict, prob_threshold):
    TP = 0
    FP = 0
    for type in ['Setosa', 'Versicolour', 'Virgincacv']:
        Y_binarized = adjust_labels_to_binary(y, type)
        cm = binarized_confusion_matrix(X, Y_binarized, all_targed_class_dict[type], prob_threshold)
        TP += cm[0][0]
        FP += cm[0][1]
    micro_ap = TP / (TP + FP)
    return micro_ap


# F
def micro_avg_recall(X, y, all_targed_class_dict, prob_threshold):
    TP = 0
    FN = 0
    for type in ['Setosa', 'Versicolour', 'Virgincacv']:
        Y_binarized = adjust_labels_to_binary(y, type)
        cm = binarized_confusion_matrix(X, Y_binarized, all_targed_class_dict[type], prob_threshold)
        TP += cm[0][0]
        FN += cm[1][0]
    micro_ar = TP / (TP + FN)
    return micro_ar


# G
def micro_avg_false_positive_rate(X, y, all_targed_class_dict, prob_threshold):
    FP = 0
    TN = 0
    for type in ['Setosa', 'Versicolour', 'Virgincacv']:
        Y_binarized = adjust_labels_to_binary(y, type)
        cm = binarized_confusion_matrix(X, Y_binarized, all_targed_class_dict[type], prob_threshold)
        TN += cm[1][1]
        FP += cm[0][1]
    micro_afpr = FP / (TN + FP)
    return micro_afpr


# H
FPR = []
TPR = []
all_targed_class_dict = {}

for type in ["Setosa", "Versicolour", "Virgincacv"]:
    all_targed_class_dict[type] = one_vs_rest(X_train, Y_train, type)

for t in [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]:
    FPR.append(micro_avg_false_positive_rate(X_test, Y_test, all_targed_class_dict, t))
    TPR.append(micro_avg_recall(X_test, Y_test, all_targed_class_dict, t))

plt.figure(figsize=(10, 6))
plt.scatter(FPR, TPR)
plt.plot(FPR, TPR)
plt.plot([0, 0.2, 0.4, 0.6, 1], [0, 0.2, 0.4, 0.6, 1], linestyle='dashed', color='orange')
plt.xlim([-0.05, 1.1])
plt.ylim([-0.05, 1.1])
plt.ylabel('TPR - micro average recall')
plt.xlabel('FPR - micro average false positive rate')
plt.title('ROC curve for test data')
plt.show()


# I
def f_beta(precision, recall, beta):
    f_b = (1 + beta ** 2) * (precision * recall / (beta ** 2 * precision + recall))
    return f_b


# J
fb_dict = {0.3: [], 0.5: [], 0.7: []}
for th in [0.3, 0.5, 0.7]:
    precision = micro_avg_precision(X_test, Y_test, all_targed_class_dict, th)
    recall = micro_avg_recall(X_test, Y_test, all_targed_class_dict, th)
    for beta in range(11):
        fb_dict[th] += [f_beta(precision, recall, beta)]

x = range(11)
plt.figure(figsize=(10, 6))

for t, curr_color in zip([0.3, 0.5, 0.7], ['orange', 'green', 'purple']):
    plt.plot(x, fb_dict[t], color=curr_color)

plt.legend(['fb_0.3', 'fb_0.5', 'fb_0.7'])
plt.ylabel('fb values')
plt.xlabel('beta')
plt.title('fb values as a function of beta for probability thresholds for test data')
plt.show()

