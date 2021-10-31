import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # get the most common element in the k neighbors
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("wdbc.data.mb.csv", names=[x for x in range(31)])
    data = z_score(df.loc[:, :29])
    X, y = data.loc[:, :29], df.loc[:, 30]
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=.3, random_state=1234)

    for k in [1, 3, 5, 7, 9]:
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        confusion_matrix = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
        for index, value in enumerate(pred):
            if value == y_test[index] and value == 1:
                confusion_matrix["TP"] += 1
            if value == y_test[index] and value == -1:
                confusion_matrix["TN"] += 1
            if value != y_test[index] and value == 1:
                confusion_matrix["FP"] += 1
            if value != y_test[index] and value == -1:
                confusion_matrix["FN"] += 1
        print("--------confusion matrix for k = ",k,"-----------------")
        tm = "\t +1\t\t -1\n +1\t TP: " + str(confusion_matrix["TP"]) + "\t FN: " + str(
            confusion_matrix["FN"]) + "\n -1\t FP: " + str(confusion_matrix["FN"]) + "\t TN: " + str(
            confusion_matrix["TN"])

        print(tm)
        accuracy = np.sum(pred == y_test) / len(y_test)
        print("\n Accuracy: ",accuracy)
        print("_+_+_+_+_+_+_+_+End of k = ",k,"_+_+_+_+_+_+_+_+_+_+_+_+_+\n")
