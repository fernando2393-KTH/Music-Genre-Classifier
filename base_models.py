import numpy as np
import loader
import load_csv_data as loader1
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def explain_cumulutive_var(pca, n_components, data):
    exp_variance = pca.explained_variance_ratio_

    print("Explained variance ratio: \n{}".format(pca.explained_variance_ratio_))
    print("\n")
    print("Number of components = {}".format(pca.n_components_))  

    # plot the explained variance using a barplot
    fig, ax = plt.subplots()
    ax.bar(range(n_components), exp_variance)
    ax.set_xlabel('number of components')
    ax.set_ylabel('explained variance')
    plt.show()

    # Calculate the cumulative explained variance
    cum_exp_variance = np.cumsum(exp_variance)

    # Plot the cumulative explained variance and draw a dashed line at 0.90.
    fig, ax = plt.subplots()
    ax.plot(range(n_components), cum_exp_variance)
    ax.axhline(y=0.9, linestyle='--')
    pca_projection = pca.transform(data)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.get_train_val_test(mode='spectrogram')
    (x_train0, y_train0), (x_val0, y_val0), (x_te0, y_test0) = loader1.get_train_val_test(
        "./Datasets/spectrogram_augment_2.csv")
    (x_train1, y_train1), (x_val1, y_val1), (x_te1, y_test1) = loader1.get_train_val_test(
        "./Datasets/spectrogram_augment_3.csv")
    (x_train2, y_train2), (x_val2, y_val2), (x_te2, y_test2) = loader1.get_train_val_test(
        "./Datasets/spectrogram_augment_4.csv")
    (x_train3, y_train3), (x_val3, y_val3), (x_te3, y_test3) = loader1.get_train_val_test(
        "./Datasets/spectrogram_augment_5.csv")

    x_train = np.hstack((x_train0, x_train1, x_train2, x_train3))
    x_val = np.hstack((x_val0, x_val1, x_val2, x_val3))
    x_te = np.hstack((x_te0, x_te1, x_te2, x_te3))

    y_train = np.hstack((y_train0, y_train1, y_train2, y_train3))
    y_val = np.hstack((y_val0, y_val1, y_val2, y_val3))
    y_test = np.hstack((y_test0, y_test1, y_test2, y_test3))

    """
    x_train = np.hstack((x_train0, x_train1))
    x_val = np.hstack((x_val0, x_val1))
    x_te = np.hstack((x_te0, x_te1))

    y_train = np.hstack((y_train0, y_train1))
    y_val = np.hstack((y_val0, y_val1))
    y_test = np.hstack((y_test0, y_test1))

 
    # Remove small samples
    x_train = np.delete(x_train, [3495, 3496, 3497])
    y_train = np.delete(y_train, [3495, 3496, 3497])

    """

    x_train = np.rollaxis(np.dstack(x_train), -1)
    x_val = np.rollaxis(np.dstack(x_val), -1)
    x_te = np.rollaxis(np.dstack(x_te), -1)
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_te = np.expand_dims(x_te, axis=3)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])
    x_te = x_te.reshape(x_te.shape[0], x_te.shape[1]*x_te.shape[2])

    scaler = StandardScaler()

    # Fit on training set only.
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    train_sc = scaler.transform(x_train)
    cv_sc = scaler.transform(x_val)
    test_sc = scaler.transform(x_te)

    for n_compts in range(2, 20, 1):
        pca = PCA(n_components=n_compts)
        pca.fit(train_sc)
        train_pca = pca.transform(train_sc)
        cv_pca = pca.transform(cv_sc)
        test_pca = pca.transform(test_sc)
        explain_cumulutive_var(pca, n_compts, train_sc)

        test_accuracies = []
        neigh = KNeighborsClassifier(n_neighbors=18, weights='distance')
        neigh.fit(train_pca, y_train)
        train_preds = neigh.predict(train_pca)
        train_acc = np.sum(train_preds == y_train)
        train_acc = train_acc / len(y_train)
        cv_preds = neigh.predict(cv_pca)
        cv_acc = np.sum(cv_preds == y_val)
        cv_acc = cv_acc / len(y_val)
        test_preds = neigh.predict(test_pca)
        test_acc = np.sum(test_preds == y_test)
        test_acc = test_acc / len(y_test)
        test_accuracies.append(test_acc)

        print('KNN')
        print(' Neigh:', 18, 'Train Accuracy: ', train_acc, "\tValidation Accuracy: ", cv_acc, "\tTest Accuracy: ", test_acc)

        # SVM
        classifier = svm.SVC(gamma='scale', verbose=False)
        classifier.fit(train_pca, y_train)
        train_preds = classifier.predict(train_pca)
        train_acc = np.sum(train_preds == y_train)
        train_acc = train_acc / len(y_train)
        cv_preds = classifier.predict(cv_pca)
        cv_acc = np.sum(cv_preds == y_val)
        cv_acc = cv_acc / len(y_val)
        test_preds = classifier.predict(test_pca)
        test_acc = np.sum(test_preds == y_test)
        test_acc = test_acc / len(y_test)

        print('SVM')
        print('Train Accuracy: ', train_acc, "\tValidation Accuracy: ", cv_acc, "\tTest Accuracy: ", test_acc)


if __name__ == "__main__":
    main()
