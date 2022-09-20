"""
Takes care of data splitting into training / testing / evaluation
"""
import sklearn.model_selection as skm


def train_test_split(data, labels, test_fraction=0.2):
    """
    Splits data into train and test, using stratify.
    data : data to split ( along 0 axis )
    labels : 2D boolean array of labels
    :return: arrays with split data and split labels
            (X_train, X_test, y_train, y_test)

    """
    X_train, X_test, y_train, y_test = skm.train_test_split(data, labels,
                                                            test_size=test_fraction,
                                                            stratify=labels,
                                                            shuffle=True)
    return X_train, X_test, y_train, y_test


def train_test_val_split(data, labels, test_fraction=0.2, val_fraction=0.2, verbose=False):
    """
    Splits data into train and test, using stratify.
    data : data to split ( along 0 axis )
    labels : 2D boolean array of labels
    :return: arrays with split data and split labels
            (X_train, X_test, X_val, y_train, y_test, y_val)

    """

    withheld_fraction = test_fraction + val_fraction
    X_train, X_test_val, y_train, y_test_val = skm.train_test_split(data, labels,
                                                                    test_size=withheld_fraction,
                                                                    stratify=labels,
                                                                    shuffle=True)
    X_val, X_test, y_val, y_test = skm.train_test_split(X_test_val, y_test_val,
                                                        test_size=test_fraction / withheld_fraction,
                                                        stratify=y_test_val,
                                                        shuffle=True)
    if verbose:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"y_val shape: {y_val.shape}")

    return X_train, X_test, X_val, y_train, y_test, y_val

# TODO :
# undersampling
# oversampling
