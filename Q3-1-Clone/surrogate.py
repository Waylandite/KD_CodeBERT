from sklearn import linear_model, neural_network



def predictor(dataset):
    X_train, y_train = dataset[0], dataset[1]
    # X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2)
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)

    # preds = reg.predict(X_test)
    # logging.info("MAE: {}".format(mean_absolute_error(y_test, preds)))

    return reg
