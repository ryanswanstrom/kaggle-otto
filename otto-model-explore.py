from __future__ import division

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.preprocessing import LabelEncoder


#np.random.seed(17411)


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss

    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_train_data(path=None, train_size=0.8):
    df = pd.read_csv('train.csv')
    X = df.values.copy()
    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))

def encoder(model, validation_y):
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(validation_y)
    assert (encoder.classes_ == model.classes_).all()
    return y_true
    

def load_test_data(path=None):
    df = pd.read_csv('test.csv')
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)

def validate_model(model, validation_x, validation_y):
    y_prob = model.predict_proba(validation_x)
    score = logloss_mc(validation_y, y_prob)
    print(" -- {} Multiclass logloss on validation set: {:.4f}.".format(type(model).__name__, score))
    return score

def train_rf(training_x, training_y, n_est=10, max_d=5, max_f='auto'):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_est, max_depth=max_d, max_features=max_f)
    clf.fit(training_x, training_y)
    return clf
    
def train_ex(training_x, training_y, n_est=10, max_d=5, max_f='auto'):
    clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=n_est, max_depth=max_d, max_features=max_f)
    clf.fit(training_x, training_y)
    return clf
    
#def train_ada(training_x, training_y, n_est=10):
#    clf = AdaBoostClassifier(n_estimators=n_est)
#    clf.fit(training_x, training_y)
#    return clf
    
def train_grad(training_x, training_y, n_est=10):
    clf = GradientBoostingClassifier(n_estimators=n_est)
    clf.fit(training_x, training_y)
    return clf


def make_submission(clf, encoder, path='my_submission.csv'):
    X_test, ids = load_test_data()
    y_prob = clf.predict_proba(X_test)
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))


def main():
    print(" - Start.")
    X_train, X_valid, y_train, y_valid = load_train_data()
    
    # set n_estimators to be 650, and now check max_depth and  max_features
    poss_max_features = ['sqrt']
    n_est = 650
    results = []
    for max_d in range(50,100,3): 
        for idx, max_f in enumerate(poss_max_features):
        
            print 'max_depth = {}, max_features = {}'.format(max_d, max_f)
            rf_model = train_rf(X_train, y_train, n_est, max_d, max_f)
            y_true = encoder(rf_model, y_valid)
            score = validate_model(rf_model, X_valid, y_true)
            results.append({'score':score,'model':type(rf_model).__name__,'n_estimators':650,'max_depth':max_d,'max_features':max_f})
            
            ex_model = train_ex(X_train, y_train, n_est, max_d, max_f)
            y_true = encoder(ex_model, y_valid)
            score = validate_model(ex_model, X_valid, y_true)
            results.append({'score':score,'model':type(ex_model).__name__,'n_estimators':650,'max_depth':max_d,'max_features':max_f})
            
        #    grad_model = train_grad(X_train, y_train, i)
        #    y_true = encoder(grad_model, y_valid)
        #    score = validate_model(grad_model, X_valid, y_true)
        #    results.append({'score':score,'model':type(grad_model).__name__,'n_estimators':i})
        #
            pd.DataFrame(results).to_csv('results\\results4.csv')
        
    #make_submission(rf, encoder)
    print(" - Finished.")


if __name__ == '__main__':
    main()
