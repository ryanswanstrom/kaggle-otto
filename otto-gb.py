from __future__ import division

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
import threading


#np.random.seed(456)
results = []


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


def load_train_data(train_size=0.8):
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
    
def train_grad(training_x, training_y, n_est=10, max_d=5, max_f='sqrt'):
    clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d, max_features=max_f)
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

class GBThread (threading.Thread):
    def __init__(self, X_train, y_train, X_valid, y_valid, n_est, max_d, max_f):
        threading.Thread.__init__(self)
        self.training_x = X_train
        self.training_y = y_train
        self.validation_x = X_valid
        self.validation_y = y_valid
        self.n_est = n_est
        self.max_d = max_d
        self.max_f = max_f
    def run(self):
        print 'GB thread running: n_estimators = {}, max_depth = {}, max_features = {}'.format(self.n_est, self.max_d, self.max_f)
        gboost(self.training_x, self.training_y, self.validation_x, self.validation_y, self.n_est, self.max_d, self.max_f)

def gboost(training_x, training_y, validation_x, validation_y, n_est=12, max_d=5, max_f='sqrt'):
    start = timer()
    
    grad_model = train_grad(training_x, training_y, n_est, max_d, max_f)
    y_true = encoder(grad_model, validation_y)
    
    score = validate_model(grad_model, validation_x, y_true)
    
    end = timer();
    elapsed_time = (end - start)
    
    print '{} {} {} elapsed time is {} score is {}'.format(n_est, max_d, max_f, elapsed_time,score)
    #results.append({'score':score,'model':type(grad_model).__name__,'n_estimators':n_est,'max_depth':max_d,'max_features':max_f,'time':elapsed_time})  
    
  

def main():
    print(" - Start.")
    X_train, X_valid, y_train, y_valid = load_train_data()
    threads = []
    
    # set n_estimators to be 650, and now check max_depth and  max_features
    poss_max_features = [1, 'sqrt','log2']
    n_est = 20
    for max_d in range(15, 20, 5): 
        for idx, max_f in enumerate(poss_max_features):
              
            # Create new threads
            thread = GBThread(X_train.copy(), y_train.copy(), X_valid.copy(), y_valid.copy(), n_est, max_d, max_f)
            
            # Start new Thread
            print 'starting thread'
            thread.start()            
            threads.append(thread)
            
        
    for t in threads:
        t.join()
        
    pd.DataFrame(results).to_csv('results\\results7.csv')
        
    #make_submission(rf, encoder)
    print(" - Finished.")


if __name__ == '__main__':
    main()
