from scipy.stats import randint,uniform

LIGHTGM_PARAMS = {
    'n_estimators':randint(100,500),
    'max_depth': randint(5,50),
    'learning_rate':uniform(0.01,0.2),
    'num_leaves': randint(20,100),
    'boosting_type':['gbdt', 'datt', 'goss'],
    }


RANDOM_SEARCH_PARAMS = {
    'n_iter' : 50,              # Number of combinations to try
    "cv" : 5,                   # 5-fold cross-validation
    "scoring" : 'f1',           # You can use 'accuracy', 'roc_auc', etc.
    "verbose" : 2,
    "n_jobs" : -1,              # Use all cores
    "random_state" : 42
}