from skopt.space import Real, Categorical, Integer

# search_spaces_xgboost = {
#         'learning_rate': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#         'min_child_weight': [1,2,3,4,5,6,7,8,9,10],
#         'max_depth': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
#         'max_delta_step': [0,1,2,3,4,5,6,7,8,9,10],
#         'subsample': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#         'colsample_bytree': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#         'colsample_bylevel': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#         'reg_lambda': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000],
#         'reg_alpha': (1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1),
#         'gamma': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,5e-1],
#         'min_child_weight': [1,2,3,4,5,6,7,8,9,10],
#         'scale_pos_weight': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100],
#     }
search_spaces_xgboost = {
        'n_estimators': Integer(100, 1000, prior='uniform'),
         'learning_rate': Real(1e-6, 1e-1, prior='log-uniform'),
         'max_depth':Integer(3, 20, prior='uniform'),
         'max_leaves':Integer(0, 50, prior='uniform'),
         'grow_policy': ['depthwise', 'lossguide'],
         'gamma': Real(1e-9, 5e-1, prior='log-uniform'),
    }