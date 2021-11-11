#!/usr/bin/python3
# coding=utf-8

##############################
## 演示基于scikit-learn
## 实现不同分类器的训练和推理
##############################

import numpy as np

# 朴素Bayes分类器
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model

# KNN分类器
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

# Logistic回归分类器
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

# 随机森林分类器
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# 决策树分类器
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

# GBDT分类器
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    return model

# SVM分类器
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM分类器(参数搜索交叉验证)
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [10**n for n in range(-2,3)], 'gamma': [10**n for n in range(-4,2)]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print('best param value')
        print(para,': ',val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

## 分类性能矩阵
# 使用例子
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.autumn_r):
    if classes is None:
        classes=list(range(cm.shape[0]))
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normlized confusion matrix')
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


## LDA数据降维
# x是待降维数据矩阵，每一行是一个数据样本的特征向量
# y是待降维的数据标签
def reduce_dim_LDA(x,y,Q=2):
    yset=set(y)                     # 数据标签
    num_label=len(yset)             # 数据标签总数
    num_data,num_feature=x.shape    # 样本数和样本维度
    
    if Q>=num_feature:
        return x.copy(),np.eye(num_feature)
    
    # 类计数
    type_cnt={t:np.sum(y==t) for t in yset}
    
    # 计算类中心
    x_mean={t:np.mean(x[y==t,:],axis=0) for t in yset}
        
    # 计算类内方差
    x_cov ={t:np.dot(x[y==t,:].T,x[y==t,:])/type_cnt[t]-np.dot(x_mean[t].T,x_mean[t]) for t in yset}
    
    # 计算类内方差和
    x_cov_all=np.zeros((num_feature,num_feature))
    for v in x_cov.values(): x_cov_all+=v
    
    # 计算数据中心
    x_mean_all=np.zeros((1,num_feature))
    for v in x_mean.values(): x_mean_all+=v
    
    # 计算类间方差
    t_cov=np.zeros((num_feature,num_feature))
    for t in yset: 
        m=x_mean[t]-x_mean_all
        t_cov+=type_cnt[t]*np.dot(m.T,m)
    
    # 计算降维矩阵W
    U,D,V=np.linalg.svd(np.dot(np.linalg.inv(x_cov_all),t_cov))
    W=U[:,:Q]
    z=np.dot(x,W)
    z=np.dot(x,W)
    
    return z,W

####################
# 测试入口
####################

if __name__=='__main__':
    from sklearn import metrics
    
    # 所有分类器列表
    classifiers_list = {'NB':(naive_bayes_classifier        , 'naive bayes classifier'          ),
                       'KNN':(knn_classifier                , 'knn classifier'                  ),
                        'LR':(logistic_regression_classifier, 'logistic regression classifier'  ),
                        'RF':(random_forest_classifier      , 'random forest classifier'        ),
                        'DT':(decision_tree_classifier      , 'decision tree classifier'        ),
                       'SVM':(svm_classifier                , 'svm classifier'                  ),
                     'SVMCV':(svm_cross_validation          , 'svm cross validation'            ),
                      'GBDT':(gradient_boosting_classifier  , 'gradient boosting classifier'    )} 


    # 加载测试数据
    from sklearn.datasets import load_digits
    digits = load_digits()
    x,y=digits['data'],digits['target']

    # 训练/测试数据集分离
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.5,shuffle=True)

    # 分类器训练和验证
    for name in ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM', 'SVMCV']:
        cls    = classifiers_list[name][0]  # 选取分类器
        model  = cls(train_x, train_y)      # 分类器训练
        
        # 分类器测试
        pred_y = model.predict(test_x)
        acc    = metrics.accuracy_score(test_y, pred_y)
        cmat=confusion_matrix(test_y, pred_y)
        
        # 显示分类器性能
        acc_info='acc: %.2f%%' % (100 * acc)
        full_name=classifiers_list[name][1]
        plt.clf()
        plot_confusion_matrix(cmat,title= full_name+': '+acc_info)
        plt.show()

