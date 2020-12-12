import numpy as np
import pandas as pd

def crossvalidation_list(LC, DS, m):
    """ List[Classifieur] * tuple[array, array] * int ->  List[tuple[tuple[float,float], tuple[float,float]]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    print("Il y a ", len(LC), "classifieurs à comparer.")
    accuracy_test = {}
    accuracy_train = {}
    for C in LC:
        accuracy_test[C] = []
        accuracy_train[C] = []
        
    size = DS[0].shape[0]//m
    for i in range(m):
        desc_test = DS[0][i*size:(i+1)*size,:] 
        desc_train_1 = DS[0][0:i*size,:]
        desc_train_2 = DS[0][(i+1)*size:DS[0].shape[0],:]
        desc_train = np.vstack((desc_train_1,desc_train_2))
        
        label_test = DS[1][i*size:(i+1)*size] 
        label_train_1 = DS[1][0:i*size]
        label_train_2 = DS[1][(i+1)*size:DS[0].shape[0]]
        label_train = np.concatenate([label_train_1,label_train_2],axis = 0)
        
        for C in LC: 
            C.train(desc_train, label_train)        
            accuracy_test[C].append(C.accuracy(desc_test, label_test))
            accuracy_train[C].append(C.accuracy(desc_train, label_train))
            
    result = []
    l_test = list(accuracy_test.values())
    l_train = list(accuracy_train.values())
    for i in range(len(l_test)): 
        result.append(((np.mean(np.array(l_train[i])), np.std(np.array(l_train[i]))), (np.mean(np.array(l_test[i])), np.std(np.array(l_test[i])))))

    return result
#------------------------------
def crossvalidation(C, DS, m=10):
    """ Classifieur * tuple[array, array] * int -> tuple[tuple[float,float], tuple[float,float]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    accuracy_test = []
    accuracy_train = []
    size = DS[0].shape[0]//m
    for i in range(m):
        desc_test = DS[0][i*size:(i+1)*size,:] 
        desc_train_1 = DS[0][0:i*size,:]
        desc_train_2 = DS[0][(i+1)*size:DS[0].shape[0],:]
        desc_train = np.vstack((desc_train_1,desc_train_2))
        
        label_test = DS[1][i*size:(i+1)*size] 
        label_train_1 = DS[1][0:i*size]
        label_train_2 = DS[1][(i+1)*size:DS[0].shape[0]]
        label_train = np.concatenate([label_train_1,label_train_2],axis = 0)
        
        
        C.train(desc_train, label_train)        
        accuracy_test.append(C.accuracy(desc_test, label_test))
        accuracy_train.append(C.accuracy(desc_train, label_train))
    return (np.mean(np.array(accuracy_train)), np.std(np.array(accuracy_train))), (np.mean(np.array(accuracy_test)), np.std(np.array(accuracy_test)))
#------------------------------------------

def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    cpt = 0   
    for i in range(DS[0].shape[0]):
        x_desc = DS[0][i]
        rest_desc = np.delete(DS[0], i, axis=0)
        
        x_label = DS[1][i]
        rest_label = np.delete(DS[1], i, axis=0)
        
        C.reset()
        C.train(rest_desc, rest_label)
        if C.predict(x_desc) == C.predict(x_label):
            cpt = cpt + 1
    
    return cpt/DS[0].shape[0]

#--------------------------------------------------
