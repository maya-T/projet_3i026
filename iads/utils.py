# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from iads import evaluation as ev
import math


# ------------------------ 
def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o') # 'o' pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])    
    
# ------------------------ 
def genere_dataset_uniform(p, n, borneInf=-1,borneSup=1):
    data_desc=np.random.uniform(borneInf,borneSup,(2*n,p))
    data_label= np.asarray([1 for i in range(0,n)] + [-1 for i in range(0,n)])
    tup=(data_desc,data_label)
    return tup  
    
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    data_desc1=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)    
    data_desc2=np.random.multivariate_normal(negative_center,negative_sigma,nb_points) 
    data_desc=np.vstack([data_desc1,data_desc2])
    data_label= np.asarray([1 for i in range(0,nb_points)] + [-1 for i in range(0,nb_points)])
    tup=(data_desc,data_label)
    return tup
# ------------------------ 
def create_XOR(nb_points, var):
    data_desc1=np.random.normal(np.array([0,0]),var,size=(nb_points//2,2)) 
    data_desc2=np.random.normal(np.array([1,1]),var,size=(nb_points//2,2)) 
    data_desc3=np.random.normal(np.array([1,0]),var,size=(nb_points//2,2)) 
    data_desc4=np.random.normal(np.array([0,1]),var,size=(nb_points//2,2)) 
    data_desc5=np.vstack([data_desc1,data_desc2,data_desc3,data_desc4])
    data_label=np.asarray([1 for i in range(0,(nb_points))] + [-1 for i in range(0,(nb_points))])
    tup=(data_desc5,data_label)
    return tup
 # ------------------------ 
def mean(labels):
    return sum(labels) / len(labels)
# ------------------------ 

def mode(labels):
    return Counter(labels).most_common(1)[0][0]
# ------------------------ 
def euclidean_distance(point1, point2):
    
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)
# ------------------------ 
def euclidean_distance2(a, b):
    return np.linalg.norm(a-b) 
# ------------------------ 
def normalise(X):
    mean = np.mean(X, axis = 0)
    if mean != 0 :
	    std = np.std(X, axis =0)
	    return (X - mean) / std
    return X.copy()

#-----------------------

def categories_2_numeriques(DF,nom_col_label =''):
    """ DataFrame * str -> DataFrame
        nom_col_label est le nom de la colonne Label pour ne pas la transformer
        si vide, il n'y a pas de colonne label
        rend l'équivalent numérique de DF
    """
    dfloc = DF.copy()  # pour ne pas modifier DF
    L_new_cols = []    # pour mémoriser le nom des nouvelles colonnes créées
    Noms_cols = [nom for nom in dfloc.columns if nom != nom_col_label]
     
    for c in Noms_cols:
        if dfloc[c].dtypes != 'object':  # pour détecter un attribut non catégoriel
            L_new_cols.append(c)  # on garde la colonne telle quelle dans ce cas
        else:
            for v in dfloc[c].unique():
                nom_col = c + '_' + v    # nom de la nouvelle colonne à créer
                dfloc[nom_col] = 0
                dfloc.loc[dfloc[c] == v, nom_col] = 1
                L_new_cols.append(nom_col)
            
    return dfloc[L_new_cols]  # on rend que les valeurs numériques
#-------------------------------
class AdaptateurCategoriel:
    """ Classe pour adapter un dataframe catégoriel par l'approche one-hot encoding
    """
    def __init__(self,DF,nom_col_label=''):
        """ Constructeur
            Arguments: 
                - DataFrame représentant le dataset avec des attributs catégoriels
                - str qui donne le nom de la colonne du label (que l'on ne doit pas convertir)
                  ou '' si pas de telle colonne. On mémorise ce nom car il permettra de toujours
                  savoir quelle est la colonne des labels.
        """
        self.DF = DF  # on garde le DF original  (rem: on pourrait le copier)
        self.nom_col_label = nom_col_label 
        
        # Conversion des colonnes catégorielles en numériques:
        self.DFcateg = categories_2_numeriques(DF, nom_col_label)
        
        # Pour faciliter les traitements, on crée 2 variables utiles:
        self.data_desc = self.DFcateg.values
        self.data_label = self.DF[nom_col_label].values
        # Dimension du dataset convertit (sera utile pour définir le classifieur)
        self.dimension = self.data_desc.shape[1]
                
    def get_dimension(self):
        """ rend la dimension du dataset dé-catégorisé 
        """
        return self.dimension
        
        
    def train(self, classifieur):
        """ Permet d'entrainer un classifieur sur les données dé-catégorisées 
        """   
        
        classifieur. train(self.data_desc, self.data_label)
        
    
    def accuracy(self, classifier):
        """ Permet de calculer l'accuracy d'un classifieur sur les données
            dé-catégorisées de l'adaptateur.
            Hypothèse: le classifieur doit avoir été entrainé avant sur des données
            similaires (mêmes colonnes/valeurs)
        """
        return classifier.accuracy(self.data_desc,self.data_label)

    def converti_categoriel(self, x):
        
        """ transforme un exemple donné sous la forme d'un dataframe contenant
            des attributs catégoriels en son équivalent dé-catégorisé selon le 
            DF qui a servi à créer cet adaptateur
            rend le dataframe numérisé correspondant             
        """
        data = [[0 for i in range(self.data_desc.shape[1])]]
        xloc = pd.DataFrame(data, columns = self.DFcateg.columns)
        for c in x.columns:
            if c != self.nom_col_label:
                if x[c].dtypes != 'object':  # pour détecter un attribut non catégoriel
                    xloc[c] = x[c]
                else:
                    xloc[c + '_' + x[c]] = 1
        return xloc  # on rend que les valeurs numériques
       
    def predict(self,x,classifier):
        """ rend la prédiction de x avec le classifieur donné
            Avant d'être classifié, x doit être converti
        """
        x_df = self.converti_categoriel(x)
        return classifier.predict(x_df[self.DFcateg.columns].values)
    def crossvalidation(self, classifier, m):
        return ev.crossvalidation(classifier, (self.data_desc, self.data_label), m)
    def crossvalidation_list(self, list_classifiers, m):
        return ev.crossvalidation_list(list_classifiers, (self.data_desc, self.data_label), m)
#-------------------------------

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.argmax(nb_fois)]

#-------------------------------
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    if 1 in P :
        return 0
    res = 0
    for p in P :
        if p != 0 :
            res = res + p*math.log(p, len(P))
    return -res
#-------------------------------

def entropie(Y):
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    total = Y.shape[0]
    P = []
    for n in nb_fois:
        P.append(n/total)
    return shannon(P)
#----------------------------------------

def discretise(desc, labels, col):
   
    ind= np.argsort(desc,axis=0)
    
    total = desc.shape[0]

    entropies = []
    for i in range(ind.shape[0] - 1):
        ind_i = ind[i]   # vecteur d'indices
        courant = desc[ind_i[col], col]
        ind_prochain = ind[i+1]
        prochain = desc[ind_prochain[col], col]
        seuil = (courant + prochain) / 2.0;
        
        
        inf_y = labels[desc[:,col] <= seuil]
        sup_y = labels[desc[:,col] > seuil]
        
        
        e1 = entropie(inf_y)
        e2 = entropie(sup_y)
                
        moyenne = e1 * inf_y.shape[0]/ total + e2 * sup_y.shape[0]/ total
        
        entropies.append(moyenne)

        
    i_best = np.argmin(entropies) 
    min_entropie = entropies[i_best]
    
    ind_i = ind[i_best]   # vecteur d'indices
    courant = desc[ind_i[col], col]
    ind_prochain = ind[i_best+1]
    prochain = desc[ind_prochain[col], col]
    
    meilleur_seuil = (courant + prochain) / 2.0;
    
    return (meilleur_seuil, min_entropie )

#-----------------------------------

def divise(desc, labels, att, seuil):
    indexess_sup = np.argwhere(desc[:, att] > seuil )
    indexes_sup =  np.reshape(indexess_sup, indexess_sup.shape[0])
    
    indexess_inf = np.argwhere(desc[:, att] <= seuil )
    indexes_inf =  np.reshape(indexess_inf, indexess_inf.shape[0])  
    
    desc_inf = desc[indexes_inf, :]
    desc_sup = desc[indexes_sup, :]
    
    labels_inf = labels[indexes_inf]
    labels_sup = labels[indexes_sup]

    return (desc_inf, labels_inf), (desc_sup, labels_sup)


