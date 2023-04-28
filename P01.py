# 21805105 Yann KIBAMBA
# 22210574 Ibrahim Soilahoudine 
# 22210566 Nabil Loudaoui

# Section 1: Les imports
import numpy as np
import operator
import P01_utils as p
import scipy as sp
from scipy.spatial.distance import cdist


# Section 2: Définitions de fonction
# 3.2.1
data_train = p.lire_donnees(100)
print(data_train)
data_test = p.lire_donnees(10)
print(data_test)
print(p.visualiser_donnees(data_train[0],data_train[1], X_test=None))
print(p.visualiser_donnees(data_test[0],data_test[1], X_test=None))

# 3.3.1
def dist(X_i,X_j):
    distance_euclidienne = ((X_i[0]-X_j[0])**2+(X_i[1]-X_j[1])**2)**(1/2)
    return distance_euclidienne

# 3.3.2
def indice_k_plus_proches(donnees_train,point,k):
    liste_knn=[]
    for indice, nombre in enumerate(donnees_train):
        distance = dist(nombre,point)
        liste_knn.append((distance,indice))
    liste_knn= sorted(liste_knn)
    indice_k = [indice for distance, indice in liste_knn[:k]]
    return indice_k

#3.3.3
def classe(liste):
    count_classe = 0
    dico= {}
    for element in liste:
        if element in dico:
            dico[element] += 1
        else:
            dico[element]=1
    return max(dico.items(), key=operator.itemgetter(1))[0]
# 3.3.4
def k_plus_proches_voisins_liste(jeu_train,jeu_test,k=1):
    prediction=[]
    liste_indice=[]
    liste_final=[]
    for point in jeu_test[0]:
            liste=[]
            for indice, nombre in enumerate(jeu_train[0]):
                distance = dist(nombre,point)
                liste.append([distance,indice])
            liste= sorted(liste)
            prediction.append(liste[:k])                  # La liste prediction contient la distance et l'indice des k plus proches voisins de chaque point de jeu de test
            for liste_prediction in prediction:
                liste1=[]
                for place in liste_prediction:            # On cherche à récupérer dans chaque liste contenue dans prediction, les indices 
                    liste1.append(place[1])
            liste_indice.append(liste1)
            for liste_sexe in liste_indice:
                sexe = classe(jeu_train[1][liste_sexe])   # Avec la fonction "classe" de la question 3, on prédit le sexe de chaque liste_sexe contenue dans liste1
            liste_final.append(sexe)
    return liste_final
# 3.4
def  k_plus_proches_voisins_numpy(data_train,data_test,k=1):
    matrice_distance = cdist(data_test[0],data_train[0])
    longueur= int(len(data_test[0])/2)
    ## Trions par ligne la matrice_distance avec argsort afin de recuiellir les indices des plus proches voisins
    trie_matrice = np.argsort(matrice_distance, axis=1)
    gender = data_train[1][trie_matrice[:,0:k]]=="F"
    gender_plus_present = np.sum(gender,axis=1)
    liste=[]
    for i in gender_plus_present:
        if i>=longueur:
            liste.append("F")
        else:
            liste.append("H")
    return liste

# Section 3: Test
# 3.3.1
print(dist((2,7),(1,5)))

# 3.3.2
# Test sur coordonnées quelconques
coord = (1,1)
coord_1 = [(0,0),(4,5),(-1,-1),(1,3),(7,10),(2,3)]
print(indice_k_plus_proches(coord_1,coord,3))
# Test sur le jeu de données data_train
print(indice_k_plus_proches(data_train[0],(0,175),3))

# 3.3.3
f = ["F","F","H"]
h = ["F","F","H","H","H"] 
print(classe(f))
print(classe(h))
# 3.3.4
print(k_plus_proches_voisins_liste(data_train,data_test,5))
# 3.4
print(k_plus_proches_voisins_numpy(data_train,data_test,5))

