# Compte Rendu TD1 Classical Machine Learning and Deep Learning
# Layth Chebbi 3-SSIR-A

1. L'IA est l'imitation d'une réponse d'un expert dans un domaine. Le ML est de l'IA reposant sur des statistiques (Le ML adapte son algorithme mais a besoin d'un expert pour les données) Le DL est du ML utilisé avec des réseaux de neurones (Le DL extrait l'information pertinente des données et adapte son algorithme)

2. Un jeu de données est formé par N objets (observation) notés  Chaque objet est caractérisé par   descripteurs, ou attributs. Ainsi la taille de la base données est × .
Une technique d'apprentissage automatique est appelé à trouver un relation de dépendance
entre les objets en entrées et un jeu d'étiquettes en sortie (nommés également targets ou
labels)
3.   
    *  Explorer ou vérifier, représenter, décrire,        les    variables, leurs liaisons et positionner les observations de l'échantillon
    * Expliquer ou tester l'influence d'une variable ou facteur dans un modèle supposé connu a priori
    * Prévoir et sélectionner un meilleur ensemble de prédicteurs comme par exemple dans la recherche de bio-marqueurs
4. 
    * #### Attribut:
    Un attribut est une propriété et ou une caractéristique de l'objet.
    * #### Descripteur:
    Descripteur: Un descripteur est une quantité mesurable ou calculable qui permet de décrire en partie un objet, un signal, une donnée… 
    * #### Etiquette:
    Les données étiquetées sont des données marquées, annotées, afin de présenter la  » cible « . Il s’agit de la réponse, que l’on souhaite que le modèle de Machine Learning apprenne à prédire.
    * ##### Classification:
    La classification statistique réside dans l’identification de la catégories à laquelle un nouveau élément appartient, sur la base d'un data set d’entraînement de données contenant des observations (ou instances) dont la catégorie est connue.
    * #### Regle de classification:
    Les principaux algorithmes du machine learning avec supervision sont les suivants : forêts aléatoires, arbres décisionnels, méthode du k plus proche voisin (k-NN), régression linéaire, classification naïve bayésienne, machine à vecteurs de support (SVM), régression logistique et boosting des gradients.
    * #### Fonction de kernel:
    En apprentissage automatique, l'astuce du noyau, ou kernel trick en anglais, est une méthode qui permet d'utiliser un classifieur linéaire pour résoudre un problème non linéaire.
5.  
    L'apprentissage supervisé est réalisé sur un jeu de données étiquetés. C'est à dire que les données d'entraînement fournis au départ à l'algorithme comportent les solutions désirées, appelées étiquettes (en anglais, labels). L'objectif revient donc à classer correctement un nouvel exemple.
6. 
    Toutes les données sont étiquetées et les algorithmes apprennent à prédire le résultat des données d'entrée. Non supervisé: toutes les données ne sont pas étiquetées et les algorithmes apprennent la structure inhérente à partir des données en entrée.
7. 
8. 
    *  La validation croisée permet de tirer plusieurs   ensembles de validation d'une même base de données et ainsi d'obtenir une estimation plus robuste, avec biais et variance, de la performance de validation du modèle.

    * La validation croisée k-fold signifie que l’ensemble de données se divise en un nombre K. Elle divise l’ensemble de données au point où l’ensemble de test utilise chaque pli. Comprenons le concept à l’aide de la validation croisée à 5 volets ou K+5. Dans ce scénario, la méthode divise l’ensemble de données en cinq volets. Le modèle utilise le premier pli dans la première itération pour tester le modèle. Il utilise les autres ensembles de données pour former le modèle. Le deuxième pli aide à tester l’ensemble de données et les autres soutiennent le processus de formation. Le même processus se répète jusqu’à ce que l’ensemble de test utilise chaque pli des cinq plis.

9. Tableau de confusion : 

![](https://www.lebigdata.fr/wp-content/uploads/2018/12/confusion-matrix-exemple-1024x576.jpg)
10.  * ### Precision:

        TP / ( TP + FP )
    * ### Recall:

        TP / ( TP + FN )
    * ### Accuracy:

        ( TP + TN ) / N
11.  * ROC: 

![](https://miro.medium.com/max/576/1*MIQXB9LDkPoHunwEXOa8Cg.png)

* ### UAC:

    L’AUC aide à comparer les différents classificateurs. Vous pouvez résumer les performances de chaque classificateur en une seule mesure. L’approche de base pour trouver la CUA est de calculer l’AUROC. Elle est similaire à la probabilité que l’instance négative aléatoire soit inférieure à l’instance positive. Si un classificateur a une CUA inférieure à celle d’un autre classificateur, cela signifie normalement que le score de la CUA élevée n’est pas bon. Cependant, la SSC fonctionne bien dans le cadre de la mesure générale de la précision prédictive.

12. Le processus de classification modélise une fonction par laquelle les données sont prédites dans des étiquettes de classe discrètes. D'autre part, la régression est le processus de création d'un modèle qui prédit une quantité continue.
Les algorithmes de classification impliquent un arbre de décision, une régression logistique, etc. En revanche, un arbre de régression (par exemple, une forêt aléatoire) et une régression linéaire sont des exemples d'algorithmes de régression.
La classification prédit des données non ordonnées tandis que la régression prédit des données ordonnées.
La régression peut être évaluée en utilisant l'erreur quadratique moyenne. Au contraire, la classification est évaluée en mesurant la précision.


13.  
* #### SVM:

    SVM (Support Vector Machine ou Machine à vecteurs de support) : Les SVMs sont une famille d’algorithmes d‘apprentissage automatique qui permettent de résoudre des problèmes tant de classification que de régression ou de détection d’anomalie. Ils sont connus pour leurs solides garanties théoriques, leur grande flexibilité ainsi que leur simplicité d’utilisation même sans grande connaissance de data mining.

* #### KNN:

    L'algorithme des K plus proches voisins ou K-nearest neighbors (kNN) est un algorithme de Machine Learning qui appartient à la classe des algorithmes d'apprentissage supervisé simple et facile à mettre en œuvre qui peut être utilisé pour résoudre les problèmes de classification et de régression.

* #### Regrission lineare:

    La régression sert à trouver la relation d'une variable par rapport à une ou plusieurs autres.
    Dans l'apprentissage automatique, le but de la régression est d'estimer une valeur (numérique) de sortie à partir des valeurs d'un ensemble de caractéristiques en entrée. Par exemple, estimer le prix d'une maison en se basant sur sa surface, nombre des étages, son emplacement, etc. Donc, le problème revient à estimer une fonction de calcul en se basant sur des données d’entraînement. 