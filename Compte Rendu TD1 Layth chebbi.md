# Compte Rendu TD1 Classical Machine Learning and Deep Learning
# Layth Chebbi 3-SSIR-A

1. L'IA est l'imitation d'une réponse d'un expert dans un domaine. Le ML est de l'IA reposant sur des statistiques (Le ML adapte son algorithme mais a besoin d'un expert pour les données) Le DL est du ML utilisé avec des réseaux de neurones (Le DL extrait l'information pertinente des données et adapte son algorithme)

1. Un jeu de données est formé par N objets (observation) notés  Chaque objet est caractérisé par   descripteurs, ou attributs. Ainsi la taille de la base données est × .
Une technique d'apprentissage automatique est appelé à trouver un relation de dépendance
entre les objets en entrées et un jeu d'étiquettes en sortie (nommés également targets ou
labels)
1.   
    *  Explorer ou vérifier, représenter, décrire,        les    variables, leurs liaisons et positionner les observations de l'échantillon
    * Expliquer ou tester l'influence d'une variable ou facteur dans un modèle supposé connu a priori
    * Prévoir et sélectionner un meilleur ensemble de prédicteurs comme par exemple dans la recherche de bio-marqueurs
1. 
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
1.  
    L'apprentissage supervisé est réalisé sur un jeu de données étiquetés. C'est à dire que les données d'entraînement fournis au départ à l'algorithme comportent les solutions désirées, appelées étiquettes (en anglais, labels). L'objectif revient donc à classer correctement un nouvel exemple.
1. 
    Toutes les données sont étiquetées et les algorithmes apprennent à prédire le résultat des données d'entrée. Non supervisé: toutes les données ne sont pas étiquetées et les algorithmes apprennent la structure inhérente à partir des données en entrée.
1. 
1. 
    *  La validation croisée permet de tirer plusieurs   ensembles de validation d'une même base de données et ainsi d'obtenir une estimation plus robuste, avec biais et variance, de la performance de validation du modèle.

    * La validation croisée k-fold signifie que l’ensemble de données se divise en un nombre K. Elle divise l’ensemble de données au point où l’ensemble de test utilise chaque pli. Comprenons le concept à l’aide de la validation croisée à 5 volets ou K+5. Dans ce scénario, la méthode divise l’ensemble de données en cinq volets. Le modèle utilise le premier pli dans la première itération pour tester le modèle. Il utilise les autres ensembles de données pour former le modèle. Le deuxième pli aide à tester l’ensemble de données et les autres soutiennent le processus de formation. Le même processus se répète jusqu’à ce que l’ensemble de test utilise chaque pli des cinq plis.

1. Tableau de confusion : 
![](data:image/png;base64,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)

1.  * ### Precision:

        TP / ( TP + FP )
    * ### Recall:

        TP / ( TP + FN )
    * ### Accuracy:

        ( TP + TN ) / N
1.  * ROC: 
![](https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg)

    * ### UAC:
    
    L’AUC aide à comparer les différents classificateurs. Vous pouvez résumer les performances de chaque classificateur en une seule mesure. L’approche de base pour trouver la CUA est de calculer l’AUROC. Elle est similaire à la probabilité que l’instance négative aléatoire soit inférieure à l’instance positive. Si un classificateur a une CUA inférieure à celle d’un autre classificateur, cela signifie normalement que le score de la CUA élevée n’est pas bon. Cependant, la SSC fonctionne bien dans le cadre de la mesure générale de la précision prédictive.

1. Le processus de classification modélise une fonction par laquelle les données sont prédites dans des étiquettes de classe discrètes. D'autre part, la régression est le processus de création d'un modèle qui prédit une quantité continue.
Les algorithmes de classification impliquent un arbre de décision, une régression logistique, etc. En revanche, un arbre de régression (par exemple, une forêt aléatoire) et une régression linéaire sont des exemples d'algorithmes de régression.
La classification prédit des données non ordonnées tandis que la régression prédit des données ordonnées.
La régression peut être évaluée en utilisant l'erreur quadratique moyenne. Au contraire, la classification est évaluée en mesurant la précision.
1.  * SVM:

    SVM (Support Vector Machine ou Machine à vecteurs de support) : Les SVMs sont une famille d’algorithmes d‘apprentissage automatique qui permettent de résoudre des problèmes tant de classification que de régression ou de détection d’anomalie. Ils sont connus pour leurs solides garanties théoriques, leur grande flexibilité ainsi que leur simplicité d’utilisation même sans grande connaissance de data mining.
    * KNN:

    L'algorithme des K plus proches voisins ou K-nearest neighbors (kNN) est un algorithme de Machine Learning qui appartient à la classe des algorithmes d'apprentissage supervisé simple et facile à mettre en œuvre qui peut être utilisé pour résoudre les problèmes de classification et de régression.
    * Regrission lineare:

    La régression sert à trouver la relation d'une variable par rapport à une ou plusieurs autres.
    Dans l'apprentissage automatique, le but de la régression est d'estimer une valeur (numérique) de sortie à partir des valeurs d'un ensemble de caractéristiques en entrée. Par exemple, estimer le prix d'une maison en se basant sur sa surface, nombre des étages, son emplacement, etc. Donc, le problème revient à estimer une fonction de calcul en se basant sur des données d’entraînement. 