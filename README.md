# Cobalt

Ce projet, sous la direction de **Cobalt**, a pour objectif de compter le nombre de cellules dans un échantillon en utilisant des techniques de **deep learning**.

---

## Organisation du projet

### Dossier **CNN** :

- L'implémentation finale que nous avons retenue du CNN est nommée **CNN_finale.ipynb**.
  - Il est nécessaire d'ajuster le `batch size` si vous souhaitez lancer le training en fonction de la RAM disponible sur votre machine.

### Dossier **Capsule Network** :

(Description à ajouter si nécessaire.)

### Dossier **U-net** :

- Le but de cet algorithme est de créer un **mask**, c'est-à-dire une image dans laquelle seules les cellules d'intérêt seront conservées et les débris seront éliminés. Cela permet de faciliter le comptage des cellules par la suite, à l'aide d'un autre algorithme.
  - L'implémentation actuelle dans le dossier n'est pas finalisée, car elle ne prend pas de véritables images avec des débris en entrée.

### Dossier **Tool_kit** :

Le dossier **Tool_kit** contient des fichiers Python que nous utilisons dans nos codes.

### Dossier **Model** :

Le dossier **Model** rassemble les modèles que nous avons entraînés.

---

## Informations importantes

Nous avons utilisé le dataset suivant : [https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1)

Veuillez le télécharger et le placer dans un dossier nommé **dataset**. Vérifiez que l'arborescence soit bien la suivante :
'dataset/archive/BBBC005_v1_images/BBBC005_v1_images'

Cela permettra d'accéder correctement aux images pour le training.
