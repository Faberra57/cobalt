# Cobalt

Ce projet, sous la direction de **Cobalt**, a pour objectif de compter le nombre de cellules dans un échantillon en utilisant des techniques de **deep learning**.

---

## Organisation du projet

### Dossier **CNN** :

- L'implémentation finale que nous avons retenue du CNN est nommée **CNN_finale.ipynb**.
  - Il est nécessaire d'ajuster le `batch size` si vous souhaitez lancer le training en fonction de la RAM disponible sur votre machine.

### Dossier **Capsule Network** :

Le dossier CapsuleNetwork contient un fichier qui se nomme **capsNet.ipyng** et vise à créer une architecture capsule network et à entrainer un model suivant cette architecture afin de compter le nombre de cellules sur le dataset.

### Dossier **U-net** :

- Le but de cet algorithme est de créer un **mask**, c'est-à-dire une image dans laquelle seules les cellules d'intérêt seront conservées et les débris seront éliminés. Cela permet de faciliter le comptage des cellules par la suite, à l'aide d'un autre algorithme.
  - L'implémentation actuelle dans le dossier n'est pas finalisée, car elle ne prend pas de véritables images avec des débris en entrée.

### Dossier **Tool_kit** :

Le dossier **Tool_kit** contient des fichiers Python que nous utilisons dans nos codes.

### Dossier **Model** :

Le dossier **Model** rassemble les modèles que nous avons entraînés.

### Dossier **Arduino** :
* **Implémenter sur la carte Raspberry**<br>
[Lien vers la documentation Raspberry Pi](https://docs.google.com/document/d/1x48W1OlW6UofIinBcvy3fAUehOSoxmO2rE9DZH93tJM/edit?tab=t.0)<br>
Utilisez le fichier *compteur de cellule.py* du dossier *Arduino* afin d'appliquer le modèle. Pour changer le modèle utilisé, il vous suffir de changer cette ligne :
```python
model_path = 'model/votre_modèle.onnx'
```


Dans le dossier *Arduino*, on trouve aussi le fichier *resize_images.py* pour changer la taille des images au format des inputs des modèles onnx afin que le dossier d'images soit moins lourds sur la carte raspberry pi.


## Résultats
Pour tester les performances des modèles ONNX, utilisez le programme *test_performance_modele_onnx.py*. Il est possible de choisir le nombre d'image TIF pour mesurer les performances avec la variable "nb_images". On peut aussi choisir le dossier où les images sont à analyser : 
```python
image_folder = '...'
```

Enfin pour choisir le modèle onnx : 
```python
model_path = '...'
```

Il ne reste plus qu'à executer ou ouvrir le fichier *Résultat modèles onnx.pdf* pour avoir les performances du modèle.


## Informations importantes

Nous avons utilisé le dataset suivant : [https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1)

Veuillez le télécharger et le placer dans un dossier nommé **dataset**. Vérifiez que l'arborescence soit bien la suivante :
'dataset/archive/BBBC005_v1_images/BBBC005_v1_images'

Cela permettra d'accéder correctement aux images pour le training.
