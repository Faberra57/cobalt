# Traitement d’images microscopiques par intelligence artificielle

Khalil ABDELHEDI | Corentin BOUTAULT | Iyed DAMMAK | Thomas FAVRE | Skander MAHJOUB | Fabien SENEQUIER | Noah SCHOUAME

<div style="text-align: center;">
  <img src="https://www.imt-atlantique.fr/sites/default/files/ecole/logos/imtatlantique/imtatlantique-cmjn-reserve.png" alt="IMT Atlantique Logo" width="25%">
  <img src="https://media.licdn.com/dms/image/v2/C4E0BAQECWVPdJief2Q/company-logo_200_200/company-logo_200_200/0/1668505576220/cobalt_contraception_logo?e=2147483647&v=beta&t=TVbzQ0YoOCeWGU_jZFDfCeVXCAjEcox2pfQ1akBoFVo" alt="Company Logo" width="25%">
</div>

## Sommaire
- [Contexte](#contexte)
- [Base de données](#dataset)
- [Algorithmes](#algorithmes)
- [Résultats](#résultats)
- [Utilisation](#utilisation)

## Contexte
Dans le cadre de notre projet commande entreprise en collaboration avec [Cobalt Contraception](https://www.cobalt-contraception.com/), nous avons développé un algorithme d’intelligence artificielle capable de déterminer le nombre de cellules présentes sur une image microscopique. Cet algorithme s’appuie sur une base de données en ligne contenant des images de cellules. Cet algorithme pourra ensuite être intégré dans une carte Raspberry.

## Base de données
Nous avons travaillé sur la base de données [Synthetic Cell Images and Masks](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1) disponible sur Kaggle. Cette base de données est composé de 20000 images de cellules microscopiques simulées particulièrement utiles pour la segmentation.

<div style="text-align: center;">
  <img src="https://data.broadinstitute.org/bbbc/BBBC005/synthetic2_in_focus_image.png" alt="Exemple d'image du dataset" />
  <p>Exemple d'image du dataset.</p>
</div>

## Algorithmes
Vous trouverez 3 dossiers :
* [CNN](https://github.com/Faberra57/cobalt/tree/main/CNN) : 
* [U-Net](https://github.com/Faberra57/cobalt/tree/main/Inet) :
* [Capsule Networks](https://github.com/Faberra57/cobalt/tree/main/CapsuleNetworks) :

## Résultats

## Utilisation

Pour utiliser ce projet, suivez les étapes suivantes :

* **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Faberra57/cobalt.git
   cd cobalt

### <u>Utiliser un modèle déjà entraîné :</u>

* **Télécharger le dataset :**<br>
Rendez vous sur la page kaggle de [Synthetic Cell Images and Masks](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1) et cliquez sur **Download** afin de télécharger le dataset.

* **Implémenter sur la carte Raspberry**<br>
[Lien vers la documentation Raspberry Pi](https://docs.google.com/document/d/1x48W1OlW6UofIinBcvy3fAUehOSoxmO2rE9DZH93tJM/edit?tab=t.0)<br>
Utilisez le fichier *compteur de cellule.py* afin d'appliquer le modèle. Pour changer le modèle utilisé, il vous suffir de changer cette ligne :
```python
model_path = 'votre_modèle.onnx'
```
Assurez-vous que *votre_modèle.onnx* se trouve dans le même dossier que *compteur de cellule.py*.

### <u>Entraîner un modèle sur son propre dataset :</u>
