# Pipeline d’analyse d’image par photogrammétrie

## Description
Ce projet à pour objetctif de mesurer les traits architecturaux de couverts végétaux dans un contexte d'étude des mélanges variétaux de blé.
Différents génotypes sont cultivés dans des bacs séparés, en culture pure ou en mélange.
Plusieurs dispositifs d'acqisition d'image, manuel ou automatique, permettent une collecte de données régulière à différents stades de croissance.

Le projet comporte deux méthodes, s'adaptant chacune à un dispositif d'aquisition.
Ici, on se propose de traiter les images par photogrammétrie à l'aide notamment du logiciel Agisoft Metashape.


## Installation

1. **Clonez le dépôt :**
    ```bash
    git clone https://github.com/aloeb-gh/Photogrammetrie_Metashape_Mobidiv.git
    ```

2. **Accédez au répertoire du projet :**
    ```bash
    cd Photogrammetrie_Metashape_Mobidiv
    ```

3. **Créez et activez un environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```



## Prérequis

- Télecharger *Agisoft Metashape Professional Edition*, version 2.0.4 minimum :

https://www.agisoft.com/downloads/installer/

- Activer le logiciel avec votre clé de licence.

- Vérifier sur cette ligne l'emplacement correct du logiciel (mettre à jour l'emplacement le cas échéant) :

https://github.com/aloeb-gh/Photogrammetrie_Metashape_Mobidiv/blob/fad16ba195dc6abc5e7ed964d93d30870f315046/photogrammetrie_metashape.py#L61



## Utilisation
**Lancer le pipeline :**

```bash
python3 Photogrammetrie_Metashape_Mobidiv.py 
```

**Selectionner votre dossier :**

Dans l'interface qui apparait, sélectionner au choix : 
- dossier *plot* contenant les images à traiter.
- dossier *Session* contenant plusieurs dossiers *plot*.
- dossier *racine* contenant plusieurs dossiers *Session*.

**Choisir le nombre de zones :**

Pour calculer localement les hauteur des plantes, la région du bac est découpée en zones de même taille.
Par défault, le nombre de zone est fixé à 100.
Augmenter ce nombre permet une meilleure résolution et d'avantage de données de hauteur.
Cependant, un nombre de zone trop important peut faire apparaitre des valeurs aberrantes.
Pour une répartition égale des zones, il est préférable de choisir une puissance (81, 100, 169, 225 etc.).


**Outputs :**

- fichier .csv comprenant la hauteur de chaque zone (dans le dossier sélectionné).
- fichier *Projet Metashape* .psx comprenant les représentations 3D de chaque plot (dans le dossier sélectionné).
- représentation graphique des hauteur de chaque zone (dans le dossier *plot*).




## Ressources utiles

Agisoft Metashape User Manual : 
https://www.agisoft.com/pdf/metashape-pro_2_1_en.pdf

Python API Reference :
https://www.agisoft.com/pdf/metashape_python_api_2_1_2.pdf

Suivi du projet : 
https://aloeb.notion.site/Suivi-du-projet-482f379e883b4974b1b2b95aec96181d

Projet Mobidiv :
https://mobidiv.hub.inrae.fr/
