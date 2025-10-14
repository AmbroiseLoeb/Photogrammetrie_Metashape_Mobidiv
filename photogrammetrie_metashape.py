# importation des bibliotheques
# Test ME
import hauteurs_plantes
import subprocess
import os
import math
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image
import csv
from itertools import zip_longest
from tqdm import tqdm
from matplotlib import pyplot as plt


def sauvegarder_image(image, path_dossier, nom_fichier):
    """Enregistre une image ou une figure dans un dossier."""

    def figure_to_numpy(fig):
        """Convertit une figure Matplotlib en tableau numpy."""

        fig.set_dpi(800)  # Résolution de l'image
        fig.canvas.draw()
        img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(path_dossier):
        os.makedirs(path_dossier)

    # Construire le chemin complet du fichier
    chemin_complet = os.path.join(path_dossier, nom_fichier)

    # Vérifier que l'image est un tableau numpy
    if isinstance(image, np.ndarray):
        # Enregistrer l'image
        cv.imwrite(str(chemin_complet), image)  # Enregistrer l'image
    else:
        image_np = figure_to_numpy(image)
        cv.imwrite(str(chemin_complet), image_np)  # Enregistrer l'image


def traiter_dossier_racine(racine_path):
    """Traite chaque dossier session du dossier racine."""

    sessionlist = os.listdir(racine_path)  # Liste des dossiers sessions dans le dossier racine
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            traiter_dossier_session_ou_plot(os.path.join(racine_path, session))


def traiter_dossier_session_ou_plot(session_or_plot_path):
    """Traite par photogramétrie les images du dossier plot, ou de tout les dossier plot du dossier session."""

    # Executer la fonction boucle du script workflow_metashape dans Metashape pour calculer la carte de profondeur
    fonction = "boucle"
    subprocess.run(
        [r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'workflow_metashape.py', fonction] + [
            session_or_plot_path])

    # Dossier d'export des DEMs
    list_dems = os.listdir(session_or_plot_path + "/" + 'DEMs')
    for file in tqdm(sorted(list_dems)):
        print(file)

        # Récupérer la DEM et la transformer en matrice
        dem = Image.open(session_or_plot_path + "/" + 'DEMs' + "/" + file)
        dem_array = np.array(dem)
        dem_array[dem_array <= -3276] = np.nan
        dem_array = dem_array * 1000  # Conversion en mm

        # Filtre des points aberrants
        mat_filtree = hauteurs_plantes.filtre_points_aberrants(dem_array)

        # Calcul des hauteurs locales
        carte_hauteur, profondeur_sol = hauteurs_plantes.carte_hauteur_absolue(mat_filtree, n_zones)  # Carte de hauteur relative au sol
        liste_hauteurs, grille_h, figure_h = hauteurs_plantes.hauteur_par_zone(carte_hauteur, n_zones)
        # liste_hauteurs, figure_sommet = hauteurs_plantes.hauteur_par_sommet(carte_hauteur)  # Optionnel

        # Enregistrement des fichiers
        sauvegarder_image(figure_h, session_or_plot_path, f"grille_hauteur_{file}_{n_zones}z.jpg")  # Représentation graphique des hauteurs locales dans le bac
        # sauvegarder_image(figure_sommet, plot_path, f"sommets_hauteur_{os.path.basename(os.path.normpath(plot_path))}_{n_zones}z.jpg")  # Représentation graphique des hauteurs par sommet (Optionnel)

        # Export de la liste des hauteurs en csv
        with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([os.path.basename(os.path.normpath(session_or_plot_path)) if 'Session' in os.path.basename(os.path.normpath(session_or_plot_path)) else 'N/A'] + [file] + [str(h) for h in liste_hauteurs if not math.isinf(h)])


def main():
    """Fonction principale pour exécuter le pipeline."""
    global n_zones, csv_path, folder_name

    # Interface utilisateur pour sélectionner un dossier
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askdirectory(initialdir="/home/loeb/Documents", title="Sélectionnez un dossier")

    # Interface utilisateur pour sélectionner le nombre de zones
    n_zones = simpledialog.askinteger("Nombre de zones", "Veuillez choisir un nombre de zones : \n (correspond au maillage utilisé lors de la reconnaissance du sol et des maximas locaux)", initialvalue=100, minvalue=1)
    print('nombre de zones =', n_zones)

    if selected_path:
        folder_name = os.path.basename(selected_path)
        csv_path = os.path.join(selected_path, f"hauteurs_opencv_{folder_name}_{n_zones}z.csv")  # Emplacement du fichier csv
        # Créer le fichier csv temporaire
        with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([' '] + ['n° zone'] + [n for n in range(1, n_zones + 1)])

        #  Définir le type de dossier sélectionné  !!!!! Commenté ME oct 2024 temporairement
        # if "2024" in os.path.basename(os.path.dirname(os.path.normpath(selected_path))):  # changer 2024 par 'Session'
        #     print("Dossier plot sélectionné")
        #     traiter_dossier_session_ou_plot(selected_path)
        # elif "2024" in os.path.basename(selected_path):  # changer 2024 par 'Session'
        #     print("Dossier session sélectionné")
        #     traiter_dossier_session_ou_plot(selected_path)
        # else:
        #     print("Dossier racine sélectionné")
        #     traiter_dossier_racine(selected_path)
        traiter_dossier_session_ou_plot(selected_path)

    # Passer du fichier csv en ligne au csv final en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w',
                                                                                                       newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
    os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))  # Supprimer le fichier csv temporaire


if __name__ == "__main__":
    main()
