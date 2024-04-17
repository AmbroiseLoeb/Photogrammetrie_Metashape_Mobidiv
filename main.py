import subprocess
import os
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from PIL import Image
import csv
from itertools import zip_longest


def mask_image(image):
    """ definir les contour du bac sur la photo, detecter et supprimer le capteur si besoin """

    # convertir en rgba (a = transparence)
    img_sans_capteur = cv.cvtColor(image, cv.COLOR_RGB2RGBA)

    # definir la colone et la ligne centrale
    height, width, color = img_sans_capteur.shape
    centre_colonne = width // 2
    centre_ligne = height // 2

    # masque des pixels verts
    hsv_img = cv.cvtColor(img_sans_capteur, cv.COLOR_BGR2HSV)
    lower_green = np.array([10, 20, 20])  # Valeurs min de teinte, saturation et valeur pour la couleur verte
    upper_green = np.array([100, 255, 255])  # Valeurs max de teinte, saturation et valeur pour la couleur verte
    mask_green = cv.inRange(hsv_img, lower_green, upper_green)
    img_without_green = cv.bitwise_and(img_sans_capteur, img_sans_capteur, mask=~mask_green)  # Appliquer le masque
    img_gray = cv.cvtColor(img_without_green, cv.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

    # seuil de gris
    seuil_gris = max(np.mean(img_gray), 20)
    _, thresholded_img = cv.threshold(img_gray, seuil_gris, 255, cv.THRESH_BINARY)
    # plt.figure() and plt.imshow(thresholded_img)

    # Étiqueter les objets, calculer leur coordonnées et leur taille
    labels, nb_labels = ndi.label(thresholded_img)
    coordinates = ndi.center_of_mass(thresholded_img, labels, range(nb_labels + 1))
    sizes = ndi.sum(thresholded_img, labels, range(nb_labels + 1))
    # Supprimer les objets inférieurs à 300 pixels et le capteur
    seuil2 = 0

    filtered_image = np.zeros_like(thresholded_img)
    for label in range(1, nb_labels + 1):
        if 1200 <= coordinates[label][0] <= 2500 and 1500 <= coordinates[label][1] <= 3200:  # capteur
            if 20000 * 255 <= sizes[label] <= 80000 * 255:
                img_sans_capteur[labels == label] = (0, 0, 0, 0)
                seuil2 = 100000  # augmentation du seuil en cas de présence du capteur
                filtered_image[labels == label] = 255
            elif sizes[label] >= 5000 * 255:
                filtered_image[labels == label] = 255
                seuil2 = 100000
        elif sizes[label] >= 300 * 255:
            filtered_image[labels == label] = 255
    # plt.figure() and plt.imshow(filtered_image)

    # filtered_image = thresholded_img
    # paramettres pour recherche des bords de bac
    largueur_min = 1000
    longueur_min = 1200
    nouvelle_longueur = 0
    nouvelle_largeur = 0
    seuil_bordure = 20000
    nouvelle_largeur_haut = centre_ligne
    nouvelle_largeur_bas = centre_ligne
    nouvelle_longueur_gauche = centre_colonne
    nouvelle_longueur_droite = centre_colonne
    n = 0

    # Recherche des bords de bac
    while nouvelle_longueur <= longueur_min or nouvelle_largeur <= largueur_min and n <= 10:
        if nouvelle_largeur <= largueur_min:
            colonne = centre_colonne
            for colonne in range(centre_colonne, width):
                if np.sum(filtered_image[1200:2500, colonne]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_longueur_droite = colonne

            for colonne in range(centre_colonne, -1, -1):
                if np.sum(filtered_image[1200:2500, colonne]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_longueur_gauche = colonne
            nouvelle_largeur = nouvelle_longueur_droite - nouvelle_longueur_gauche

        if nouvelle_longueur <= longueur_min:
            ligne = centre_ligne
            for ligne in range(centre_ligne, height):
                if np.sum(filtered_image[ligne, min(1800, nouvelle_longueur_gauche):max(3200, nouvelle_longueur_droite)]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_bas = ligne

            for ligne in range(centre_ligne - 100, -1, -1):
                if np.sum(filtered_image[ligne, min(1800, nouvelle_longueur_gauche):max(3200, nouvelle_longueur_droite)]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_haut = ligne
            nouvelle_longueur = nouvelle_largeur_bas - nouvelle_largeur_haut

        filtered_image[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0
        n = n+1

    nouvelle_largeur_haut = max(0, nouvelle_largeur_haut - 50)
    nouvelle_largeur_bas = min(height, nouvelle_largeur_bas + 50)
    nouvelle_longueur_gauche = max(0, nouvelle_longueur_gauche - 50)
    nouvelle_longueur_droite = min(width, nouvelle_longueur_droite + 50)
    masked_image = np.zeros_like(img_sans_capteur)
    masked_image[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 255

    return (nouvelle_largeur_haut, nouvelle_largeur_bas, nouvelle_longueur_gauche, nouvelle_longueur_droite,
            masked_image)


def filtre_points_aberrants(matrice):
    """ Supprime les points aberrants jusqu'à ce que la variation de la moyenne soit inférieure à seuil_stable_moy % """
    matrice_filtree = matrice.copy()  # Copie de la matrice pour éviter les modifications inattendues
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
    matrice_filtree[matrice_filtree <= -1900] = np.nan
    seuil_stable_moy = 0.0001

    while True:
        # Calculer la moyenne actuelle
        moyenne_actuelle = np.nanmean(matrice_filtree)

        # Trouver un seuil pour filtrer les points aberrants
        ecart_type = np.nanstd(matrice_filtree)
        limite_inf = moyenne_actuelle - 2.5 * ecart_type
        limite_sup = moyenne_actuelle + 5 * ecart_type

        # Remplacer les points aberrants par NaN
        nouvelle_matrice_filtree = matrice_filtree.copy()
        nouvelle_matrice_filtree[(matrice_filtree < limite_inf) | (matrice_filtree > limite_sup)] = np.nan

        # Calculer la nouvelle moyenne
        nouvelle_moyenne = np.nanmean(nouvelle_matrice_filtree)

        # Si la variation de la moyenne est inférieure au seuil, arrêter
        if abs(abs(nouvelle_moyenne - moyenne_actuelle) / moyenne_actuelle) < seuil_stable_moy:
            break

        # Mettre à jour la matrice filtrée
        matrice_filtree = nouvelle_matrice_filtree

    return matrice_filtree


def hauteur_locale(matrice, nombre_zones):
    # Taille des zones représentant n% de la matrice
    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (int(matrice.shape[0] * coeff), int(matrice.shape[1] * coeff))
    # Calculer le nombre total de zones dans la matrice

    # Initialiser les listes pour stocker les résultats
    max_locals = []
    sol_locaux = []
    hauteur = []
    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = np.median(np.sort(mat_sans_nan.flatten())[:int(mat_sans_nan.size * 0.03)])

    # Parcourir chaque zone
    for i in range(0, matrice.shape[0], zone_size[0]):
        for j in range(0, matrice.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = matrice[i:i + zone_size[0], j:j + zone_size[1]]

            # Calculer max_local et sol_local pour la zone
            zone_sans_nan = zone[~np.isnan(zone)]
            max_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.01)])
            sol_local = np.median(np.sort(zone_sans_nan.flatten())[:int(zone_sans_nan.size * 0.03)])

            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteur.append(np.nan)
            else:
                # Ajouter les résultats à la liste
                max_locals.append(max_local)
                sol_locaux.append(sol_local)
                if sol_bac - 20 <= sol_local <= sol_bac + 20:
                    hauteur.append(abs(sol_local - max_local))
                else:
                    hauteur.append(abs(sol_bac - max_local))

    # Convertir les listes en tableaux numpy
    max_locals = np.array(max_locals)
    sol_locaux = np.array(sol_locaux)
    hauteur_a = np.array(hauteur)
    hauteur = hauteur_a[~np.isnan(hauteur_a)]

    mat_hauteur = matrice.copy()  # Copie de mat_filtree pour ne pas modifier l'original
    index = 0
    for i in range(0, mat_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_hauteur.shape[1], zone_size[1]):
            # Assigner la valeur de hauteur correspondante à chaque point de la zone
            mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]] = hauteur_a[index]
            index += 1

    return hauteur, mat_hauteur


# lancer metashape depuis python
def main():

    path_annee = r"C:\Users\U108-N806\Desktop\Comparaison mesures"
    # path_annee = '/home/loeb/Documents/Comparaison_mesures'
    sessionlist = os.listdir(path_annee)
    for session in sorted(sessionlist):
        if session.find("Session") == 0:
            print(session)
            plotlist = os.listdir(path_annee + "/" + session)
            for plot in sorted(plotlist):
                if plot.find("uplot_7_1") == 0:
                    print(plot)
                    imglist = os.listdir(path_annee + "/" + session + "/" + plot)
                    for file in imglist:
                        if file.endswith("RGB.jpg") and "camera_3" not in file:
                            print(file)
                            # Creer et exporter le masque associe a la photo
                            photo = cv.imread(path_annee + "/" + session + "/" + plot + "/" + file)
                            mask_photo = mask_image(photo)[-1]
                            save_path = path_annee + "/" + session + "/" + plot + "/" + os.path.basename(file).replace("RGB", "MASK")
                            cv.imwrite(save_path, mask_photo)

    # Executer le script correspondant dans Metashape
    fonction = "boucle"
    subprocess.run([r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'test_script_metashape.py', fonction] + [path_annee])

    # PATH
    n_zones = 100
    print('nombre de zones =', n_zones)
    csv_path = path_annee + "/" + "hauteurs_metashape" + str(n_zones) + ".csv"
    sessionlist = os.listdir(path_annee)
    for session in sorted(sessionlist):
        if session.find("Session") == 0:
            print(session)
            list_dems = os.listdir(path_annee + "/" + session + "/" + 'DEMs3')
            for file in sorted(list_dems):
                print(file)
                # Récupérer la DEM et la transformer en matrice
                dem = Image.open(path_annee + "/" + session + "/" + 'DEMs3' + "/" + file)
                dem_array = np.array(dem)
                dem_array[dem_array <= -3276] = np.nan
                dem_array = dem_array * 1000
                # plt.figure() and plt.imshow(dem_array)

                # Filtre des points aberrants
                mat_filtree = filtre_points_aberrants(dem_array)
                # plt.figure() and plt.imshow(mat_filtree)

                # Calcul des hauteurs locales
                liste_hauteurs, z_mat = hauteur_locale(mat_filtree, n_zones)
                print(liste_hauteurs)
                # plt.figure() and plt.imshow(z_mat, cmap='jet', vmin=0, vmax=1000)

                # Export des hauteurs locales en csv
                with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([session] + [file] + [str(h) for h in liste_hauteurs])
            # csv en ligne -> csv en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w', newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
        os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))


if __name__ == "__main__":
    main()
