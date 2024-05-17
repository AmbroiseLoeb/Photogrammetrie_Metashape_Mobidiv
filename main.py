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
from tqdm import tqdm


def mask_image(image, seuil_small_obj=300):
    """ definir les contour du bac sur la photo """

    def label_objects(image):
        # Étiqueter les objets, calculer leur coordonnées et leur taille
        labels_, nb_labels_ = ndi.label(image)
        coordinates_ = np.array(ndi.center_of_mass(image, labels_, range(1, nb_labels_ + 1)))
        sizes_ = ndi.sum(image, labels_, range(nb_labels_ + 1))
        return labels_, nb_labels_, coordinates_, sizes_

    height, width = image.shape[:2]

    # masque des pixels verts
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_green = np.array([10, 20, 20])  # Valeurs min de teinte, saturation et valeur pour la couleur verte
    upper_green = np.array([100, 255, 255])  # Valeurs max de teinte, saturation et valeur pour la couleur verte
    mask_green = cv.inRange(hsv_img, lower_green, upper_green)
    img_without_green = cv.bitwise_and(image, image, mask=~mask_green)  # Appliquer le masque
    # plt.figure() and plt.imshow(img_without_green)

    # masque des pixels non gris, seuil de gris
    img_gray = cv.cvtColor(img_without_green, cv.COLOR_BGR2GRAY)
    seuil_gris = max(np.mean(img_gray), 20)
    _, image_filtree = cv.threshold(img_gray, seuil_gris, 255, cv.THRESH_BINARY)
    # plt.figure() and plt.imshow(image_filtree)

    # Supprimer les petits objets
    labels, nb_labels, coordinates, sizes = label_objects(image_filtree)
    image_filtree2 = np.zeros_like(image_filtree)
    for label in (range(1, nb_labels + 1)):
        if sizes[label] >= seuil_small_obj * 255:
            image_filtree2[labels == label] = 255
    # plt.figure() and plt.imshow(image_filtree2)

    # Supprimer le capteur (même si en plusieurs morceaux)
    labels, nb_labels, coordinates, sizes = label_objects(image_filtree2)
    image_filtree3 = image_filtree2.copy()
    for i in (range(len(coordinates))):
        if 1200 <= coordinates[i][0] <= 2100 and 1800 <= coordinates[i][1] <= 2900:
            for j in range(i + 1, len(coordinates)):
                if (abs(coordinates[i][0] - coordinates[j][0]) <= 300) and (
                        abs(coordinates[i][1] - coordinates[j][1]) <= 300):
                    # print(coordinates[i])
                    # print(coordinates[j])
                    # print(sizes[i])
                    # print(sizes[j])
                    if 2000 * 255 <= sizes[i] + sizes[j] <= 200000 * 255:  # capteur
                        # print(sizes[i] + sizes[j])
                        image_filtree3[labels == i + 1] = 0
                        image_filtree3[labels == j + 1] = 0
                        image_filtree3[int(coordinates[i][0] - 300): int(coordinates[i][0] + 300), int(coordinates[i][1] - 300): int(coordinates[i][1] + 300)] = 0
                        image_filtree3[int(coordinates[j][0] - 300): int(coordinates[j][0] + 300), int(coordinates[j][1] - 300): int(coordinates[j][1] + 300)] = 0
                elif 2000 * 255 <= sizes[i] <= 200000 * 255:  # capteur
                    # print(sizes[i])
                    image_filtree3[labels == i + 1] = 0

    # Tracer des lignes entre les objets ayant les mêmes coordonnées horizontales
    image_with_lines = image_filtree3.copy()
    labels, nb_labels, coordinates, sizes = label_objects(image_filtree3)
    for i in (range(len(coordinates))):
        for j in range(i + 1, len(coordinates)):
            if abs(coordinates[i][0] - coordinates[j][0]) <= 100 and abs(
                    coordinates[i][1] - coordinates[j][1]) <= 1000:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                        (int(coordinates[j][1]), int(coordinates[j][0])), (255, 255, 255), 30)
        # Tracer une ligne si l'objet est proche du bord gauche ou droit de l'image
        '''
        if coordinates[i][1] <= 1500:
            cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])), (0, int(coordinates[i][0])),
                    (255, 255, 255), 15)
        elif coordinates[i][1] >= width - 1500:
            cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                    (image_with_lines.shape[1] - 1, int(coordinates[i][0])), (255, 255, 255), 15)
        '''

    # Tracer des lignes entre les objets ayant les mêmes coordonnées verticales
    for i in (range(len(coordinates))):
        for j in range(i + 1, len(coordinates)):
            if abs(coordinates[i][1] - coordinates[j][1]) <= 100 and abs(
                    coordinates[i][0] - coordinates[j][0]) <= 1500:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                        (int(coordinates[j][1]), int(coordinates[j][0])), (255, 255, 255), 30)
                # Tracer une ligne si l'objet est proche du bord supérieur ou inférieur de l'image
        '''
            if coordinates[i][0] <= 1500:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])), (int(coordinates[i][1]), 0),
                        (255, 255, 255), 15)
            elif coordinates[i][0] >= image_with_lines.shape[0] - 1500:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                        (int(coordinates[i][1]), image_with_lines.shape[0] - 1), (255, 255, 255), 15)
        '''
    # plt.figure() and plt.imshow(image_with_lines)

    # paramettres pour la recherche des bords de bac
    largueur_min = 1600
    longueur_min = 1800
    nouvelle_longueur = 0
    nouvelle_largeur = 0
    seuil_bordure = 300 * 255
    centre_ligne = height // 2
    centre_colonne = width // 2
    nouvelle_largeur_haut = centre_ligne
    nouvelle_largeur_bas = centre_ligne
    nouvelle_longueur_gauche = centre_colonne
    nouvelle_longueur_droite = centre_colonne

    # Recherche des bords du bac
    image_with_lines = image_with_lines + image_filtree2
    while (nouvelle_longueur <= longueur_min) or (nouvelle_largeur <= largueur_min):
        if nouvelle_longueur <= longueur_min:
            ligne = centre_ligne
            for ligne in range(centre_ligne, height):
                if np.sum(image_with_lines[ligne, 1800:2900]) > seuil_bordure:
                    # print(np.sum(image_with_lines[ligne, 1800:2900]))
                    break
            nouvelle_largeur_bas = ligne

            for ligne in range(centre_ligne, 0, -1):
                if np.sum(image_with_lines[ligne, 1800:2900]) > seuil_bordure:
                    # print(np.sum(image_with_lines[ligne, 1800:2900]))
                    break
            nouvelle_largeur_haut = ligne
            nouvelle_longueur = nouvelle_largeur_bas - nouvelle_largeur_haut

        if nouvelle_largeur <= largueur_min:
            colonne = centre_colonne
            for colonne in range(centre_colonne, width):
                if np.sum(image_with_lines[1200:2100, colonne]) > seuil_bordure:
                    # print(np.sum(image_with_lines[1200:2100, colonne]))
                    break
            nouvelle_longueur_droite = colonne

            for colonne in range(centre_colonne, 0, -1):
                if np.sum(image_with_lines[1200:2100, colonne]) > seuil_bordure:
                    # print(np.sum(image_with_lines[1200:2100, colonne]))
                    break
            nouvelle_longueur_gauche = colonne
            nouvelle_largeur = nouvelle_longueur_droite - nouvelle_longueur_gauche

        # image_with_lines[nouvelle_largeur_haut:nouvelle_largeur_bas, 1800:2900] = 0
        # image_with_lines[1200:2100, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0
        image_with_lines[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0

    # plt.figure() and plt.imshow(image_with_lines)
    masked_image = np.zeros_like(image)
    masked_image[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 255

    return (nouvelle_largeur_haut, nouvelle_largeur_bas, nouvelle_longueur_gauche, nouvelle_longueur_droite,
            masked_image)


def filtre_points_aberrants(matrice):
    """ Supprime les points aberrants jusqu'à ce que la variation de la moyenne soit inférieure à seuil_stable_moy % """
    matrice_filtree = matrice.copy()  # Copie de la matrice pour éviter les modifications inattendues
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
    #matrice_filtree[matrice_filtree <= -1900] = np.nan
    seuil_stable_moy = 0.0001

    while True:
        # Calculer la moyenne actuelle
        moyenne_actuelle = np.nanmean(matrice_filtree)

        # Trouver un seuil pour filtrer les points aberrants
        ecart_type = np.nanstd(matrice_filtree)
        limite_inf = moyenne_actuelle - 5 * ecart_type
        limite_sup = moyenne_actuelle + 4 * ecart_type

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

        # Re-filtrer les points les plus hauts
        matrice_filtree[(matrice_filtree < np.median(np.sort(matrice_filtree.flatten())[:int(matrice_filtree.size * 0.0005)]))] = np.nan

    return matrice_filtree


def hauteur_locale(matrice, nombre_zones):
    # Taille des zones représentant n% de la matrice
    coeff = 1/math.sqrt(nombre_zones)
    zone_size = (int(matrice.shape[0] * coeff), int(matrice.shape[1] * coeff))

    # Initialiser les listes pour stocker les résultats
    max_locals = []
    sol_locaux = []
    hauteur = []
    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = - np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.03)])
    max_glob = abs(sol_bac + np.median(np.sort(mat_sans_nan.flatten())[:int(mat_sans_nan.size * 0.02)]))
    mat_hauteur = -1 * matrice.copy()

    # Parcourir chaque zone
    for i in range(0, matrice.shape[0], zone_size[0]):
        for j in range(0, matrice.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]]

            # Calculer max_local et sol_local pour la zone
            zone_sans_nan = zone[~np.isnan(zone)]
            sol_local = np.median(np.sort(zone_sans_nan.flatten())[:int(zone_sans_nan.size * 0.03)])
            sol_locaux.append(sol_local)

            # Ramener le sol à zero
            if sol_bac - 100 <= sol_local <= sol_bac + 50:
                zone -= sol_local
                #print('new_sol')
            else:
                zone -= sol_bac
                #print('sol_bac')

            zone = mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]]
            zone_sans_nan = zone[~np.isnan(zone)]
            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteur.append(np.nan)
            else:
                mean_local = np.mean(zone_sans_nan.flatten())
                max_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.03)])
                max_locals.append(max_local)
                if mean_local > max_glob/4 and max_local > max_glob/3:
                    # Ajouter les résultats à la liste
                    hauteur.append(max_local)
                else:
                    hauteur.append(np.nan)
    # plt.figure() and plt.imshow(mat_hauteur)

    # Convertir les listes en tableaux numpy
    max_locals = np.array(max_locals)
    sol_locaux = np.array(sol_locaux)
    hauteur_a = np.array(hauteur)
    hauteur = hauteur_a[~np.isnan(hauteur_a)]

    mat_zones_hauteur = np.zeros_like(matrice)
    index = 0
    for i in range(0, mat_zones_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_zones_hauteur.shape[1], zone_size[1]):
            # Assigner la valeur de hauteur correspondante à chaque point de la zone
            mat_zones_hauteur[i:i + zone_size[0], j:j + zone_size[1]] = hauteur_a[index]
            index += 1
    # plt.figure() and plt.imshow(mat_zones_hauteur)
    return hauteur, mat_zones_hauteur


# lancer metashape depuis python
def main():

    path_annee = r"C:\Users\U108-N806\Desktop\Comparaison mesures"
    # path_annee = '/home/loeb/Documents/Comparaison_mesures'
    sessionlist = os.listdir(path_annee)
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            plotlist = os.listdir(path_annee + "/" + session)
            for plot in sorted(plotlist):
                if plot.find("uplot") == 0:
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
    subprocess.run([r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'workflow_metashape.py', fonction] + [path_annee])

    # PATH
    n_zones = 100
    print('nombre de zones =', n_zones)
    csv_path = path_annee + "/" + "hauteurs_metashape" + str(n_zones) + ".csv"
    sessionlist = os.listdir(path_annee)
    for session in sorted(sessionlist):
        if session.find("Session") == 0:
            print(session)
            list_dems = os.listdir(path_annee + "/" + session + "/" + 'DEMs')
            for file in tqdm(sorted(list_dems)):
                print(file)
                # Récupérer la DEM et la transformer en matrice
                dem = Image.open(path_annee + "/" + session + "/" + 'DEMs' + "/" + file)
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

                # Stats hauteurs locales
                hauteur_moyenne = np.mean(liste_hauteurs)
                hauteur_mediane = np.median(liste_hauteurs)
                hauteur_min = np.min(liste_hauteurs)
                hauteur_max = np.max(liste_hauteurs)
                variance_hauteur = np.var(liste_hauteurs)
                ecartype_hauteur = np.std(variance_hauteur)

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
