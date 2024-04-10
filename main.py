import subprocess
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import csv


def filtre_points_aberrants(matrice):
    """ Supprime les points aberrants jusqu'à ce que la variation de la moyenne soit inférieure à seuil_stable_moy % """
    matrice_filtree = matrice.copy()  # Copie de la matrice pour éviter les modifications inattendues
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
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

        # Si la variation de la moyenne est inférieure à 10%, arrêter
        if abs(nouvelle_moyenne - moyenne_actuelle) / moyenne_actuelle < seuil_stable_moy:
            break

        # Mettre à jour la matrice filtrée
        matrice_filtree = nouvelle_matrice_filtree

    return matrice_filtree


def hauteur_locale(matrice):
    # Taille des zones représentant 10% de la matrice
    zone_size = (int(matrice.shape[0] * 0.1), int(matrice.shape[1] * 0.1))
    # Calculer le nombre total de zones dans la matrice
    nombre_zones = (matrice.shape[0] // zone_size[0]) * (matrice.shape[1] // zone_size[1])

    # Initialiser les listes pour stocker les résultats
    max_locals = []
    sol_locaux = []
    hauteur = []
    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.02)])

    # Parcourir chaque zone
    for i in range(0, matrice.shape[0], zone_size[0]):
        for j in range(0, matrice.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = matrice[i:i + zone_size[0], j:j + zone_size[1]]

            # Calculer max_local et sol_local pour la zone
            zone_sans_nan = zone[~np.isnan(zone)]
            max_local = np.median(np.sort(zone_sans_nan.flatten())[:int(zone_sans_nan.size * 0.05)])
            sol_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.1)])

            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteur.append(np.nan)
            else:
                # Ajouter les résultats à la liste
                max_locals.append(max_local)
                sol_locaux.append(sol_local)
                if sol_bac - 50 <= sol_local <= sol_bac + 50:
                    hauteur.append(sol_local - max_local)
                else:
                    hauteur.append(sol_bac - max_local)

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

    # path_annee = r"C:\Users\U108-N806\Desktop\Comparaison mesures"
    path_annee = '/home/loeb/Documents/Comparaison_mesures'

    fonction = "boucle"

    # Executer le script correspondant dans Metashape
    subprocess.run([r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'test_script_metashape.py', fonction] + [path_annee])

    # PATH
    sessionlist = os.listdir(path_annee)
    for session in sorted(sessionlist):
        if session.find("Session") == 0:
            print(session)
            list_dems = os.listdir(path_annee + "/" + session + "/" + 'DEMs')
            for file in sorted(list_dems):
                print(file)
                # Récupérer la DEM et la transformer en matrice
                dem = Image.open(path_annee + "/" + session + "/" + 'DEMs' + "/" + file)
                dem_array = np.array(dem)
                dem_array[dem_array <= 0] = np.nan
                dem_array = dem_array * 1000
                plt.figure() and plt.imshow(dem_array)

                # Extraire la region du bac
                dem_cut = dem_array

                # Filtre des points aberrants
                mat_filtree = filtre_points_aberrants(dem_cut)
                plt.figure() and plt.imshow(mat_filtree)

                # Calcul des hauteurs locales
                liste_hauteurs, z_mat = hauteur_locale(mat_filtree)
                print(liste_hauteurs)
                plt.figure() and plt.imshow(z_mat, cmap='jet', vmin=0, vmax=1000)

                # Export des hauteurs locales en csv
                with open(path_annee + "/" + session + "/" + "hauteurs_l_metashape.csv", 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([session] + [file] + [str(h) for h in liste_hauteurs])


if __name__ == "__main__":
    main()
