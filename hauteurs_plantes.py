# importation des bibliotheques
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


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

    # Re-filtrer les points les plus hauts
    matrice_filtree[(matrice_filtree < np.median(np.sort(matrice_filtree.flatten())[:int(matrice_filtree.size * 0.0005)]))] = np.nan

    return matrice_filtree


def carte_hauteur_absolue(matrice, nombre_zones):
    # Taille des zones représentant n% de la matrice
    coeff = 1/math.sqrt(nombre_zones)
    zone_size = (int(matrice.shape[0] * coeff), int(matrice.shape[1] * coeff))

    # Initialiser les listes pour stocker les résultats
    sol_locaux = []

    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = - np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.03)])
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

    return mat_hauteur, sol_bac


def hauteur_par_zone(matrice_h, nombre_zones):
    # Taille des zones représentant n% de la matrice
    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (int(matrice_h.shape[0] * coeff), int(matrice_h.shape[1] * coeff))

    # Initialiser les listes pour stocker les résultats
    max_locals = []
    hauteurs = []
    mat_sans_nan = matrice_h[~np.isnan(matrice_h)]
    max_glob = np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.02)])

    for i in range(0, matrice_h.shape[0], zone_size[0]):
        for j in range(0, matrice_h.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = matrice_h[i:i + zone_size[0], j:j + zone_size[1]]

            zone_sans_nan = zone[~np.isnan(zone)]
            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteurs.append(np.inf)
            else:
                mean_local = np.mean(zone_sans_nan.flatten())
                max_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.03)])
                max_locals.append(max_local)
                if max_local > max_glob/5:
                    # Ajouter les résultats à la liste
                    hauteurs.append(max_local)
                else:
                    hauteurs.append(np.nan)

    # Representation graphique des hauteurs par zone
    hauteur_a = np.array([int(round(h)) if not math.isinf(h) and not math.isnan(h) else (np.inf if math.isinf(h) else np.nan) for h in hauteurs])
    mat_zones_hauteur = np.zeros_like(matrice_h)
    index = 0
    for i in range(0, mat_zones_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_zones_hauteur.shape[1], zone_size[1]):
            # Assigner la valeur de hauteur correspondante à chaque point de la zone
            mat_zones_hauteur[i:i + zone_size[0], j:j + zone_size[1]] = hauteur_a[index]
            index += 1

    # Création de la figure et de l'axe
    plt.ioff()  # desactive l'affichage automatique
    figure_hauteurs, ax = plt.subplots()
    cax = ax.imshow(mat_zones_hauteur, cmap='viridis', interpolation='none')
    figure_hauteurs.colorbar(cax, ax=ax, label='Hauteur (mm)')
    plt.axis('off')  # Désactiver les axes

    # Ajouter les valeurs numériques sur la partie supérieure de chaque zone
    index = 0
    numero_z = 1
    for i in range(0, mat_zones_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_zones_hauteur.shape[1], zone_size[1]):
            if not np.isinf(hauteur_a[index]):
                ax.text(j + zone_size[1] / 10, i + zone_size[0] * 0.9, f"{numero_z}", color='red', ha='left', va='bottom', fontsize=4)
                numero_z += 1
                if not np.isnan(hauteur_a[index]):
                    ax.text(j + zone_size[1] / 2, i + zone_size[0] / 2 - zone_size[0] / 4, f"{int(hauteur_a[index]):3d}", color='white', ha='center', va='center', fontsize=7)
            index += 1

    ax.set_title(f'Hauteurs maximale du couvert par zone ({nombre_zones})')

    # Retourner l'objet figure
    return hauteurs, mat_zones_hauteur, figure_hauteurs

