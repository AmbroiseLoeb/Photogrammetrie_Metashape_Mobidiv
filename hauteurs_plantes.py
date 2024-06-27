# importation des bibliotheques
import math
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
import scipy.ndimage as ndi
from scipy.ndimage import label, maximum_filter, generate_binary_structure, binary_dilation
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def filtre_points_aberrants(matrice):
    """Filtre les points aberrants en utilisant des seuils basés sur l'écart-type."""

    matrice_filtree = matrice.copy()
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
    seuil_stable_moy = 0.0001

    while True:
        # Calculer la moyenne actuelle
        moyenne_actuelle = np.nanmean(matrice_filtree)

        # Définir des bornes pour filtrer les points aberrants
        ecart_type = np.nanstd(matrice_filtree)
        limite_inf = moyenne_actuelle - 5 * ecart_type
        limite_sup = moyenne_actuelle + 4 * ecart_type

        # Remplacer les points aberrants par NaN
        nouvelle_matrice_filtree = matrice_filtree.copy()
        nouvelle_matrice_filtree[(matrice_filtree < limite_inf) | (matrice_filtree > limite_sup)] = np.nan

        # Calculer la nouvelle moyenne
        nouvelle_moyenne = np.nanmean(nouvelle_matrice_filtree)

        # Si la variation de la moyenne est inférieure au seuil, arrêter
        if abs(nouvelle_moyenne - moyenne_actuelle) / moyenne_actuelle < seuil_stable_moy:
            break

        # Mettre à jour la matrice filtrée
        matrice_filtree = nouvelle_matrice_filtree

    # Re-filtrer les points les plus hauts
    matrice_filtree[(matrice_filtree < np.median(np.sort(matrice_filtree.flatten())[:int(matrice_filtree.size * 0.0005)]))] = np.nan

    return matrice_filtree


def carte_hauteur_absolue(matrice, nombre_zones):
    """Calcule la carte des hauteurs en ajustant le sol à zéro."""

    # Taille des zones
    coeff = 1/math.sqrt(nombre_zones)
    zone_size = (int(matrice.shape[0] * coeff), int(matrice.shape[1] * coeff))

    # Trouver la profondeur globale du sol
    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = np.median(np.sort(mat_sans_nan.flatten())[:int(mat_sans_nan.size * 0.03)])
    mat_hauteur = 1 * matrice.copy()
    sol_locaux = []

    for i in range(0, matrice.shape[0], zone_size[0]):  # Parcourir chaque zone
        for j in range(0, matrice.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]]

            # Calculer le sol local pour la zone
            zone_sans_nan = zone[~np.isnan(zone)]
            sol_local = np.median(np.sort(zone_sans_nan.flatten())[:int(zone_sans_nan.size * 0.03)])
            sol_locaux.append(sol_local)

            # Ramener le sol à zero
            if sol_bac - 100 <= sol_local <= sol_bac + 50:
                zone -= sol_local
            else:
                zone -= sol_bac

    return mat_hauteur, sol_bac


def hauteur_par_zone(matrice_h, nombre_zones):
    """Calcule les hauteurs locales par zone."""

    # Taille des zones
    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (int(matrice_h.shape[0] * coeff), int(matrice_h.shape[1] * coeff))

    # Calculer le maximum global du bac
    mat_sans_nan = matrice_h[~np.isnan(matrice_h)]
    max_glob = np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.02)])
    max_locals = []
    hauteurs = []

    for i in range(0, matrice_h.shape[0], zone_size[0]):  # Parcourir chaque zone
        for j in range(0, matrice_h.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = matrice_h[i:i + zone_size[0], j:j + zone_size[1]]

            zone_sans_nan = zone[~np.isnan(zone)]
            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteurs.append(np.inf)
            else:
                mean_local = np.mean(zone_sans_nan.flatten())
                # Calculer la hauteur locale pour la zone
                max_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.03)])
                max_locals.append(max_local)
                if max_local > max_glob/5:
                    # Ajouter le résultat à la liste
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

    # Ajouter les valeurs de hauteur et le numéro de la zone
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

    return hauteurs, mat_zones_hauteur, figure_hauteurs


def hauteur_par_sommet(matrice_h):
    """Calcule les hauteurs locales par sommet."""

    # Combler valeurs manquantes
    matrice_h[np.isinf(matrice_h)] = 0
    matrice_h[np.isnan(matrice_h)] = 0
    matrice_h[matrice_h <= 50] = 0

    # Convertir la matrice de hauteur filtrée en une image 8 bits pour le traitement d'image
    z_map_8u = cv.normalize(matrice_h, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Appliquer la fermeture morphologique (dilatation suivie d'érosion)
    kernel = np.ones((5, 5), np.uint8)
    closed_height_map = cv.morphologyEx(z_map_8u, cv.MORPH_CLOSE, kernel, iterations=3)

    # Trouver les maxima locaux
    neighborhood_base = generate_binary_structure(2, 2)
    neighborhood = binary_dilation(neighborhood_base, iterations=3).astype(neighborhood_base.dtype)
    local_max = maximum_filter(closed_height_map, footprint=neighborhood) == closed_height_map
    local_max = local_max & (closed_height_map > 0)  # Filtrer les points où la distance est nulle

    # Appliquer un seuil de hauteur pour réduire les maxima locaux
    height_threshold = 0.5 * np.max(closed_height_map)  # Ajuster ce seuil selon les besoins
    local_max = local_max & (closed_height_map > height_threshold)

    # Étiqueter les composants connectés des maxima locaux
    labeled, num_features = ndi.label(local_max)

    # Extraire les coordonnées des sommets
    coords = np.column_stack(np.where(local_max))

    # Appliquer DBSCAN pour regrouper les sommets proches
    clustering = DBSCAN(eps=20, min_samples=5).fit(coords)  # Ajuster 'eps' selon la taille des plantes
    labels = clustering.labels_

    # Trouver les sommets les plus élevés dans chaque groupe
    unique_labels = set(labels)
    summit_heights = []
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            label_mask = (labels == label)
            summit_coords = coords[label_mask]
            summit_heights.append(np.max(matrice_h[summit_coords[:, 0], summit_coords[:, 1]]))

    # Afficher les résultats visuellement
    fig, ax = plt.subplots()
    ax.imshow(matrice_h, cmap='gray')
    plt.axis('off')

    for idx, label in enumerate(unique_labels):
        if label != -1:  # Ignore noise points
            label_mask = (labels == label)
            summit_coords = coords[label_mask]
            summit_height = np.max(matrice_h[summit_coords[:, 0], summit_coords[:, 1]])
            if summit_height >= 0:
                centroid = summit_coords.mean(axis=0).astype(int)
                ax.scatter(centroid[1], centroid[0], c='white', s=3
                           )
                ax.text(centroid[1], centroid[0], f"{int(summit_height):3d}", color='blue', fontsize=5, ha='center', va='bottom')
                ax.text(centroid[1], centroid[0], str(idx + 1), color='red', fontsize=3, ha='right', va='top')

    ax.set_title('Sommets des plantes identifiés')
    blue_patch = mpatches.Patch(color='blue', label='Hauteur du sommet en mm')
    red_patch = mpatches.Patch(color='red', label='Numéro du sommet')
    plt.legend(handles=[blue_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
               fontsize='small')
    plt.subplots_adjust(bottom=0.2)  # Ajustement de la figure pour éviter que la légende ne soit coupée

    return summit_heights, fig
