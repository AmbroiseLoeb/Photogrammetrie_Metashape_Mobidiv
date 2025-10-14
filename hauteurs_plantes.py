import open3d as o3d
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_dilation, label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


# === CHARGEMENT DU FICHIER PLY ===
def charger_ply_vers_numpy(path_ply):
    """Charge un fichier PLY et retourne les points sous forme de tableau NumPy (en mm)."""
    try:
        pcd = o3d.io.read_point_cloud(path_ply)
        points = np.asarray(pcd.points)
        if points.size == 0:
            raise ValueError("Le fichier PLY est vide ou corrompu.")
        points *= 1000  # Convertir mètres en mm
        return points
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier PLY : {e}")


# === CONVERSION VERS UNE Z-MAP ===
def creer_carte_hauteur_from_points(points, resolution=0.5, max_attempts=5, attempt=1):
    """Convertit un nuage de points en une carte de hauteur (Z-map)."""
    if points.shape[0] == 0:
        raise ValueError("Aucun point fourni pour créer la carte de hauteur.")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x = x - x.min()
    y = y - y.min()

    x_range = x.max() if x.max() > 0 else 1e-3
    y_range = y.max() if y.max() > 0 else 1e-3
    if x_range < 1e-3 or y_range < 1e-3:
        raise ValueError("Les points sont trop concentrés (étendue spatiale quasi nulle).")

    width = int(np.ceil(x.max() / resolution)) + 1
    height = int(np.ceil(y.max() / resolution)) + 1
    if width <= 0 or height <= 0:
        raise ValueError("Résolution trop grande ou données insuffisantes pour créer la carte.")

    if width < 4 or height < 4:
        if attempt >= max_attempts:
            raise ValueError(f"Z-map trop petite ({height}x{width}) après {max_attempts} tentatives.")
        print(f"Avertissement : Z-map trop petite ({height}x{width}). Tentative {attempt}/{max_attempts}.")
        new_resolution = min(x_range, y_range) / 4
        print(f"Ajustement de la résolution à {new_resolution:.3f}...")
        return creer_carte_hauteur_from_points(points, resolution=new_resolution, max_attempts=max_attempts, attempt=attempt + 1)

    height_map = np.full((height, width), np.nan)
    indices = (y / resolution).astype(int) * width + (x / resolution).astype(int)
    for idx, zi in zip(indices, z):
        i, j = divmod(int(idx), width)
        if 0 <= i < height and 0 <= j < width:
            if np.isnan(height_map[i, j]) or zi > height_map[i, j]:
                height_map[i, j] = zi
    return height_map


# === FILTRAGE DE BRUIT ===
def filtre_points_aberrants(matrice):
    """Filtre les points aberrants en utilisant des seuils basés sur l'écart-type."""
    if np.all(np.isnan(matrice)):
        raise ValueError("La matrice est entièrement NaN, impossible de filtrer.")

    matrice_filtree = matrice.copy()
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
    seuil_stable_moy = 0.0001
    print(np.nanmax(matrice_filtree))
    while True:
        valid_data = matrice_filtree[~np.isnan(matrice_filtree)]
        if len(valid_data) == 0:
            raise ValueError("Aucune donnée valide après filtrage.")

        moy = np.nanmean(matrice_filtree)
        std = np.nanstd(matrice_filtree)
        lim_inf = moy - 3 * std
        lim_sup = moy + 10 * std

        nouvelle = matrice_filtree.copy()
        nouvelle[(matrice_filtree < lim_inf) | (matrice_filtree > lim_sup)] = np.nan

        print(np.nanmax(nouvelle))
        new_valid_data = nouvelle[~np.isnan(nouvelle)]
        if len(new_valid_data) == 0 or abs(np.nanmean(nouvelle) - moy) / (moy + 1e-10) < seuil_stable_moy:
            break
        matrice_filtree = nouvelle

    print(f"Valeurs non-NaN après filtrage : {np.count_nonzero(~np.isnan(matrice_filtree))}")
    return matrice_filtree


# === AJUSTEMENT AU SOL ===
def carte_hauteur_absolue(matrice, nombre_zones=16):
    """Calcule la carte des hauteurs en ajustant le sol à zéro."""
    if np.all(np.isnan(matrice)):
        raise ValueError("La matrice est entièrement NaN, impossible d'ajuster.")

    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (max(int(matrice.shape[0] * coeff), 1), max(int(matrice.shape[1] * coeff), 1))

    mat_sans_nan = matrice[~np.isnan(matrice)]
    if len(mat_sans_nan) == 0:
        raise ValueError("Aucune donnée valide pour estimer le sol.")
    sol_bac = np.median(np.sort(mat_sans_nan)[:max(int(len(mat_sans_nan) * 0.03), 1)])

    mat_hauteur = matrice.copy()
    for i in range(0, matrice.shape[0], zone_size[0]):
        for j in range(0, matrice.shape[1], zone_size[1]):
            zone = mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]]
            sans_nan = zone[~np.isnan(zone)]
            if len(sans_nan) == 0:
                continue
            sol_local = np.median(np.sort(sans_nan)[:max(int(len(sans_nan) * 0.03), 1)])
            if sol_bac - 200 <= sol_local <= sol_bac + 200:
                zone -= sol_local
            else:
                zone -= sol_bac

    print(f"Valeurs non-NaN après recalage : {np.count_nonzero(~np.isnan(mat_hauteur))}")
    print(f"Plage des hauteurs après recalage : {np.nanmin(mat_hauteur):.3f} à {np.nanmax(mat_hauteur):.3f}")
    return mat_hauteur, sol_bac


# === HAUTEURS PAR ZONE ===
def hauteur_par_zone(matrice_h, nombre_zones=16):
    """Calcule les hauteurs locales par zone."""
    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (max(int(matrice_h.shape[0] * coeff), 1), max(int(matrice_h.shape[1] * coeff), 1))

    mat_sans_nan = matrice_h[~np.isnan(matrice_h)]
    if len(mat_sans_nan) == 0:
        print("Aucune donnée valide pour les zones.")
        return [], np.zeros_like(matrice_h), plt.figure()

    max_glob = np.median(np.sort(mat_sans_nan)[::-1][:max(int(len(mat_sans_nan) * 0.001), 1)])
    hauteurs = []

    for i in range(0, matrice_h.shape[0], zone_size[0]):
        for j in range(0, matrice_h.shape[1], zone_size[1]):
            zone = matrice_h[i:i + zone_size[0], j:j + zone_size[1]]
            zone_sans_nan = zone[~np.isnan(zone)]
            expected_pixels = zone_size[0] * zone_size[1]
            if len(zone_sans_nan) < 0.1 * expected_pixels:  # Seuil réduit
                hauteurs.append(np.inf)
            else:
                max_local = np.median(np.sort(zone_sans_nan)[::-1][:max(int(len(zone_sans_nan) * 0.001), 1)])
                hauteurs.append(max_local)
                # Pourquoi Ambroise avait-il choisi ça ??
                # if max_local > max_glob / 10:  # Seuil réduit
                #     hauteurs.append(max_local)
                # else:
                #     hauteurs.append(np.nan)

    print(f"Zones marquées inf : {len([h for h in hauteurs if math.isinf(h)])}")
    print(f"Zones marquées NaN : {len([h for h in hauteurs if math.isnan(h)])}")
    print(f"Zones valides : {len([h for h in hauteurs if not math.isnan(h) and not math.isinf(h)])}")

    hauteur_a = np.array([int(round(h)) if not math.isinf(h) and not math.isnan(h) else (np.inf if math.isinf(h) else np.nan) for h in hauteurs])
    mat_zones_hauteur = np.zeros_like(matrice_h)
    index = 0
    for i in range(0, mat_zones_hauteur.shape[0], zone_size[0]):

        for j in range(0, mat_zones_hauteur.shape[1], zone_size[1]):
            mat_zones_hauteur[i:i + zone_size[0], j:j + zone_size[1]] = hauteur_a[index]
            index += 1

    plt.ioff()
    figure_hauteurs, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(mat_zones_hauteur, cmap='viridis', interpolation='none')
    figure_hauteurs.colorbar(cax, ax=ax, label='Hauteur (mm)')
    plt.axis('off')

    index = 0
    numero_z = 1
    for i in range(0, mat_zones_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_zones_hauteur.shape[1], zone_size[1]):
            if not np.isinf(hauteur_a[index]):
                ax.text(j + zone_size[1] / 10, i + zone_size[0] * 0.9, f"{numero_z}", color='red', ha='left', va='bottom', fontsize=6)
                numero_z += 1
                if not np.isnan(hauteur_a[index]):
                    ax.text(j + zone_size[1] / 2, i + zone_size[0] / 2 - zone_size[0] / 4, f"{int(hauteur_a[index]):3d}", color='white', ha='center', va='center', fontsize=8)
            index += 1

    ax.set_title(f'Hauteurs maximale du couvert par zone ({nombre_zones})')
    return hauteurs, mat_zones_hauteur, figure_hauteurs


# === HAUTEURS PAR SOMMET ===
def hauteur_par_sommet(matrice_h):
    """Calcule les hauteurs locales par sommet."""
    matrice_h = matrice_h.copy()
    matrice_h[np.isinf(matrice_h)] = 0
    matrice_h[np.isnan(matrice_h)] = 0

    valid_heights = matrice_h[matrice_h > 0]
    if len(valid_heights) == 0:
        print("Aucune donnée valide dans la carte de hauteur.")
        return [], plt.figure()

    seuil = np.percentile(valid_heights, 10) if len(valid_heights) > 0 else 1.0
    print(f"Seuil de hauteur utilisé : {seuil:.3f} mm")
    matrice_h[matrice_h <= seuil] = 0

    z_map_8u = cv.normalize(matrice_h, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    print(f"Valeurs non-zéro après normalisation : {np.count_nonzero(z_map_8u)}")
    closed = cv.morphologyEx(z_map_8u, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    neighborhood = binary_dilation(generate_binary_structure(2, 2), iterations=2)
    local_max = maximum_filter(closed, footprint=neighborhood) == closed
    local_max = local_max & (closed > 0)
    local_max &= closed > 0.3 * np.max(closed)

    labeled, _ = label(local_max)
    coords = np.column_stack(np.where(local_max))
    print(f"Nombre de maxima locaux détectés : {len(coords)}")
    if coords.size == 0:
        print("Aucun maximum local détecté.")
        return [], plt.figure()

    clustering = DBSCAN(eps=3, min_samples=2).fit(coords)
    labels = clustering.labels_
    print(f"Labels DBSCAN : {labels}")
    if len(labels) == 0:
        print("Aucun cluster détecté par DBSCAN.")
        return [], plt.figure()

    unique = set(labels)
    if not unique or unique == {-1}:
        print("Aucun cluster valide détecté par DBSCAN.")
        return [], plt.figure()

    summit_heights = []
    summit_coords_list = []
    for lbl in unique - {-1}:
        mask = labels == lbl
        summit_coords = coords[mask]
        if summit_coords.size == 0:
            continue
        hauteur = np.max(matrice_h[summit_coords[:, 0], summit_coords[:, 1]])
        if hauteur > 0:
            summit_heights.append(hauteur)
            summit_coords_list.append(summit_coords)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(matrice_h, cmap='gray')
    plt.axis('off')

    for idx, (hauteur, coords) in enumerate(zip(summit_heights, summit_coords_list)):
        centroid = coords.mean(axis=0).astype(int)
        ax.scatter(centroid[1], centroid[0], c='white', s=5)
        ax.text(centroid[1], centroid[0], f"{int(hauteur):3d}", color='blue', fontsize=8, ha='center', va='bottom')
        ax.text(centroid[1], centroid[0], str(idx + 1), color='red', fontsize=6, ha='right', va='top')

    ax.set_title('Sommets des plantes identifiés')
    blue_patch = mpatches.Patch(color='blue', label='Hauteur du sommet en mm')
    red_patch = mpatches.Patch(color='red', label='Numéro du sommet')
    plt.legend(handles=[blue_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize='small')
    plt.subplots_adjust(bottom=0.2)

    if not summit_heights:
        print("Aucun sommet avec une hauteur significative détecté.")

    return summit_heights, fig


# === PIPELINE PRINCIPAL ===
def analyser_plante_depuis_ply(path_ply, nombre_zones=9):  # Réduit à 9 zones
    """Pipeline pour analyser un fichier PLY et extraire hauteurs par zone et sommet."""
    try:
        print("Chargement...")
        points = charger_ply_vers_numpy(path_ply)
        print(f"Points chargés : {points.shape[0]}")
        print(f"X range: {points[:, 0].min():.3f} à {points[:, 0].max():.3f} mm")
        print(f"Y range: {points[:, 1].min():.3f} à {points[:, 1].max():.3f} mm")
        print(f"Z range: {points[:, 2].min():.3f} à {points[:, 2].max():.3f} mm")

        print("Conversion en Z-map...")
        zmap = creer_carte_hauteur_from_points(points, resolution=0.5)
        print(f"Dimensions de la Z-map : {zmap.shape}")
        print(f"Valeurs non-NaN dans Z-map initiale : {np.count_nonzero(~np.isnan(zmap))}")

        print("Filtrage...")
        z_filtrée = filtre_points_aberrants(zmap)

        print("Recalage sol...")
        z_corrigée, sol = carte_hauteur_absolue(z_filtrée, nombre_zones)
        print(f"Niveau du sol estimé : {sol:.3f} mm")

        print("Calcul hauteurs par zone...")
        hauteurs_zones, mat_zones_hauteur, fig_zones = hauteur_par_zone(z_corrigée, nombre_zones)
        valid_zones = [h for h in hauteurs_zones if not math.isnan(h) and not math.isinf(h)]
        print(f"Nombre de zones valides : {len(valid_zones)}")
        fig_zones.savefig('hauteurs_zones.png')
        plt.close(fig_zones)

        print("Calcul hauteurs par sommet...")
        hauteurs_sommets, fig_sommets = hauteur_par_sommet(z_corrigée)
        print(f"Nombre de sommets détectés : {len(hauteurs_sommets)}")
        fig_sommets.savefig('hauteurs_sommets.png')
        plt.close(fig_sommets)

        print("\n✅ Résultats :")
        print(f" - Hauteur max (zones) : {max(valid_zones, default=0):.1f} mm")
        print(f" - Hauteur moyenne (zones) : {np.mean(valid_zones) if valid_zones else 0:.1f} mm")
        print(f" - Hauteur max (sommets) : {max(hauteurs_sommets, default=0):.1f} mm")
        print(f" - Hauteur moyenne (sommets) : {np.mean(hauteurs_sommets) if hauteurs_sommets else 0:.1f} mm")
        print(f" - Nombre de sommets : {len(hauteurs_sommets)}")

    except Exception as e:
        print(f"❌ Erreur dans le pipeline : {e}")
        raise


# === UTILISATION ===
if __name__ == "__main__":
    try:
        analyser_plante_depuis_ply(r"C:\Users\U108-N806\Desktop\stage_rafik_2025\maquettes_ble\maquette_rapide.ply", nombre_zones=9)
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")