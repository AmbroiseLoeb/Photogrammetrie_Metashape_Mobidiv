import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_dilation, label
import cv2 as cv
import math


# === CHARGEMENT DU FICHIER PLY ===
def charger_ply_vers_numpy(path_ply):
    """Charge un fichier PLY et retourne les points sous forme de tableau NumPy (en mm)."""
    try:
        pcd = o3d.io.read_point_cloud(path_ply)
        points = np.asarray(pcd.points)
        if points.size == 0:
            raise ValueError("Le fichier PLY est vide ou corrompu.")
        # Convertir mètres en millimètres
        points *= 1000
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

    # Vérifier l'étendue des points
    x_range = x.max() if x.max() > 0 else 1e-3  # Éviter division par zéro (en mm)
    y_range = y.max() if y.max() > 0 else 1e-3
    if x_range < 1e-3 or y_range < 1e-3:
        raise ValueError("Les points sont trop concentrés (étendue spatiale quasi nulle).")

    # Calculer les dimensions de la carte
    width = int(np.ceil(x.max() / resolution)) + 1
    height = int(np.ceil(y.max() / resolution)) + 1
    if width <= 0 or height <= 0:
        raise ValueError("Résolution trop grande ou données insuffisantes pour créer la carte.")

    # Vérifier si la Z-map est trop petite
    if width < 4 or height < 4:
        if attempt >= max_attempts:
            raise ValueError(f"Z-map trop petite ({height}x{width}) après {max_attempts} tentatives. Résolution inadaptée ou données insuffisantes.")
        print(f"Avertissement : Z-map trop petite ({height}x{width}). Tentative {attempt}/{max_attempts}.")
        new_resolution = min(x_range, y_range) / 4
        print(f"Ajustement de la résolution à {new_resolution:.3f}...")
        return creer_carte_hauteur_from_points(points, resolution=new_resolution, max_attempts=max_attempts, attempt=attempt + 1)

    # Initialiser la carte avec NaN
    height_map = np.full((height, width), np.nan)

    # Remplir la carte avec les hauteurs maximales
    indices = (y / resolution).astype(int) * width + (x / resolution).astype(int)
    for idx, zi in zip(indices, z):
        i, j = divmod(int(idx), width)
        if 0 <= i < height and 0 <= j < width:
            if np.isnan(height_map[i, j]) or zi > height_map[i, j]:
                height_map[i, j] = zi
    return height_map


# === FILTRAGE DE BRUIT ===
def filtre_points_aberrants(matrice):
    """Filtre les points aberrants dans la carte de hauteur."""
    if np.all(np.isnan(matrice)):
        raise ValueError("La matrice est entièrement NaN, impossible de filtrer.")

    matrice = matrice.copy()
    matrice[np.isinf(matrice)] = np.nan

    # Filtrage itératif basé sur la moyenne et l'écart-type
    seuil_stable_moy = 0.0001
    while True:
        valid_data = matrice[~np.isnan(matrice)]
        if len(valid_data) == 0:
            raise ValueError("Aucune donnée valide après filtrage.")

        moy = np.mean(valid_data)
        std = np.std(valid_data)
        lim_inf = moy - 3 * std
        lim_sup = moy + 3 * std

        nouvelle = matrice.copy()
        nouvelle[(matrice < lim_inf) | (matrice > lim_sup)] = np.nan

        new_valid_data = nouvelle[~np.isnan(nouvelle)]
        if len(new_valid_data) == 0 or abs(np.mean(new_valid_data) - moy) / (moy + 1e-10) < seuil_stable_moy:
            break
        matrice = nouvelle

    print(f"Valeurs non-NaN après filtrage : {np.count_nonzero(~np.isnan(matrice))}")
    return matrice


# === AJUSTEMENT AU SOL ===
def carte_hauteur_absolue(matrice, nombre_zones=16):
    """Ajuste la carte de hauteur par rapport au sol."""
    if np.all(np.isnan(matrice)):
        raise ValueError("La matrice est entièrement NaN, impossible d'ajuster.")

    # Calculer la taille des zones avec un minimum de 1
    coeff = 1 / math.sqrt(nombre_zones)
    zone_size = (max(int(matrice.shape[0] * coeff), 1), max(int(matrice.shape[1] * coeff), 1))

    # Estimer le niveau du sol global
    mat_sans_nan = matrice[~np.isnan(matrice)]
    if len(mat_sans_nan) == 0:
        raise ValueError("Aucune donnée valide pour estimer le sol.")
    sol_bac = np.median(np.sort(mat_sans_nan)[:max(int(len(mat_sans_nan) * 0.03), 1)])

    # Ajuster chaque zone
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


# === DETECTION DE SOMMETS ===
def hauteur_par_sommet(matrice_h):
    """Détecte les sommets dans la carte de hauteur."""
    matrice_h = matrice_h.copy()
    matrice_h[np.isnan(matrice_h)] = 0

    # Adapter le seuil dynamiquement à l'échelle des données
    valid_heights = matrice_h[matrice_h > 0]
    if len(valid_heights) == 0:
        print("Aucune donnée valide dans la carte de hauteur après filtrage.")
        return []
    seuil = np.percentile(valid_heights, 10) if len(valid_heights) > 0 else 1.0  # En mm
    print(f"Seuil de hauteur utilisé : {seuil:.3f} mm")
    matrice_h[matrice_h <= seuil] = 0

    # Normalisation pour traitement d'image
    z_map_8u = cv.normalize(matrice_h, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    print(f"Valeurs non-zéro après normalisation : {np.count_nonzero(z_map_8u)}")
    closed = cv.morphologyEx(z_map_8u, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)  # Noyau minimal

    # Détection des maxima locaux
    neighborhood = binary_dilation(generate_binary_structure(2, 2), iterations=1)  # Minimal
    local_max = maximum_filter(closed, footprint=neighborhood) == closed
    local_max = local_max & (closed > 0)
    local_max &= closed > 0.3 * np.max(closed)  # Seuil encore plus bas

    # Étiquetage et clustering
    labeled, _ = label(local_max)
    coords = np.column_stack(np.where(local_max))
    print(f"Nombre de maxima locaux détectés : {len(coords)}")
    if coords.size == 0:
        print("Aucun maximum local détecté dans la carte de hauteur.")
        return []

    clustering = DBSCAN(eps=3, min_samples=2).fit(coords)  # Très permissif
    labels = clustering.labels_
    print(f"Labels DBSCAN : {labels}")
    if len(labels) == 0:
        print("Aucun cluster détecté par DBSCAN.")
        return []

    unique = set(labels)
    if not unique or unique == {-1}:
        print("Aucun cluster valide détecté par DBSCAN (tous les points sont bruit).")
        return []

    # Calculer les hauteurs des sommets
    sommet_hauteurs = []
    for lbl in unique - {-1}:
        mask = labels == lbl
        sommet_coords = coords[mask]
        if sommet_coords.size == 0:
            continue
        hauteur = np.max(matrice_h[sommet_coords[:, 0], sommet_coords[:, 1]])
        if hauteur > 0:
            sommet_hauteurs.append(hauteur)

    if not sommet_hauteurs:
        print("Aucun sommet avec une hauteur significative détecté.")
    return sommet_hauteurs


# === EXTRACTION DE FEATURES ===
def extraire_features(mat_hauteur, sommets):
    """Extrait les caractéristiques de la plante."""
    surface = np.count_nonzero(mat_hauteur > 5.0)  # Seuil en mm
    volume = np.nansum(mat_hauteur)
    h_max = max(sommets) if sommets else 0
    h_moy = np.mean(sommets) if sommets else 0
    nb_sommets = len(sommets)
    return {
        "hauteur_max": round(h_max, 1),
        "hauteur_moyenne": round(h_moy, 1),
        "surface_proj_mm2": surface,
        "volume_mm3": int(volume),
        "nb_sommets": nb_sommets
    }


# === PIPELINE PRINCIPAL ===
def analyser_plante_depuis_ply(path_ply):
    """Pipeline principal pour analyser un fichier PLY d'une plante."""
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
        z_corrigée, sol = carte_hauteur_absolue(z_filtrée)
        print(f"Niveau du sol estimé : {sol:.3f} mm")

        print("Détection sommets...")
        sommets = hauteur_par_sommet(z_corrigée)
        print(f"Nombre de sommets détectés : {len(sommets)}")

        print("Extraction features...")
        features = extraire_features(z_corrigée, sommets)

        print("\n✅ Résultats :")
        for k, v in features.items():
            print(f" - {k}: {v}")

        # Visualisation
        plt.figure(figsize=(8, 6))
        plt.imshow(z_corrigée, cmap='viridis')
        plt.title("Carte de hauteur (plante)")
        plt.axis('off')
        plt.colorbar(label='Hauteur (mm)')
        plt.savefig('plant_height_map.png')
        print("Carte de hauteur sauvegardée sous 'plant_height_map.png'")

    except Exception as e:
        print(f"❌ Erreur dans le pipeline : {e}")
        raise


# === UTILISATION ===
if __name__ == "__main__":
    try:
        analyser_plante_depuis_ply(r"C:\Users\U108-N806\Desktop\stage_rafik_2025\maquettes_ble\maquette_rapide.ply")
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")