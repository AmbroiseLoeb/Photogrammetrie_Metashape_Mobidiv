# importation des bibliotheques
import Metashape
import os
import sys


def boucle(path):
    """Fonction principale pour traiter chaque dossier plot."""

    def workflow(path_dossier, dossier_plot=''):
        """Traite les images par photogramétrie jusqu'au DEM (digital elevation model)."""

        # Creation du chunk
        chk = doc.addChunk()

        # Importer les photos
        liste_images = []
        print("      PATH_IMAGE")
        print(path_dossier + '/' + dossier_plot)
        for filename in os.listdir(path_dossier + '/' + dossier_plot):
            if filename.endswith(".jpg"):
                # Ajouter le nom du fichier à la liste d'images
                liste_images.append(filename)
        for image in liste_images:
            path_image = path_dossier + '/' + dossier_plot + '/' + image
            chk.addPhotos(path_image)

        # Importer des masques pour les photos (optionnel)
        '''
        for camera in chk.cameras:
            path_mask = str(path_dossier + '/' + dossier_plot + '/' + camera.label.split('RGB')[0] + 'MASK.png')
            chk.generateMasks(path=path_mask, masking_mode=Metashape.MaskingModeFile,
                              mask_operation=Metashape.MaskOperationReplacement, cameras=camera, tolerance=10)
        '''

        # Créer l'échelle de référence à partir de la distance entre deux caméras
        chk.addScalebar(chk.cameras[0], chk.cameras[1])
        chk.scalebars[0].label = "dist_cam_ref"
        chk.scalebars[0].reference.distance = 0.343  # mesuré sur la nacelle
        # chk.scalebars[0].reference.distance = 0.06511302126767482  # entre camera Literal
        chk.updateTransform()

        # Aligner les photos
        chk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True, keypoint_limit=0)
        chk.alignCameras()

        # Délimiter la region d'intérêt (le bac) :

        # Définir le centre des 4 cameras formant un rectangle
        for camera in chk.cameras:
            if 'CAM5' in camera.label:
                c1 = camera.center
            if 'CAM4' in camera.label:
                c2 = camera.center
            if 'CAM6' in camera.label:
                c3 = camera.center
            if 'CAM3' in camera.label:
                c4 = camera.center

        # Calculer le centre de la nouvelle région (bbox)
        bbox_center = (c1 + c2 + c3 + c4) / 4

        # Calculer les dimensions de la nouvelle région
        x_size = (c1 - c2).norm()
        y_size = 1.2 * x_size
        z_size = 5  # Définir une profondeur fixe

        # Calculer les axes directionnels basés sur les centres des caméras
        x_axis = (c2 - c1).normalized()
        y_axis = Metashape.Vector.cross(c2 - c1, c4 - c1).normalized()
        y_axis = Metashape.Vector.cross(y_axis, x_axis).normalized()  # Recalculer y_axis pour qu'il soit orthogonal à x_axis
        z_axis = - Metashape.Vector.cross(x_axis, y_axis).normalized()  # z_axis est perpendiculaire à x_axis et y_axis

        # Créer la matrice de rotation
        R = Metashape.Matrix([x_axis, y_axis, z_axis]).t()  # Transposer la matrice pour l'aligner correctement
        bbox_center -= z_axis * (z_size / 2)  # Ajuster le centre de la région pour que les caméras soient aux coins supérieurs

        # Définir la nouvelle région (bbox)
        region = chk.region
        region.center = bbox_center
        region.size = Metashape.Vector([x_size, y_size, z_size])
        region.rot = R
        chk.region = region

        # Convertir la région en coordonnées locales (à l'échelle) :

        # Récupérer le centre, la taille et la rotation de la région
        bbox_center = region.center
        bbox_size = region.size
        bbox_rot = region.rot

        # Définir les axes de la région
        x_axis = Metashape.Vector([bbox_rot[0, 0], bbox_rot[1, 0], bbox_rot[2, 0]])
        y_axis = - Metashape.Vector([bbox_rot[0, 1], bbox_rot[1, 1], bbox_rot[2, 1]])
        z_axis = Metashape.Vector([bbox_rot[0, 2], bbox_rot[1, 2], bbox_rot[2, 2]])

        # Créer une matrice 4x4 pour la transformation
        transform_matrix = Metashape.Matrix([
            [x_axis.x, y_axis.x, z_axis.x, bbox_center.x],
            [x_axis.y, y_axis.y, z_axis.y, bbox_center.y],
            [x_axis.z, y_axis.z, z_axis.z, bbox_center.z],
            [0, 0, 0, 1]
        ])

        # Inverser la matrice de transformation
        transform_matrix_inv = transform_matrix.inv()
        # Appliquer la transformation de coordonnées
        chk.transform.matrix = transform_matrix_inv
        chk.updateTransform()

        # Construire le nuage de point
        chk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering, reuse_depth=False, max_neighbors=16,
                           subdivide_task=True, workitem_size_cameras=20, max_workgroup_size=100)

        chk.buildPointCloud(source_data=Metashape.DepthMapsData, point_colors=True, point_confidence=True,
                            keep_depth=True, max_neighbors=10000, uniform_sampling=True, points_spacing=0.1,
                            subdivide_task=True, workitem_size_cameras=20, max_workgroup_size=100, replace_asset=False)
        '''
        Optionel : 
        
        # Filtre par confiance
        chk.point_cloud.setConfidenceFilter(0, 1)
        all_points_classes = list(range(128))
        chk.point_cloud.removePoints(all_points_classes)  # removes all active points of the point cloud, i.e. removing all low-confidence points
        chk.point_cloud.resetFilters()  # resetting filter, so high-confidence points are now active

        # Filtre par couleur
        gray_color = [84, 66, 52]
        tolerance = 25
        chk.point_cloud.selectPointsByColor(gray_color, tolerance, channels='RGB')
        chk.point_cloud.assignClassToSelection(7)
        
        # Classification du sol
        chk.point_cloud.classifyGroundPoints(max_angle=30, max_distance=0.01, max_terrain_slope=10, cell_size=0.5)
        # chk.point_cloud.setClassesFilter(Metashape.Ground)  # Affiche les points de sol
        '''

        doc.save()  # Sauvegarde du projet

        # Construire le modele numerique d'elévation (DEM) :

        # DEM toutes classes
        chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.DisabledInterpolation)
        chk.elevation.label = "DEM"
        key_allclass = chk.elevation.key

        """
        # DEM sol, si classification des points de sol
        chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.Extrapolated, classes=Metashape.PointClass.Ground)
        chk.elevation.label = "DEM_sol"
        key_sol = chk.elevation.key
        
        # DEM sol, si import d'une DEM sol existante
        chk.importRaster(path='path_vers_DEM_sol.tif', raster_type=Metashape.ElevationData, nodata_value=-32767)
        chk.elevation.active = True  # S'assurer que la DEM est active
        chk.elevation.label = "DEM_sol"
        key_sol = chk.elevation.key

        # DEM finale
        chk.transformRaster(source_data=Metashape.ElevationData, asset=key_allclass, subtract=True, operand_asset=key_sol)
        chk.elevation.label = "DEM_finale"
        """

        # Export de la DEM
        if len(dossier_plot) > 1:  # Si dossier session
            label_dem = chk.elevation.label + '_' + dossier_plot
        else:  # Si dossier plot
            label_dem = chk.elevation.label + '_' + os.path.basename(os.path.normpath(path_dossier))
        doc.save()
        chk.exportRaster(path_dossier + '/' + "DEMs" '/' + label_dem + '.tif',
                         source_data=Metashape.ElevationData)

    # Créer un nouveau document projet Metashape
    doc = Metashape.Document()

    # Enregistrer le projet
    # path = Metashape.app.getSaveFileName("Save Project As")  # Laisser à l'utilisateur le choix du nom et de l'emplacement du projet
    path_project = path + r'\export.psx'
    doc.save(path_project)

    # Créer le dossier d'export des DEMs
    if not os.path.exists(path + "/" + "DEMs"):
        os.makedirs(path + "/" + "DEMs")  # Crée le fichier s'il n'existe pas

    if 'Session' in os.path.basename(os.path.normpath(path)):  # Si le dossier est un dossier session
        print(os.path.basename(os.path.normpath(path)))
        plotlist = os.listdir(path)
        for plot in plotlist:  # Traiter dans le même projet tout les dossiers plots de la session
            if plot.find("1") == 0:
                print(plot)
                workflow(path, plot)
    else:  # Si le dossier est un dossier plot
        print(os.path.basename(os.path.normpath(path)))
        workflow(path)

boucle(sys.argv[2])
