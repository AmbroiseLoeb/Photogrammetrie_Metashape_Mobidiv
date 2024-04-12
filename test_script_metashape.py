# importation librairies
import Metashape
import os
import sys


def boucle(path):
    def workflow(path_dossier, dossier):
        """traitement des images jusqu'au DEM (digital elevation model)"""

        chk = doc.addChunk()

        liste_images = []
        for filename in os.listdir(path_dossier + '/' + dossier):
            # vérifier si le fichier se termine par "RGB.jpg"
            if filename.endswith("RGB.jpg") and "camera_3" not in filename:
                # ajouter le nom du fichier à la liste d'images
                liste_images.append(filename)

        for image in liste_images:
            path_image = str(path_dossier + '/' + dossier + '/' + image)
            chk.addPhotos(path_image)

        for camera in chk.cameras:
            path_mask = str(path_dossier + '/' + dossier + '/' + camera.label.split('RGB')[0] + 'MASK.jpg')
            chk.generateMasks(path=path_mask, masking_mode=Metashape.MaskingModeFile,
                              mask_operation=Metashape.MaskOperationReplacement, cameras=camera, tolerance=10)

        # creation reference distance (avec distance camera)
        chk.addScalebar(chk.cameras[0], chk.cameras[2])
        chk.scalebars[0].label = "dist_cam_ref"
        chk.scalebars[0].reference.distance = 0.06511302126767482
        chk.updateTransform()

        # alignement des photos
        # chk.remove(chk.cameras[5]) and chk.remove(chk.cameras[4])
        chk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True)
        chk.alignCameras()

        # construction du nuage de point
        chk.buildDepthMaps(downscale=2, filter_mode=Metashape.NoFiltering, reuse_depth=True)
        chk.buildPointCloud(point_colors=True, point_confidence=True, keep_depth=True, max_neighbors=16)

        '''
        # filtre par confiance
        chk.point_cloud.setConfidenceFilter(0, 1)
        all_points_classes = list(range(128))
        chk.point_cloud.removePoints(all_points_classes)  # removes all active points of the point cloud, i.e. removing all low-confidence points
        chk.point_cloud.resetFilters()  # resetting filter, so high-confidence points are now active

        # filtre par couleur
        gray_color = [84, 66, 52]
        tolerance = 25
        chk.point_cloud.selectPointsByColor(gray_color, tolerance, channels='RGB')
        chk.point_cloud.assignClassToSelection(7)
        '''

        '''
        # classification sol
        chk.point_cloud.classifyGroundPoints(max_angle=30, max_distance=0.01, max_terrain_slope=10, cell_size=0.5)
        # chk.point_cloud.setClassesFilter(Metashape.Ground)
        '''

        # position de la région
        chk.transform.matrix = chk.transform.matrix
        '''
        camera = chk.cameras[1]
        T = chk.transform.matrix
        m = chk.crs.localframe(T.mulp(camera.center))
        R = m * T * camera.transform * Metashape.Matrix().Diag([1, 1, -1, -1])
        chk.transform.matrix = R
        '''
        doc.save()
        # construction du modele numerique d'elévation (DEM)
        # DEM toutes classes
        chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.DisabledInterpolation)
        chk.elevation.label = "Elevation toutes classes"
        key_allclass = chk.elevation.key
        """
        # DEM sol
        chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.Extrapolated, classes=Metashape.PointClass.Ground)
        chk.elevation.label = "Elevation sol"
        key_sol = chk.elevation.key
        # DEM finale
        chk.transformRaster(source_data=Metashape.ElevationData, asset=key_allclass, subtract=True, operand_asset=key_sol)
        chk.elevation.label = "DEM_finale"
        """
        label_dem = chk.elevation.label

        # exportation du DEM
        doc.save()
        chk.exportRaster(path_dossier + '/' + "DEMs" '/' + dossier + label_dem + '_export3.tif',
                         source_data=Metashape.ElevationData)

    #  creation du doc
    doc = Metashape.Document()

    # enregistrement du projet
    # path = Metashape.app.getSaveFileName("Save Project As")
    path_project = path + r'\Export3.psx'
    try:
        doc.save(path_project)
    except RuntimeError:
        Metashape.app.messageBox("Can't save project")
        print("Can't save project")

    # importation des photos
    # PATH = r"C:\Users\U108-N806\Desktop\Comparaison mesures"
    sessionlist = os.listdir(path)
    for session in sessionlist:
        if session.find("Session") == 0:
            print(session)
            plotlist = os.listdir(path + "/" + session)
            if not os.path.exists(path + "/" + session + "/" + "DEMs"):
                # Crée le fichier s'il n'existe pas
                os.makedirs(path + "/" + session + "/" + "DEMs")
            for plot in plotlist:
                if plot.find("uplot_7_1") == 0:
                    print(plot)
                    path_session = path + "/" + session
                    print(path_session)
                    workflow(path_session, plot)


boucle(sys.argv[2])
