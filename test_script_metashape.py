# importation librairies
import Metashape


def workflow(dossier):
    """traitement des images jusqu'au DEM (digital elevation model)"""

    liste_images = (r"\uplot_100_camera_1_1_RGB.jpg", r"\uplot_100_camera_2_1_RGB.jpg",
                    r"\uplot_100_camera_1_2_RGB.jpg", r"\uplot_100_camera_2_2_RGB.jpg",
                    r"\uplot_100_camera_3_1_RGB.jpg", r"\uplot_100_camera_3_2_RGB.jpg")
    for image in liste_images:
        path_image = str(dossier + image)
        chk.addPhotos(path_image)

    # creation reference distance (avec distance camera)
    chk.addScalebar(chk.cameras[0], chk.cameras[2])
    chk.scalebars[0].label = "dist_cam_ref"
    chk.scalebars[0].reference.distance = 0.06511302126767482
    chk.updateTransform()

    # alignement des photos
    chk.matchPhotos(downscale=1, generic_preselection=False, reference_preselection=False)
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

    # classification sol
    chk.transform.matrix = chk.transform.matrix
    chk.point_cloud.classifyGroundPoints(max_angle=40, max_distance=0.01, cell_size=0.7)
    # chk.point_cloud.setClassesFilter(Metashape.Ground)

    doc.save()
    # construction du modele numerique d'elévation (DEM)
    # DEM toutes classes
    chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.DisabledInterpolation)
    chk.elevation.label = "Elevation toutes classes"
    key_allclass = chk.elevation.key
    # DEM sol
    chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.Extrapolated, classes=Metashape.PointClass.Ground)
    chk.elevation.label = "Elevation sol"
    key_sol = chk.elevation.key
    # DEM finale
    chk.transformRaster(source_data=Metashape.ElevationData, asset=key_allclass, subtract=True, operand_asset=key_sol)
    chk.elevation.label = "DEM_finale"
    label_dem = chk.elevation.label
    # exportation du DEM
    doc.save()
    chk.exportRaster(r"C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\export" + '/' + label_dem + '_export.tif',
                     source_data=Metashape.ElevationData)


#  creation du doc
doc = Metashape.Document()

# enregistrement du projet
# path = Metashape.app.getSaveFileName("Save Project As")
path = r'C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\export\retest.psx'
try:
    doc.save(path)
except RuntimeError:
    Metashape.app.messageBox("Can't save project")
    print("Can't save project")

#  creation du chunk
chk = doc.chunk
if not chk:
    chk = doc.addChunk()

# importation des photos
path_dossier = r"C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\Session 2023-03-08 12-45-21\uplot_100_1"

workflow(path_dossier)
