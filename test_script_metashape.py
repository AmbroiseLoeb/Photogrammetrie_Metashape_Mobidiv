# importation librairies
import Metashape

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
liste_images = (r"\uplot_100_camera_1_1_RGB.jpg", r"\uplot_100_camera_2_1_RGB.jpg",
                r"\uplot_100_camera_1_2_RGB.jpg", r"\uplot_100_camera_2_2_RGB.jpg")
for image in liste_images:
    path_image = str(path_dossier + image)
    chk.addPhotos(path_image)

# creation reference distance (avec distance camera)
chk.addScalebar(chk.cameras[0], chk.cameras[2])
chk.scalebars[0].label = "dist_cam_ref"
chk.scalebars[0].reference.distance = 0.06511302126767482
chk.updateTransform()

# alignement des photos
chk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
chk.alignCameras()

# construction du nuage de point
chk.buildDepthMaps(downscale=2, filter_mode=Metashape.NoFiltering, reuse_depth=True)
chk.buildPointCloud(point_colors=True, point_confidence=True, keep_depth=True, max_neighbors=16)

# filtre par confiance
chk.point_cloud.setConfidenceFilter(0, 1)
all_points_classes = list(range(128))
chk.point_cloud.removePoints(
    all_points_classes)  # removes all active points of the point cloud, i.e. removing all low-confidence points
chk.point_cloud.resetFilters()  # resetting filter, so high-confidence points are now active

'''
# filtre par couleur
gray_color = [84, 66, 52]
tolerance = 25
chk.point_cloud.selectPointsByColor(gray_color, tolerance, channels='RGB')
chk.point_cloud.assignClassToSelection(7)
'''

doc.save()
# construction du modele numerique d'elévation (DEM)
chk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.DisabledInterpolation, classes=[0])

# exportation du DEM
doc.save()
chk.exportRaster(r"C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\export" + '/DEM_export.tif',
                 source_data=Metashape.ElevationData)
