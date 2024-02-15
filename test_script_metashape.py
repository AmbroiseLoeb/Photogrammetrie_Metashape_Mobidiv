# importation librairies
import Metashape

#  création du chunk
doc = Metashape.app.document
chk =doc.chunk
if not chk:
		chk =doc.addChunk()

# importation des photos
path_dossier = r"C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\Session 2023-03-08 12-45-21\uplot_100_1" 
liste_images = (r"\uplot_100_camera_1_1_RGB.jpg" , r"\uplot_100_camera_2_1_RGB.jpg", r"\uplot_100_camera_1_2_RGB.jpg", r"\uplot_100_camera_2_2_RGB.jpg")
for image in liste_images :
    path_image = str(path_dossier + image)
    chk.addPhotos(path_image)

# alignement des photos
chk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
chk.alignCameras()

# construction du nuage de point
chk.buildDepthMaps(downscale=2, filter_mode=Metashape.NoFiltering, reuse_depth=True)
chk.buildPointCloud(point_colors=True, point_confidence=True,keep_depth=True, max_neighbors=16)

# construction du model 3D
chk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)

# avec texture
chk.buildUV(mapping_mode=Metashape.GenericMapping)
chk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)

# enregistrement du projet metashape
doc.save(r"C:\Users\U108-N806\Desktop\Literal_mobidiv_2023\test3.psx")

