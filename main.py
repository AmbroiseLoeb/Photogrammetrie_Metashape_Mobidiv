import subprocess
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# lancer metashape depuis python
def main():

    path_annee = r"C:\Users\U108-N806\Desktop\Comparaison mesures"
    # path_annee = '/home/loeb/Documents/Comparaison_mesures'

    fonction = "boucle"

    # Executer le script correspondant dans Metashape
    subprocess.run([r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'test_script_metashape.py', fonction] + [path_annee])

    # Récupérer les DEMs
    sessionlist = os.listdir(path_annee)
    for session in sessionlist:
        if session.find("Session") == 0:
            print(session)
            list_dems = os.listdir(path_annee + "/" + session + "/" + 'DEMs')
            for file in list_dems:
                print(file)

                dem = Image.open(path_annee + "/" + session + "/" + 'DEMs' + "/" + file)
                dem_array = np.array(dem)
                dem_array[dem_array <= 0] = np.nan
                dem_array = dem_array * 1000
                plt.figure() and plt.imshow(dem_array)



if __name__ == "__main__":
    main()
