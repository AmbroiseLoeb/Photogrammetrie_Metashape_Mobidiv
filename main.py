import subprocess

# test lancer metashape depuis python


def main():

    # Lancer Metashape
    subprocess.run([r'C:\Program Files\Agisoft\Metashape Pro\metashape.exe', '-r', r'test_script_metashape.py'])

    # Traiter les résultats


if __name__ == "__main__":
    main()
