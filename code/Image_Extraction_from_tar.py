import os, tarfile


def extract_images_from_tar(brain_views, extraction_dir, classes):
    # replace this function later with recurssion
    # loop over coronal, sagittal and horizontal
    for ind, view in enumerate(brain_views):
        os.chdir(view)
        view_resol = [folder for folder in os.listdir(os.getcwd()) if not folder.startswith(".")]
        view_resol_paths = [view + "/" + folder for folder in view_resol]

        # loop over ache, nissl, parv, etc
        for ind1, img_dir in enumerate(view_resol_paths):
            os.chdir(img_dir)
            imgs = [folder for folder in os.listdir(os.getcwd()) if not folder.startswith(".")]
            extract_paths = [extraction_dir + "/" + classes[ind] + "/" + view_resol[ind1] + "_" + i for i in imgs]
            img = [img_dir + "/" + folder for folder in imgs]
            # loop over internal directories
            for ind2, dirs in enumerate(img):
                os.chdir(dirs)
                dirs2 = [dirs + "/" + folder for folder in os.listdir(os.getcwd()) if not folder.startswith(".")]
                tar_file = [file for file in dirs2 if file.endswith(".tar")][0]
                tar = tarfile.open(tar_file)
                tar.extractall(extract_paths[ind2])


def main():
    code_dir = os.getcwd()  # get the current directory
    os.chdir('../data/extracted')
    extraction_dir = os.getcwd()
    os.chdir('../../data/unextracted')  # traverse to the data directory
    data_dir = os.getcwd()  # get the path of the data directory
    print("Code dir: ", code_dir)
    print("Data dir: ", data_dir)
    print("Extraction dir: ", extraction_dir)

    # acquire list of folders in the given directory ignoring hidden files
    classes = [folder for folder in os.listdir(data_dir) if not folder.startswith(".")]
    print(classes)
    brain_views = [data_dir + "/" + folder for folder in classes]

    extract_images_from_tar(brain_views, extraction_dir, classes)


if __name__ == "__main__":
    main()
