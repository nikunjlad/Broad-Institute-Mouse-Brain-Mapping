import os
from distutils.dir_util import copy_tree


def main():
    code_dir = os.getcwd()  # get the current directory
    os.chdir('../data/prestitched')
    prestitch_dir = os.getcwd()
    os.chdir('../../data/extracted')
    extracted_dir = os.getcwd()
    os.chdir('../../data/stitched')
    stitched_dir = os.getcwd()

    print("Extracted dir: ", extracted_dir)
    print("Stitched dir: ", stitched_dir)
    print("Prestitched dir: ", prestitch_dir)

    classes = [folder for folder in os.listdir(extracted_dir) if not folder.startswith(".")]
    print(classes)
    extract_paths = [extracted_dir + "/" + folder for folder in classes]
    prestitch_paths = [prestitch_dir + "/" + folder for folder in classes]
    print(extract_paths)
    print(prestitch_paths)

    coronal_dirs = ['001-ache', '001-nissl', '001-parv', '001-smi32', '002-nissl']
    horizontal_dirs = ['001-nissl', 'C57m1-highres', 'C57m2', 'cfos']
    sagittal_dirs = ['sag-highres']

    cor_path = "/HBP2/mus.musculus/cor/"
    hor_dict = {
        "001-nissl": "/HBP2/mus.musculus/hor/001-nissl",
        "C57m1-highres": "/HBP2/mus.musculus/C57m1/C57m1-highres",
        "C57m2": "/HBP2/mus.musculus/C57m2/C57m2",
        "cfos": "/HBP2/mus.musculus/C57m1/cfos"
    }
    sag_path = "/HBP2/mus.musculus/sag/"
    tiles = ['TileGroup0', 'TileGroup1']

    sag_images = list()
    hor_images = list()
    cor_images = list()

    for img_dir in extract_paths:
        os.chdir(img_dir)

        # use this directory list to store moved images
        imgs = [folder for folder in os.listdir(os.getcwd()) if not folder.startswith(".")]
        imgs_dir = [img_dir + "/" + folder for folder in imgs]
        print("To be moved here: ", imgs)
        print("\n")

        for ind1, im in enumerate(imgs_dir):

            print('Image_name: ', im)
            if "sagittal" in im:
                dest_path = prestitch_dir + "/sagittal/" + imgs[ind1]
                print("Destination: ", dest_path)
                os.mkdir(dest_path)
                sag_images.append(dest_path)
                for ind, s in enumerate(sagittal_dirs):
                    if s in im:
                        os.chdir(im + sag_path + s)
                        os.chdir(os.getcwd() + "/" + os.listdir(os.getcwd())[0])
                        before_move_dir = os.getcwd()
                        for tile in tiles:
                            os.chdir(before_move_dir + "/" + tile)
                            copy_tree(os.getcwd(), dest_path)
            elif "horizontal" in im:
                dest_path = prestitch_dir + "/horizontal/" + imgs[ind1]
                print("Destination: ", dest_path)
                os.mkdir(dest_path)
                hor_images.append(dest_path)
                for ind, s in enumerate(horizontal_dirs):
                    if s in im:
                        os.chdir(im + hor_dict[s])
                        os.chdir(os.getcwd() + "/" + os.listdir(os.getcwd())[0])
                        before_move_dir = os.getcwd()
                        for tile in tiles:
                            os.chdir(before_move_dir + "/" + tile)
                            copy_tree(os.getcwd(), dest_path)
            elif "coronal" in im:
                dest_path = prestitch_dir + "/coronal/" + imgs[ind1]
                print("Destination: ", dest_path)
                os.mkdir(dest_path)
                cor_images.append(dest_path)
                for ind, s in enumerate(coronal_dirs):
                    if s in im:
                        os.chdir(im + cor_path + s)
                        os.chdir(os.getcwd() + "/" + os.listdir(os.getcwd())[0])
                        before_move_dir = os.getcwd()
                        for tile in tiles:
                            os.chdir(before_move_dir + "/" + tile)
                            copy_tree(os.getcwd(), dest_path)


if __name__ == "__main__":
    main()
