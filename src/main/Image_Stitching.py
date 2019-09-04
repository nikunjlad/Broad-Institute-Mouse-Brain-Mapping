import os
import numpy as np
import cv2


def get_image(cur_dir, dest_path, magnif, name, magnification):
    tile_map = dict()
    for images in magnif:
        # create a tile map dictionary where key is the tile co-ordinate in the tile grid, while value is the image path
        tile_map[
            (int(images.split('.')[0].split('-')[2]), int(images.split('.')[0].split('-')[1]))] = cur_dir + "/" + images
        # print('Tile map: ', tile_map)

    # finding out the tile grid dimension to be initialized
    tile_dimension = [max([int(ims.split('.')[0].split('-')[2]) for ims in magnif]) + 1,
                      max([int(ims.split('.')[0].split('-')[1]) for ims in magnif]) + 1]

    print('Tile Dimension: ', tile_dimension)

    # fitting tiles in the grid and writing out the image
    hor_stack = list()
    for col in range(tile_dimension[1]):
        vert_stack = list()
        for row in range(tile_dimension[0]):
            tile = cv2.imread(tile_map[(row, col)], cv2.IMREAD_COLOR)
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGBA)
            vert_stack.append(tile)
        hor_stack.append(np.vstack(vert_stack))

    image = np.hstack(hor_stack)
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(dest_path + "/" + name + "_" + magnification + ".png", bgr_img)
    print("Image " + dest_path + "/" + name + "_" + magnification + ".png" + " written successfully!")


def main():

    magnification = str(input("Enter the magnification level to stitch (1,2,3 or 4): "))
    code_dir = os.getcwd()  # get the current directory
    os.chdir('../data/stitched')
    stitch_dir = os.getcwd()
    os.chdir('../../data/prestitched')
    prestitched_dir = os.getcwd()
    print("Stitched dir: ", stitch_dir)
    print("Prestitched dir: ", prestitched_dir)
    print("Code dir: ", code_dir)

    classes = [folder for folder in os.listdir(prestitched_dir) if not folder.startswith(".")]
    print("Class names: ", classes)
    class_paths = [prestitched_dir + "/" + folder for folder in classes]
    print("Class paths: ", class_paths)

    for ind1, img_dir in enumerate(class_paths):
        os.chdir(img_dir)
        imgs = [folder for folder in os.listdir(os.getcwd()) if not folder.startswith(".")]
        imgs_dir = [img_dir + "/" + folder for folder in imgs]
        print("\n")
        print("Inside " + classes[ind1] + "folder")
        print("All images in this class: ", imgs)
        print("All the image paths: ", imgs_dir)
        print("\n")
        dest_path = stitch_dir + "/" + classes[ind1]
        print("Destination path: ", dest_path)
        print("\n")

        for ind2, image_path in enumerate(imgs_dir):
            os.chdir(image_path)
            cur_dir = os.getcwd()
            all_imgs = os.listdir()
            magnif = [im for im in all_imgs if im.split('-')[0] == str(magnification)]
            magnif.sort()
            get_image(cur_dir, dest_path, magnif, imgs[ind2], magnification)


if __name__ == "__main__":
    main()