
import torch
import torchvision.transforms.functional as TF
import random
def rotate_images(image1, image2,angle):
    rotated_image1 = TF.rotate(image1, angle)
    rotated_image2 = TF.rotate(image2, angle)
    return rotated_image1, rotated_image2
def transform_images(image1):
    # Randomly select an angle for rotation and affine
    angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    angle = random.choice(angles)

    # Set probability for vertical and horizontal flip
    p_vertical = 0.5
    p_horizontal = 0.5

    # Set parameters for affine transformation
    translate_range = [0, 0.5]  # for example, translating up to 50% of the image size
    translate = (random.uniform(*translate_range), random.uniform(*translate_range))
    shear_range = [-30, 30]  # shear degrees can be between -30 and 30
    shear = random.uniform(*shear_range)

    # Set hue adjustment parameter
    hue_range = [-0.1, 0.1]  # for example, hue changes can be between -0.1 and 0.1
    hue = random.uniform(*hue_range)

    # Rotate images
    image1 = TF.rotate(image1, angle)

    # Vertical flip
    if torch.rand(1) < p_vertical:
        image1 = TF.vflip(image1)

    # Horizontal flip
    if torch.rand(1) < p_horizontal:
        image1 = TF.hflip(image1)

    # Affine transformation
    image1 = TF.affine(image1, angle=angle, translate=translate, scale=1.0, shear=shear)

    # Color adjustment
    image1 = TF.adjust_hue(image1, hue)

    return image1


def transform_images_parameters(image1):
    # Randomly select an angle for rotation and affine
    angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    angle = random.choice(angles)

    # Set probability for vertical and horizontal flip
    p_vertical = 0.5
    p_horizontal = 0.5

    # Set parameters for affine transformation
    translate_range = [0, 0.5]  # for example, translating up to 50% of the image size
    translate = (random.uniform(*translate_range), random.uniform(*translate_range))
    shear_range = [-30, 30]  # shear degrees can be between -30 and 30
    shear = random.uniform(*shear_range)

    # Set hue adjustment parameter
    hue_range = [-0.1, 0.1]  # for example, hue changes can be between -0.1 and 0.1
    hue = random.uniform(*hue_range)

    # Rotate images
    image1 = TF.rotate(image1, angle)

    # Vertical flip
    vertical_flag=torch.rand(1)< p_vertical
    if vertical_flag:
        image1 = TF.vflip(image1)

    # Horizontal flip
    horizontal_flag=torch.rand(1) < p_horizontal
    if horizontal_flag:
        image1 = TF.hflip(image1)

    # Affine transformation
    image1 = TF.affine(image1, angle=angle, translate=translate, scale=1.0, shear=shear)

    # Color adjustment
    image1 = TF.adjust_hue(image1, hue)

    return image1,vertical_flag,horizontal_flag,angle


def fix_transform_images_parameters(image1,vertical_flag,horizontal_flag,angle):
    # Randomly select an angle for rotation and affine

    # Rotate images
    image1 = TF.rotate(image1, angle)

    # Vertical flip

    if vertical_flag:
        image1 = TF.vflip(image1)

    # Horizontal flip

    if horizontal_flag:
        image1 = TF.hflip(image1)


    return image1