import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import filters
import matplotlib.pyplot as plt


def multiblend(source, transfer, mask):
    images = [];
    size = len(mask);
    for q in range(0, size):
        m = mask[q];
        length = len(m);
        width = len(m[0]);
        new_image = np.zeros(shape = (length, width, 3));
        for c in range(0, 3):
            s = source[c][q];
            t = transfer[c][q];
            for i in range(0, length):
                for j in range(0, width):
                    new_image[i][j][c] = m[i][j] * t[i][j] + (1 - m[i][j])*s[i][j];
        images.append(new_image);
    return images;

def gaussian(im, N):
    r = []
    g = []
    b = []
    for i in range(0, N):
        r.append(sk.filters.gaussian(im[:, :, 0], 3 * i));
        g.append(sk.filters.gaussian(im[:, :, 1], 3 * i));
        b.append(sk.filters.gaussian(im[:, :, 2], 3 * i));
    return [r, g, b];

def mask_gaussian(im, N):
    arr = [];
    for i in range(0, N - 1):
        arr.append(sk.filters.gaussian(im, 3 * i))
    return arr;

def laplacian(arr):
    r = arr[0];
    g = arr[1];
    b = arr[2];
    new_r = [];
    new_g = [];
    new_b = [];
    for i in range(0, len(r) - 1):
        new_r.append(r[i] - r[i + 1]);
        new_g.append(g[i] - g[i + 1]);
        new_b.append(b[i] - b[i + 1]);
    return [new_r, new_g, new_b];


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def construct_blend(earth, neptune, mask):
    mask = rgb2gray(mask);
    mask[mask > 0.5] = 1;
    mask[mask < 0.5] = 0;
    # plt.imshow(mask);
    # plt.show();
    earth_gaussian = gaussian(earth, 6);
    neptune_gaussian = gaussian(neptune, 6);
    mask_gauss = mask_gaussian(mask, 6);

    earth_laplacian = laplacian(earth_gaussian);
    neptune_laplacian = laplacian(neptune_gaussian);
    gauss_final = multiblend(earth_gaussian, neptune_gaussian, mask_gauss);

    laplace_final = multiblend(earth_laplacian, neptune_laplacian, mask_gauss);

    im_final = gauss_final[3] + laplace_final[2] + laplace_final[1] + laplace_final[0]

    red_channel = im_final[:, :, 0];
    red_channel[red_channel < 0] = 0;
    red_channel[red_channel > 1] = 1;
    im_final[:, :, 0] = red_channel;

    green_channel = im_final[:, :, 1];
    green_channel[green_channel < 0] = 0;
    green_channel[green_channel > 1] = 1;
    im_final[:, :, 1] = green_channel;

    blue_channel = im_final[:, :, 2];
    blue_channel[blue_channel < 0] = 0;
    blue_channel[blue_channel > 1] = 1;
    im_final[:, :, 2] = blue_channel;

    skio.imshow(im_final);
    skio.show();
    return im_final;


