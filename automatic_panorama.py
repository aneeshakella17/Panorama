import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import panorama
import sklearn
import harris
import os
from sklearn import preprocessing
from skimage.transform import resize
from skimage.feature import corner_harris, corner_peaks, peak_local_max
from random import *
import scipy;
import multiresolution_blend;

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_corners(image):

    height, width, n = np.shape(image);
    h, coords = harris.get_harris_corners(rgb2gray(image));
    pos_matrix = np.zeros(shape = (height, width));

    best_neighbors = np.array([-1, -1, -1]);

    for i in range (0, len(coords[0])):
        pos_y, pos_x = coords[0][i], coords[1][i];
        pos_matrix[pos_y][pos_x] = 1;


    for j in range (0, len(coords[0])):
        pos_y, pos_x = coords[0][j], coords[1][j];
        vector = find_best_radius(pos_y, pos_x, pos_matrix, h);
        best_neighbors = np.vstack((best_neighbors, vector));


    best_neighbors = sorted(best_neighbors, key = lambda x: x[0], reverse = True);
    new_coords = best_neighbors[0:250];
    new_coords = np.array(new_coords)

    y_coords = new_coords[:, 1];
    x_coords = new_coords[:, 2];


    return y_coords, x_coords;


def find_best_radius(pos_y, pos_x, pos_matrix, h):
    i = 1;
    while (True):
        for y in range(pos_y - i, pos_y + i):
            for x in range(pos_x - i,  pos_x + i):
                if(i == 50):
                    return float('inf'), pos_y, pos_x;
                try:
                    if (pos_matrix[y][x] == 1):
                        if(h[pos_y][pos_x] < 0.9 * h[y][x]):
                            return norm((y, x), (pos_y, pos_x)), pos_y, pos_x;
                except:
                    continue;

        i += 1;

    return None;

def feature_extract(y_points, x_points, im):
    regions = [];
    for i in range(0, len(y_points)):
        point_y, point_x = int(y_points[i]), int(x_points[i]);
        feature_region = im[point_y - 20: point_y + 20, point_x - 20: point_x + 20];
        if(len(feature_region) == 0):
            regions.append(None)
            continue;

        feature_region = np.reshape(feature_region, (1600, 3));
        new_feature_region = np.zeros((64, 3));

        for i in range(0, 1600, 25):
            index = int(i/25);
            new_feature_region[index] = feature_region[i];

        new_feature_region = sk.filters.gaussian(new_feature_region, 3);
        feature_region = new_feature_region;
        feature_region = feature_region[:] - np.mean(feature_region);
        feature_region = feature_region/np.std(feature_region);
        regions.append(feature_region);

    return regions;


def region_ssd(first_regions, second_regions):
    indices = [];

    for i in range(0, len(first_regions)):
        image = first_regions[i];

        if image is None:
            continue;

        best_index = -100;
        best = float('inf');
        second_best = float('inf');

        for j in range(0, len(second_regions)):
            region = second_regions[j];

            if region is None:
                continue;


            new_difference = np.sum((image - region) ** 2);


            if(new_difference < best):
                second_best = best;
                best = new_difference;
                best_index = j;


        if(best/second_best < 0.3):
            plt.imshow(first_regions[i]);
            plt.imshow(second_regions[best_index]);
            indices.append([i, best_index]);



    return indices;


def norm(point, point2):
    x1 = point[0];
    x2 = point2[0];
    y1 = point[1];
    y2 = point2[1];
    return np.math.sqrt(np.math.pow(x1 - x2, 2) + np.math.pow(y1 - y2, 2))


def ransac(im1_pts, im2_pts):
    trials = 0;
    total_inliers = [];

    im1_pts_copy = np.copy(im1_pts);
    im2_pts_copy = np.copy(im2_pts);


    unos = np.ones(shape=(len(im1_pts), 1));
    im1_pts_for_homo = np.hstack((im1_pts_copy, unos));
    im1_pts_for_homo = im1_pts_for_homo.T;

    tmp = np.copy(im1_pts_for_homo[0, :]);
    im1_pts_for_homo[0, :] = im1_pts_for_homo[1, :]
    im1_pts_for_homo[1, :] = tmp;

    im2_pts_copy = np.copy(im2_pts);
    im2_pts_for_homo = im2_pts_copy.T;
    tmp = np.copy(im2_pts_for_homo[0, :]);
    im2_pts_for_homo[0, :] = im2_pts_for_homo[1, :]
    im2_pts_for_homo[1, :] = tmp;

    while(trials < 10000):
        indices = np.random.choice(len(im1_pts), 4, replace = False);
        im1_sample = np.array([im1_pts[indices[0]][1], im1_pts[indices[0]][0]]);
        im2_sample = np.array([im2_pts[indices[0]][1], im2_pts[indices[0]][0]]);

        for i in range(1, len(indices)):
            im1_sample = np.vstack((im1_sample, [im1_pts[indices[i]][1], im1_pts[indices[i]][0]]));
            im2_sample = np.vstack((im2_sample, [im2_pts[indices[i]][1], im2_pts[indices[i]][0]]));

        H = panorama.computeH(im1_sample, im2_sample)

        inliers = [];
        SSD_sum = 0;

        for j in range(0, len(im2_pts)):
            new_coords = np.matmul(H, im1_pts_for_homo[:, j]);
            x, y, w = new_coords;
            answer = np.array([x/w, y/w]);
            SSD = np.sum((im2_pts_for_homo[:, j] - answer.T) ** 2);

            if(SSD < 11):
                SSD_sum += SSD;
                inliers.append([im1_pts_for_homo[:, j], im2_pts_for_homo[:, j]])


        trials += 1;
        total_inliers.append([inliers, H, SSD_sum] );


    return getH(total_inliers);


def getH(total_inliers):
    max_length = 0;
    index = 0;
    best_SSD = float('inf');
    my_inliers = [];
    bestH = [];

    for i in range(0, len(total_inliers)):
        inliers, H, SSD = total_inliers[i]
        if(len(inliers) >= max_length):
            if(len(inliers) == max_length and best_SSD < SSD ):
                    continue;
            else:
                max_length = len(inliers);
                bestSSD = SSD;
                bestH = H;
                index = i;
                my_inliers = inliers;

    #
    list_1, list_2 = my_inliers[0];
    list_1, list_2 = np.array(list_1[0:2]), np.array(list_2);

    for j in range(1, len(my_inliers)):
        add_1, add_2 = my_inliers[j]
        list_1 = np.vstack((list_1, add_1[0:2] ));
        list_2 = np.vstack((list_2, add_2));

    return bestH, list_1, list_2;


def create_automatic_panorama(im1, im2, gauss = False):
    if gauss:
        gauss1 = multiresolution_blend.mask_gaussian(im1, 6);
        best_corners_left = create_corners(gauss1[4]);
        gauss2 = multiresolution_blend.mask_gaussian(im2, 6);
        best_corners_right = create_corners(gauss2[4]);
    else:
        best_corners_left = create_corners(im1);
        best_corners_right = create_corners(im2);

        left_regions = feature_extract(best_corners_left[:][0], best_corners_left[:][1], im1);
        right_regions = feature_extract(best_corners_right[:][0], best_corners_right[:][1], im2);


    indices = region_ssd(left_regions, right_regions);

    left_index = indices[0][0];
    right_index = indices[0][1];


    new_left = np.array([best_corners_left[0][left_index], best_corners_left[1][left_index]]);
    new_right = np.array([best_corners_right[0][right_index], best_corners_right[1][right_index]]);


    for i in range(1, len(indices)):
        left_index = indices[i][0];
        right_index = indices[i][1];
        new_left = np.vstack((new_left, [best_corners_left[0][left_index], best_corners_left[1][left_index]]));
        new_right = np.vstack((new_right, [best_corners_right[0][right_index], best_corners_right[1][right_index]]));


    return ransac(new_left, new_right);


def auto_panorama(imgs, current_image):
    if(len(imgs) == 0):
        return;
    identity = np.array([ [1, 0, 0], [0, 1, 0] [0, 0, 1]])
    for j in range(0, len(imgs)):
        potential_image = imgs[j];
        H, im1_pts, im2_pts = create_automatic_panorama(current_image, potential_image);
        if(len(im1_pts) < 6):
            continue;
        else:
            if(im1_pts[0] < im2_pts[0]):
                H = create_automatic_panorama(current_image, current_image);
                base = panorama.warpImage(current_image, H);
                new_H, list_1, list_2 = create_automatic_panorama(potential_image, base);
                transformed_potential = panorama.warpImage(potential_image, new_H);
                return weighted_blend(base, transformed_potential);
            else:
                H = create_automatic_panorama(potential_image, potential_image);
                base = panorama.warpImage(potential_image, H);
                new_H, list_1, list_2 = create_automatic_panorama(current_image, base);
                transformed_potential = panorama.warpImage(current_image, new_H);
                return weighted_blend(base, transformed_potential);


def weighted_blend(im1, im2):
    height, width, n = np.shape(im1);
    new_im = np.zeros(shape = (height, width, n));
    for y in range(0, height):
        for x in range(0, (width)):
            overlap = (0.1 * width/2);
            if( x > (width/2 - overlap) and  x < width/2):
                a = 0.5 * (width/2)/(x);
                b = 1 - a;
            elif (x < (width/2 - overlap)):
                a = 1
                b = 0;
            else:
                a = 0;
                b = 1;

            new_im[y][x] = a *im1[y][x] + b * im2[y][x];
    return new_im;


def rotate_normalize(patch):
    avg_patch = rgb2gray(patch)
    ddx = np.gradient(avg_patch, axis = 0);
    ddy = np.gradient(avg_patch, axis = 1);
    angle = np.arctan(ddy[20][20]/ddx[20][20]);
    return scipy.misc.imrotate(patch, -1*angle);


