import matplotlib.pyplot as plt
import numpy as np
import os

def computeH(im1_pts, im2_pts):
    x = im1_pts[0][0];
    y = im1_pts[0][1];
    xprime = im2_pts[0][0];
    yprime = im2_pts[0][1];
    im1_matrix = np.array([x, y, 1, 0, 0, 0, -1*x*xprime, -1*y*xprime]);
    add = np.array([0, 0, 0, x, y, 1, (-1 * x * yprime), (-1 * y * yprime)]);
    im1_matrix = np.vstack((im1_matrix, add));
    im2_matrix = np.array([xprime, yprime]);
    im2_matrix = np.array([[im2_pts[0][0]], [im2_pts[0][1]]]);
    for i in range(1, len(im1_pts)):
        x = im1_pts[i][0];
        y = im1_pts[i][1];
        xprime = im2_pts[i][0];
        yprime = im2_pts[i][1];
        add_1= np.array([x, y, 1, 0, 0, 0, (-1 * x * xprime), (-1 * y * xprime)]);
        add_2 = np.array([0, 0, 0, x, y, 1, (-1 * x * yprime), (-1 * y * yprime)]);
        im1_matrix = np.vstack((im1_matrix, add_1));
        im1_matrix = np.vstack((im1_matrix, add_2));
        b = np.array([[im2_pts[i][0]], [im2_pts[i][1]]]);
        im2_matrix = np.vstack((im2_matrix, b));

    homography = np.linalg.lstsq(im1_matrix, im2_matrix)[0];
    homography = np.vstack((homography, [1]))
    homography = np.reshape(homography, (3, 3));
    return homography;

def cylinder_projection(f, im):
    height, width, n = np.shape(im);
    yc, xc = height/2, width/2;
    new_im = np.zeros(shape = (height, width, n));
    for y in range(0, height):
        for x in range(0, width):
            theta = (x - xc)/f;
            h = (y - yc)/f;
            xbar = np.math.sin(theta);
            ybar = h;
            zbar = np.math.cos(theta);
            new_x = f * (xbar/zbar) + xc;
            new_y = f * (ybar/zbar) + yc;
            new_x, new_y = int(new_x), int(new_y)
            if(new_y < len(im) and new_x < len(im[0]) and new_y >= 0 and new_x >= 0):
                new_im[y][x] = im[new_y][new_x];




    return new_im;

def spherical_projection(f, im):
    height, width, n = np.shape(im);
    yc, xc = height/2, width/2;
    new_im = np.zeros(shape = (height, int(width), n));
    for y in range(0, height):
        for x in range(0, width):
            theta = (x - xc)/f;
            phi = (y - yc)/f;
            xbar = np.math.sin(theta) * np.math.cos(phi);
            ybar = np.math.sin(phi);
            zbar = np.math.cos(theta) * np.math.cos(phi);
            new_x = f * (xbar/zbar) + xc;
            new_y = f * (ybar/zbar) + yc;
            new_x, new_y = int(new_x), int(new_y)
            if(new_y < len(im) and new_x < len(im[0]) and new_y >= 0 and new_x >= 0):
                new_im[y][x] = im[new_y][new_x];

    return new_im;


def warpImage(im, H):
    height, width, n = np.shape(im);
    new_im = np.zeros(shape = (height, int(width * 1.5), n));
    for y in range(0, height):
        for x in range(0, int(width * 1.5)):
            vector = [x, y, 1];
            new_coords = np.matmul(np.linalg.inv(H), vector);
            new_x, new_y, rand = new_coords
            new_x, new_y = int(new_x/rand), int(new_y/rand)
            if(new_y < len(im) and new_x < len(im[0]) and new_y >= 0 and new_x >= 0):
                try:
                    new_im[y][x] = im[new_y][new_x];
                except:
                    continue;
    return new_im;

def selectPoints(im1, im2, n_value):
    plt.imshow(im1);
    im1_pts = plt.ginput(n = n_value, timeout = 0, show_clicks = True);
    np.savetxt('im1_pts.txt', im1_pts);
    plt.imshow(im2);
    im2_pts = plt.ginput(n = n_value, timeout = 0, show_clicks = True);
    np.savetxt('im2_pts.txt', im2_pts);
    return im1_pts, im2_pts;

def toRGB(im):
    height, length, n = np.shape(im);
    if(n == 3):
        return im;
    new_im = np.zeros(shape = (height, length, 3));
    for y in range(0, height):
        for x in range(0, length):
            new_im[y][x][0] = im[y][x][0] * im[y][x][3];
            new_im[y][x][1] = im[y][x][1] * im[y][x][3];
            new_im[y][x][2] = im[y][x][2] * im[y][x][3];
    return new_im;

