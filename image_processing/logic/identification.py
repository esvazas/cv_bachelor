import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import random
import skimage.io
import skimage.transform
import skimage.filters
import skimage.feature
import skimage.morphology
import scipy.ndimage.morphology
import skimage.measure
import skimage.color
import sklearn.cluster
import collections
import sklearn.metrics
import matplotlib.cm
import gap_statistic
import multiprocessing
import functools
import errno
import warnings
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import pickle


def background_entropy(img):
    """
        Compute image parameters: mean, std
        Parameters
        ----------
        img : np.array(2D) - 2D array representing image

        Returns
        -------
        int : average of the cropped images
        int : std of the cropped images
        """

    #Crop Bounds
    y, x = img.shape
    zone = int(min(y,x)/10)

    startx = 0
    starty = 0
    image1 = img[starty:zone, startx:x]

    startx = 0
    starty = zone
    image2 = img[starty:y - zone, startx:zone]

    startx = 0
    starty = y - zone
    image3 = img[starty:y, startx:x]

    startx = x - zone
    starty = zone
    image4 = img[starty: y - zone, startx: x]

    mean = (np.mean(image1) + np.mean(image2) + np.mean(image3) + np.mean(image4))/4
    std = (np.std(image1) + np.std(image2) + np.std(image3) + np.std(image4))/4

    return mean,std

def process_image(path_save_pics, site_nr, pic, channel='gray eye', noise_filter='gaussian', edge_detection='sobel', noise_filter2 = 'gaussian',
                  thresholding='otsu', closing='closing', fill_holes='fill holes',
                  filter_params=[None, 5, 5, 5, 5, 2.1, None, 0],plot=False):
    """
        Main function to process image

        Parameters
        ----------
        path_save_pics   : str        - path to location where processed images will be saved
        site_nr          : int        - index of LIDT image in a dataset
        pic :            : np.array() - 3D array representing image
        channel          : str        - possible values: 'r','g','b','gray eye','gray equal','None'
        noise_filter     : str        - possible values: 'gaussian','median','mean','None'
        edge_detection   : str        - possible values: 'sobel','canny','entropy','None'
        noise_filter2    : str        - possible values: 'gaussian','median','mean','None'
        thresholding     : str        - possible values: 'otsu', 'yen', 'mean', 'minimum', 'local otsu', 'None'
        closing          : str        - possible values: 'closing','None'
        fill_holes       : str        - possible values: 'fill holes', 'None'

        Returns
        -------
        np.array() : processed image
        """

    titles_list = ['original']
    pics_list = [pic]

    if channel != None:

        if channel == 'rgb r':
            pic_channels = pic[:, :, 0]
        elif channel == 'rgb g':
            pic_channels = pic[:, :, 1]
        elif channel == 'rgb b':
            pic_channels = pic[:, :, 2]
        elif channel == 'hsv h':
            pic_channels = skimage.color.rgb2hsv(pic)[:, :, 0]
        elif channel == 'hsv s':
            pic_channels = skimage.color.rgb2hsv(pic)[:, :, 1]
        elif channel == 'hsv v':
            pic_channels = skimage.color.rgb2hsv(pic)[:, :, 2]
        elif channel == 'gray eye':
            pic_channels = skimage.color.rgb2gray(pic)
        elif channel == 'gray equal':
            pic_channels = np.average(pic, axis=2)
        elif channel == 'normalised':
            normalized = (pic.astype(float) / np.mean(pic, axis=(0, 1)))
            pic_channels = (np.min(normalized, axis=-1) * 200).astype(np.uint8)
        elif channel == 'original':
            pic_channels = pic
        else:
            pic_channels = np.average(pic, axis=2)

        titles_list.append("channel")
        pics_list.append(pic_channels)

    if noise_filter != None:

        if noise_filter == 'gaussian':
            noise = skimage.filters.gaussian(pic_channels, sigma=filter_params[1], mode= 'nearest')
        elif noise_filter == 'median':
            noise = skimage.filters.median(skimage.img_as_ubyte(pic_channels), selem=skimage.morphology.disk(int((filter_params[1]))))
        elif noise_filter == 'mean':
            noise = skimage.filters.rank.mean(skimage.img_as_float(pic_channels/pic_channels.max()), selem=skimage.morphology.disk(int(filter_params[1])))
        else:
            noise = pic_channels

        titles_list.append("noiseFilter" )
        pics_list.append(noise)

    if edge_detection != None:

        if edge_detection == 'sobel':
            edge = skimage.filters.sobel(noise, mask=None)
        elif edge_detection == 'canny':
            edge = skimage.feature.canny(skimage.img_as_ubyte(noise), sigma=float(filter_params[2]))
        elif edge_detection == 'entropy':
            edge = skimage.filters.rank.entropy(skimage.img_as_float(noise/255), selem=skimage.morphology.disk(int(filter_params[2])))
        elif edge_detection == 'background':

            pic = (noise * 255).astype(np.uint8)
            print("filter_params[6]", filter_params[6])
            dalis = pic[::10, ::10, :]
            subset = ((dalis - np.median(dalis, axis=[0, 1])) * 1.2 + np.median(dalis, axis=[0, 1])).astype(np.uint8)

            data_background = pd.DataFrame(
                ({'R': subset[:, :, 0].flatten(), 'G': subset[:, :, 1].flatten(), 'B': subset[:, :, 2].flatten()}))
            data_all = pd.DataFrame(
                ({'R': pic[:, :, 0].flatten(), 'G': pic[:, :, 1].flatten(), 'B': pic[:, :, 2].flatten()}))
            print("Min_samples", dalis.size, int(0.3*dalis.size))
            klasifikatorius = sklearn.cluster.DBSCAN(
                eps=float(filter_params[2]), min_samples=int(0.3*dalis.size), metric='euclidean').fit(data_background)


            data_background['Labels'] = klasifikatorius.labels_
            print(data_background['Labels'].value_counts())

            if len(np.unique(data_background['Labels'])) != 1:
                data_background = data_background[data_background['Labels'] != -1]  # Delete points classified as outliers
                hull = scipy.spatial.Delaunay(data_background[['R', 'G', 'B']].values)
                outside = hull.find_simplex(data_all[['R', 'G', 'B']].values) < 0
                edge = outside.reshape(pic.shape[0], pic.shape[1])
            else:
                print("Bad DBSCAN cloud clustering")
                edge = np.zeros((pic.shape[0], pic.shape[1]))

        else:
            edge = noise

        titles_list.append("edgeDetection")
        pics_list.append(edge)

    if noise_filter2 != None:

        if noise_filter2 == 'gaussian':
            noise2 = skimage.filters.gaussian(edge, sigma=float(filter_params[3]))
        elif noise_filter2 == 'median':
            noise2 = skimage.filters.median(skimage.img_as_ubyte(edge), mask=skimage.morphology.disk(int(filter_params[3])))
        elif noise_filter2 == 'mean':
            noise2 = skimage.filters.rank.mean(skimage.img_as_ubyte(edge/edge.max()), selem=skimage.morphology.disk(int(filter_params[3])))
        else:
            noise2 = edge

    titles_list.append("noiseFilter2")
    pics_list.append(noise2)


    if thresholding != None:
        # http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html

        if len(np.unique(noise2)) == 1:
            thresh = noise2
        else:

            if thresholding == 'otsu >':
                tre = skimage.filters.threshold_otsu(skimage.color.rgb2gray(noise2.astype(float)))
                thresh = noise2 >= tre
            elif thresholding == 'otsu <':
                tre = skimage.filters.threshold_otsu(skimage.color.rgb2gray(noise2.astype(float)))
                thresh = noise2 <= tre
            elif thresholding == 'yen >':
                tre = skimage.filters.threshold_yen(noise2, nbins=1256)
                thresh = noise2 > tre
            elif thresholding == 'yen <':
                tre = skimage.filters.threshold_yen(noise2, nbins=1256)
                thresh = noise2 < tre

            elif thresholding == 'manual >':
                if filter_params[7]:
                    edge_mean,edge_std = background_entropy(edge)
                    tre = edge_mean - float(filter_params[5]) * edge_std
                    thresh = noise2 >= tre
                else:
                    edge_mean,edge_std = background_entropy(edge)
                    tre = edge_mean + float(filter_params[5]) * edge_std
                    thresh = noise2 >= tre

            elif thresholding == 'manual <':
                if filter_params[7]:
                    edge_mean,edge_std = background_entropy(edge)
                    tre = edge_mean - float(filter_params[5]) * edge_std
                    thresh = noise2 <= tre
                else:
                    edge_mean,edge_std = background_entropy(edge)
                    tre = edge_mean + float(filter_params[5]) * edge_std
                    thresh = noise2 <= tre

            elif thresholding == 'mean >':
                tre = skimage.filters.threshold_mean(noise2)
                thresh = noise2 >= tre
            elif thresholding == 'mean <':
                tre = skimage.filters.threshold_mean(noise2)
                thresh = noise2 <= tre
            elif thresholding == 'minimum >':
                tre = skimage.filters.threshold_minimum(noise2, nbins=256, max_iter=10000)
                thresh = noise2 >= tre
            elif thresholding == 'minimum <':
                tre = skimage.filters.threshold_minimum(noise2, nbins=256, max_iter=10000)
                thresh = noise2 <= tre
            else:
                thresh = noise2

        titles_list.append("thresholding")
        pics_list.append(thresh)

    if closing != None:

        if closing == 'closing':
            bin_closing = skimage.morphology.binary_closing(thresh, selem=skimage.morphology.disk(int(filter_params[4])))

        else:
            bin_closing = thresh

        titles_list.append("binClosing")
        pics_list.append(bin_closing)

    if fill_holes != None:

        if fill_holes == 'fill holes':
            bin_fill = scipy.ndimage.morphology.binary_fill_holes(bin_closing).astype(int)

        else:
            bin_fill = bin_closing

        titles_list.append("fillHoles")
        pics_list.append(bin_fill)

        if plot:
            fig, axes = plt.subplots(2, 4, figsize=(6, 6))
            ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flatten()

            ax0.set_title(titles_list[0])
            ax0.axis('off')
            img0 = ax0.imshow(pics_list[0])
            ax1.set_title(titles_list[1])
            ax1.axis('off')
            img1 = ax1.imshow(pics_list[1])
            ax2.set_title(titles_list[2])
            ax2.axis('off')
            img2 = ax2.imshow(pics_list[2])
            ax3.set_title(titles_list[3])
            ax3.axis('off')
            img3 = ax3.imshow(pics_list[3])

            if edge_detection != 'background':
                fig.colorbar(img3, ax=ax3)

            ax4.set_title(titles_list[4])
            ax4.axis('off')
            img4 = ax4.imshow(pics_list[4])

            if edge_detection != 'background':
                fig.colorbar(img4, ax=ax4)

            ax5.set_title(titles_list[5])
            ax5.axis('off')
            ax5.imshow(pics_list[5])
            ax6.set_title(titles_list[6])
            ax6.axis('off')
            ax6.imshow(pics_list[6])
            ax7.set_title(titles_list[7])
            ax7.axis('off')
            ax7.imshow(pics_list[7])
            plt.axis('off')
            plt.suptitle("Site: %s" % str(site_nr))
            plt.show(block=False)

    return bin_fill


def make_circle(points):
    """
        Make circle over the figure which encloses all the given points.
        Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.

        Parameters
        ----------
        points : list - a sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].

        Returns
        -------
        np.array() : a triple of floats representing a circle.
    """

    # Initially: No boundary points known
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


def _make_circle_one_point(points, p):
    """
        Make circle over the figure if one boundary point is known.

        Parameters
        ----------
        points : list - a sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        p      : list - contains two coordinates of the know boundary point

        Returns
        -------
        np.array() : a triple of floats representing a circle.
    """

    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


def _make_circle_two_points(points, p, q):
    """
        Make circle over the figure if two boundary points are known.

        Parameters
        ----------
        points : list - a sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        p      : list - contains two coordinates of the know boundary point
        q      : list - contains two coordinates of the know boundary point

        Returns
        -------
        np.array() : a triple of floats representing a circle.
    """

    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
                        left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy,
                                                                                                    left[0],
                                                                                                    left[1])):
            left = c
        elif cross < 0.0 and (
                        right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy,
                                                                                                     right[0],
                                                                                                     right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_circumcircle(p0, p1, p2):
    """
        Implementaiton of circumscribed circle algorithm from Wikipedia.

        Parameters
        ----------
        p0      : list - pair of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        p1      : list - pair of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        p2      : list - pair of floats or ints, e.g. [(0,5), (3.1,-2.7)].

        Returns
        -------
        tuple : contains center coordinates and a radius required.
    """
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox;
    ay -= oy
    bx -= ox;
    by -= oy
    cx -= ox;
    cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
        ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
        bx - ax)) / d
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c, p):
    """
        Check if the provided point p is in circle

        Parameters
        ----------
        c      : np.array - contains coordinates of the circle and a radius
        p      : np.array - contains two coordinates of a point

        Returns
        -------
        np.bool : True if point is in circle
    """
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def Three_D(masked_image, centers, plot):
    """
        3D plot function

        Parameters
        ----------
        masked_image  : np.array - image array
        centers       : np.array - centers of a cluster
        plot          : bool     - to plot?

    """
    # define number of points to visualize
    points_to_visualise = 20000

    r = masked_image[:, :, 0].flatten() *255
    g = masked_image[:, :, 1].flatten() *255
    b = masked_image[:, :, 2].flatten() *255

    rgb = [(i, j, k) for i, j, k in zip(r, g, b)]
    rgb2 = list(filter(lambda x: x != (0, 0, 0), rgb))  # loosing (0,0,0) elements

    # take random points for better visualisation
    if len(rgb2) >= points_to_visualise:
        ind = np.random.choice(range(len(rgb2)), points_to_visualise)
        rgb2 = [rgb2[i] for i in ind]
        r, g, b = zip(*rgb2)

    else:
        r, g, b = zip(*rgb2)

    if plot == True:

        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.scatter(r, g, b, marker='o', c=np.array(rgb2) / 255, alpha=0.3, s=4)

        for j in list(range(0, len(centers), 1)):  # Plot the centers of clusters
            ax5.scatter(centers[j][0], centers[j][1], centers[j][2],
                        c=centers[j] / 255, marker='+', alpha=1.0, s=100)

        ax5.set_xlabel('X-axis')
        ax5.set_ylabel('Y-axis')
        ax5.set_zlabel('Z-axis')
        plt.title("3D color values scatterplot")
        plt.show()

def plot_centers(centers):
    extra_part = 0.0
    patches = []

    fig1 = plt.figure("Object's main colors by frequency")
    ax3 = fig1.add_subplot(111)

    for j in list(range(0, len(centers), 1)):
        rect = mpatches.Rectangle(
            (0.1 + extra_part, 0.1), 0.9, 0.9,
            facecolor=centers[j] / 255)

        patches.append(rect)
        extra_part += 1.0

    for p in patches:
        ax3.add_patch(p)

    ax3.set_axis_off()
    ax3.set_frame_on(False)
    plt.axis([0, len(centers), 0, 1])
    plt.tight_layout()
    plt.show()

def most_common_colors(image, image_mask=None, plot_feature=False, cluster_num=2, silhouette_threshold=0.6,
                       size_threshold=1400, scale=True, scale_to_px=50, k_min=2, k_max=5):
    """
        Cluster (K-means) colors in the image to find the most common colors according to silhouette score.

        Parameters
        ----------
        image                 : np.array - image of interest
        image_mask            : np.array - mask of the image
        plot_feature          : bool     - if True: silhoutte graph will be made
        cluster_num           : int      - initial number of clusters
        silhouette_threshold  : int      - threshold to define the number of clusters
        size_threshold        : int      - minimal size of the image to run use silhouette evaluation
        scale                 : bool     - if True: scale image so that the longest side is no more than scale_to_px
        scale_to_px           : int      - if scale: maximum number of pixels for the longest image border
        k_min                 : int      - minimal number of K-means clusters
        k_max                 : int      - maximum number of K-means clusters

        Returns
        -------
        np.array : cluster centers coordinates
        np.array : size of each cluster
        object   : K-means object
    """

    if scale:
        # Scale image so that the longest side is no more than scale_to_px
        scale_factor = scale_to_px / max(image.shape) if max(image.shape) > scale_to_px else 1
        image = skimage.img_as_ubyte(skimage.transform.rescale(image, scale_factor))
        if image_mask is not None:
            image_mask = skimage.img_as_ubyte(skimage.transform.rescale(image_mask, scale_factor))

    # Run classifier for each number of clusters
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))

    # Uzkomentuotas iterpiant Original pic opcija
    #if image_mask is not None:
    #    image_array = image_array[image_mask.reshape(image_mask.shape[0] * image_mask.shape[1]).astype(bool), :]

    bug = True

    # TODO: Fix bug on macOS - sklearn.metrics.silhouette_score doesn't work in multiprocessing app
    if not bug and cluster_num != 1:

        classifiers = [sklearn.cluster.KMeans(n_clusters=n).fit(image_array) for n in np.arange(k_min, k_max)]
        # Find best classifier according to silhouette score
        sillhouette = [sklearn.metrics.silhouette_score(image_array, clas.labels_, metric='euclidean') for clas in
                       classifiers]

        if (np.amax(sillhouette) > silhouette_threshold) or ((image.shape[0] * image.shape[1]) >= size_threshold):
            best_classifier = classifiers[np.argmax(sillhouette)]
        else:
            best_classifier = sklearn.cluster.KMeans(n_clusters=1).fit(image_array)
    else:
        best_classifier = sklearn.cluster.KMeans(n_clusters=cluster_num).fit(image_array)
        sillhouette = [0.6, 0.5, 0.4]

    # Sort cluster centers according to cluster size
    _, counts = np.unique(best_classifier.labels_, return_counts=True)
    sort_index = np.argsort(counts)[::-1]
    cluster_centers = best_classifier.cluster_centers_[sort_index]
    cluster_size = counts[sort_index] / np.sum(counts)

    if plot_feature:
        plt.figure()
        plt.title("Silhouette variance score for different number of clusters")
        plt.plot(np.arange(k_min, k_max), sillhouette)
        plt.figure()
        plt.imshow(image * image_mask[:, :, None])
        plot_centers(cluster_centers)
    return cluster_centers, cluster_size, best_classifier

def get_geometrical_features(image_mask, features=None):
    """
        Extract geometrical features from image.

        Parameters
        ----------
        image_mask : np.array - image mask of interest (contains only one object)
        features   : list     - features for extraction

        Returns
        -------
        pd.DataFrame : all the extracted features regarding the object
    """

    if features is None:
        features = ['Shapes Area', 'Shapes Perimeter',
                    'Centroids row', 'Centroids column', 'Eccentricity', 'Orientation',
                    'Area/Square diff', 'Area/Circle diff', 'Object Perimeter/Circle Perimeter',
                    'Oval', 'Rect', 'Bounding']  # Default features

    region = skimage.measure.regionprops(image_mask)[0]

    features_dict = collections.OrderedDict({})

    # Saving one object information in the image
    x, y = region.coords.T  # x-row, y-column

    feat_num = 0  # to maintain that Three_D() enter once
    for feature in features:

        if feature == 'Shapes Area':
            feature_value = region.area  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Shapes Perimeter':
            feature_value = region.perimeter  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Centroids row':
            feature_value = np.round(region.centroid[0], decimals=2)  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Centroids column':
            feature_value = np.round(region.centroid[1], decimals=2)  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Eccentricity':
            feature_value = region.eccentricity  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Orientation':
            feature_value = region.orientation  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Area/Square diff':

            # draw invisible rectangle around segmented objects to get the percentage object_area/square around the object
            minr, minc, maxr, maxc = region.bbox
            rect_area = (maxc - minc) * (maxr - minr)
            feature_value = region.area / rect_area  # Get feature value
            features_dict[feature] = [feature_value]

        elif (feature == 'Area/Circle diff'):
            circle = (make_circle(region.coords))
            # Calculating the region.area/the circle area
            circle_area = math.pi * circle[2] ** 2
            circle_diff = region.area / circle_area

            feature_value = circle_diff  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Object Perimeter/Circle Perimeter':  # a = Co/Ca (Ca - equivalent_diameter perimeter)
            circle_perimeter = region.equivalent_diameter * math.pi
            feature_value = region.perimeter / circle_perimeter
            features_dict[feature] = [feature_value]

        elif feature == 'Oval':
            if 'Area/Circle diff' not in features:
                circle = (make_circle(region.coords))

            oval = mpatches.Circle((circle[1], circle[0]), radius=circle[2], fill=False, edgecolor='pink',
                                   linewidth=1)
            feature_value = oval  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Rect':
            if 'Area/Square diff' not in features:
                minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='salmon', linewidth=2)

            feature_value = rect  # Get feature value
            features_dict[feature] = [feature_value]

        elif feature == 'Bounding':
            minr, minc, maxr, maxc = region.bbox

            features_dict['Min row'] = minr
            features_dict['Min column'] = minc
            features_dict['Max row'] = maxr
            features_dict['Max column'] = maxc

    features_dataframe = pd.DataFrame(features_dict)
    return features_dataframe



def get_features_for_region(image, image_mask, features_dict, features=None, plot_feature=False,
                            entropy_kernel=skimage.morphology.disk(5)):
    """
        Extract color features from image.

        Parameters
        ----------
        image          : np.array  - image of interest
        image_mask     : np.array  - image mask of interest (contains only one object of interest)
        features_dict  : dict      - extracted features
        features       : list      - features to extract
        plot_feature   : bool      - if True, plot extracted features
        entropy_kernel : object

        features   : list     - features for extraction

        Returns
        -------
        pd.DataFrame : all the extracted features regarding the object
        np.array()   : cropped image
        np.array()   : mask corresponding to a cropped image
    """

    if features is None:
        features = ['Red part', 'Green part', 'Blue part', 'RGB clusters number',
                    'Color inertia', 'Texture', 'Moments',
                    'Gray', 'Entropy', 'Sobel', 'Masked_image']  # Default features

    region = skimage.measure.regionprops(image_mask)[0]
    masked_image = image_mask[:, :, None] * image

    # Saving one object information in the image
    x, y = region.coords.T  # x-row, y-column
    feat_num = 0  # to maintain that Three_D() enter once

    for feature in features:

        if ((feature == 'Red part') or (feature == 'Green part') or (feature == 'Blue part')
                or (feature == 'RGB clusters number') or (feature == 'Color inertia')
                or (feature == 'Texture') or (feature == 'Moments')):

            if feat_num == 0:
                minr, minc, maxr, maxc = region.bbox
                cropped_image = image[minr:maxr, minc:maxc]
                cropped_mask = image_mask[minr:maxr, minc:maxc]
                cropped_image_mask = cropped_mask[:, :, None] * cropped_image

                centers, counts, classifier = most_common_colors(cropped_image, cropped_mask, plot_feature,
                                                                 cluster_num=1, silhouette_threshold=0.6)
                cropped_mask_background = np.logical_not(cropped_mask).astype(bool)
                background_centers, background_counts, _ = most_common_colors(cropped_image, cropped_mask_background,
                                                                              plot_feature, cluster_num=1,
                                                                              silhouette_threshold=0.6)

                color_distance = (np.sqrt(np.sum((np.absolute(np.subtract(centers, background_centers)) ** 2), axis=1)))
                feat_num += 1

                if plot_feature:  # Plotting 3D scatter graph of color values and their centers by kmeans
                    Three_D(cropped_image_mask, centers, plot_feature)

            if feature == 'Red part': #taking the main color of the object
                features_dict[feature] = centers[:, 0].tolist()[np.argmax(color_distance)]
            elif feature == 'Green part':
                features_dict[feature] = centers[:, 1].tolist()[np.argmax(color_distance)]
            elif feature == 'Blue part':
                features_dict[feature] = centers[:, 2].tolist()[np.argmax(color_distance)]
            elif feature == 'RGB clusters number':
                features_dict[feature] = len(centers[:, 0].tolist())
            elif feature == 'Color inertia':
                features_dict[feature] = classifier.inertia_
            elif feature == 'Texture':

                gray_image = skimage.color.rgb2grey(cropped_image_mask)
                glcm_matrix = skimage.feature.greycomatrix(skimage.img_as_ubyte(gray_image), [2], [0], 256, symmetric=True, normed=True)

                features_dict['Texture dissimilarity'] = skimage.feature.greycoprops(glcm_matrix, 'dissimilarity')[0, 0]
                features_dict['Texture contrast'] = skimage.feature.greycoprops(glcm_matrix, 'contrast')[0, 0]
                features_dict['Texture homogeneity'] = skimage.feature.greycoprops(glcm_matrix, 'homogeneity')[0, 0]
                features_dict['Texture ASM'] = skimage.feature.greycoprops(glcm_matrix, 'ASM')[0, 0]
                features_dict['Texture energy'] = skimage.feature.greycoprops(glcm_matrix, 'energy')[0, 0]
                features_dict['Texture correlation'] = skimage.feature.greycoprops(glcm_matrix, 'correlation')[0, 0]

            elif feature == 'Moments':

                gray_image = skimage.color.rgb2grey(cropped_image_mask)
                moments = skimage.measure.moments(gray_image)
                moments_normalised = skimage.measure.moments_normalized(moments)
                moments_hu = skimage.measure.moments_hu(moments_normalised)

                features_dict['First moment'] = moments_hu[0]
                features_dict['Second moment'] = moments_hu[1]
                features_dict['Third moment'] = moments_hu[2]
                features_dict['Fourth moment'] = moments_hu[3]
                features_dict['Fifth moment'] = moments_hu[4]
                features_dict['Sixth moment'] = moments_hu[5]
                features_dict['Seventh moment'] = moments_hu[6]

        # Features which need to use gray masked image
        elif ((feature == 'Gray') or (feature == 'Entropy') or (feature == 'Sobel') or (feature == 'Masked_image')):
            gray_masked = (skimage.color.rgb2gray(masked_image))

            if feature == 'Gray':
                gray = gray_masked[x, y].flatten()

                feature_value = max(gray)  # Get feature value
                features_dict['Gray Max'] = [feature_value]

                feature_value = min(gray)  # Get feature value
                features_dict['Gray Min'] = [feature_value]

                feature_value = np.mean(gray)  # Get feature value
                features_dict['Gray Mean'] = [feature_value]

                feature_value = np.std(gray)  # Get feature value
                features_dict['Gray Std'] = [feature_value]

            elif feature == 'Entropy':

                entropy_image = skimage.filters.rank.entropy(gray_masked, selem=entropy_kernel, mask=image_mask)
                etr = entropy_image[x, y].flatten()

                feature_value = min(etr)  # Get feature value
                features_dict['Entropy min'] = [feature_value]

                feature_value = max(etr)  # Get feature value
                features_dict['Entropy max'] = [feature_value]

                feature_value = np.mean(etr)  # Get feature value
                features_dict['Entropy mean'] = [feature_value]

                feature_value = np.std(etr)  # Get feature value
                features_dict['Entropy std'] = [feature_value]

            elif feature == 'Sobel':

                sobel_image = skimage.filters.sobel(gray_masked, mask=image_mask)
                sob = sobel_image[x, y].flatten()

                feature_value = max(sob)  # Get feature value
                features_dict['Sobel max'] = [feature_value]

                feature_value = np.mean(sob)  # Get feature value
                features_dict['Sobel mean'] = [feature_value]

                feature_value = np.std(sob)  # Get feature value
                features_dict['Sobel std'] = [feature_value]

            elif feature == 'Masked_image':
                feature_value = gray_masked  # Get feature value
                features_dict[feature] = [feature_value]


    features_dataframe = pd.DataFrame(features_dict)
    return features_dataframe, cropped_image, cropped_mask


def get_regions_from_image(path, path_save_pics, kwargs_features, channel='gray_eye', noise_filter='gaussian',
                           edge_detection='sobel', noise_filter2='gaussian', thresholding='otsu',
                           closing='closing', fill_holes='fill_holes', filter_params= [None,5,5,5,5,2.1,None,0],
                           plot_filters=False, plot_object=False, min_region_size=100, cropped_image_save=False):
    """
        Main function to divide image to smaller areas of interest and extract features.

        Parameters
        ----------
        path             : str        - path to image
        path_save_pics   : str        - path to location where processed images will be saved
        kwargs_features  : dict       - features to process
        channel          : str        - possible values: 'r','g','b','gray eye','gray equal','None'
        noise_filter     : str        - possible values: 'gaussian','median','mean','None'
        edge_detection   : str        - possible values: 'sobel','canny','entropy','None'
        noise_filter2    : str        - possible values: 'gaussian','median','mean','None'
        thresholding     : str        - possible values: 'otsu', 'yen', 'mean', 'minimum', 'local otsu', 'None'
        closing          : str        - possible values: 'closing','None'
        fill_holes       : str        - possible values: 'fill holes', 'None'
        filter_params    : list       - contains associated parameter with all the operations above

        Returns
        -------
        pd.DataFrame : extracted information from the image
        img          : image to show
        patches      : all circles and rectangles around the damage in the gray picture
        site_nr      : site number associated with image
    """

    site_nr = re.findall(r'\d+', path.split(os.sep)[-1])[0]
    image = skimage.img_as_ubyte(skimage.io.imread(path, ))

    bin_pic = process_image(path_save_pics, site_nr, image, channel, noise_filter, edge_detection, noise_filter2, thresholding,
                            closing, fill_holes, filter_params, plot_filters)

    labeled_image, n_labels = skimage.measure.label(bin_pic, return_num=True)

    # define lists to hold extracted info
    image_regions = []
    patches = []

    # create mask
    mask = np.zeros((image.shape[0], image.shape[1]))
    img = np.zeros_like(image)
    number = 0

    if path.rsplit(os.sep,2)[1] == 'thumb':
        eq_hist_site = os.path.join(os.path.join(path.rsplit(os.sep, 2)[0], 'eq_hist'), 'site{}.jpg'.format(site_nr))
        image2 = skimage.img_as_ubyte(skimage.io.imread(eq_hist_site, ))
    else:
        thumb_site = os.path.join(os.path.join(path.rsplit(os.sep, 2)[0], 'thumb'), 'site{}.jpg'.format(site_nr))
        image2 = skimage.img_as_ubyte(skimage.io.imread(thumb_site, ))

    for n in range(1, n_labels + 1):  # leave zero as it is background
        if ((kwargs_features['features'] is not None) and (n == 1) and (plot_object == True)):
            kwargs_features['features'].extend(['Masked_image', 'Oval', 'Rect', 'Bounding'])

        image_mask = (labeled_image == n).astype(np.uint8)
        region_area = np.sum(image_mask)
        feat = get_geometrical_features(image_mask, kwargs_features['features'])

        if (region_area < min_region_size or region_area > 0.75 * (image.shape[0]*image.shape[1])):
            continue


        feat_image, cropped_image, cropped_mask = get_features_for_region(image, image_mask, feat, **kwargs_features)
        img[labeled_image == n] = np.array([feat['Red part'], feat['Green part'], feat['Blue part']]).astype(np.uint8).T
        feat_image2, _, _ = get_features_for_region(image2, image_mask, collections.OrderedDict({}), **kwargs_features)

        if 'Masked_image' in feat_image2:
            mask = np.sum([mask, feat_image2['Masked_image'][0]], axis=0)
            del feat_image2['Masked_image']

        if 'Masked_image' in feat_image:
            mask = np.sum([mask, feat_image['Masked_image'][0]], axis=0)
            del feat_image['Masked_image']

        if 'Oval' in feat_image:
            patches.append(feat['Oval'][0])
            del feat_image['Oval']

        if 'Rect' in feat_image:
            patches.append(feat['Rect'][0])
            del feat_image['Rect']

        if cropped_image_save:
            skimage.io.imsave(os.path.join(path_save_pics, 'cropped_site{}_{}.png'.format(site_nr, number)),
                              cropped_image)

        names = []

        if path.rsplit(os.sep, 2)[1] == 'thumb':
            oposite_name = 'eq_hist'
        else:
            oposite_name = 'thumb'

        for i in feat_image2.columns:
            names.append(i + " " + oposite_name)
        feat_image2.columns = names  # change names of the image2 features columns

        feat_image.insert(0, 'Object in image nr', number)
        all_feat_image = pd.concat([feat_image, feat_image2], axis=1)
        number += 1
        image_regions.append(all_feat_image)

    if plot_object:

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img.astype(np.uint8))

        for p in patches:
            new_p = copy(p)
            ax.add_patch(new_p)

        ax.set_axis_off()
        plt.tight_layout()
        plt.title("All objects of interest")
        plt.show(block=False)

    # pass the empty dataframe to concat in case if there is no objects of interest in the image
    if len(image_regions) == 0:
        features_dict = collections.OrderedDict({})
        features_dataframe = pd.DataFrame(data=features_dict)
        image_regions.append(features_dataframe)

    image_regions_dataframe = pd.concat(image_regions, ignore_index=True)
    image_regions_dataframe.insert(0, 'site nr', site_nr)
    return image_regions_dataframe, img, patches, site_nr

def get_regions_from_images_mp(directory_read, multi, path_save_pics, cores_number, kwargs_features, **kwargs_regions):
    '''
    Add multiprocessing functionality for feature extraction.

    Parameters
    ----------
    directory_read   : str        - directory to read images from
    multi            : bool       - if True: use multiprocessing
    path_save_pics   : dict       - path to the directory where images will be saved
    cores_number     : int        - number of cores to use

    Returns
    -------
    pd.DataFrame : extracted information from all the images in the directory
    '''

    paths = [os.path.join(directory_read, fname) for fname in os.listdir(directory_read) if fname.lower().startswith('site')]

    patches_list = dict()
    if multi:
        with multiprocessing.Pool(processes=cores_number) as pool:
            images_regions = pool.map(
                functools.partial(
                    get_regions_from_image,
                    path_save_pics=path_save_pics,
                    kwargs_features=kwargs_features,
                    **kwargs_regions),
                paths)

            images_regions, img, patches, site_nr = zip(*images_regions)
            dataset = pd.concat(images_regions, ignore_index=True)
        dataset.to_csv(os.path.join(path_save_pics, 'directory pictures data.csv'))
        return dataset

    else:
        one_image = []
        for i in paths:
            one_image.append(get_regions_from_image(i, path_save_pics, kwargs_features, **kwargs_regions))
        dataset = pd.concat(one_image, ignore_index=True)
        dataset.to_csv(os.path.join(path_save_pics,'directory pictures data.csv'))
        return dataset