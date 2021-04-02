import os
import errno
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import skimage.io
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cluster
import sklearn.metrics
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def show_pics(train_pca, best_classifier, objects, directory_cropped_image):

    points = pd.DataFrame(train_pca)
    points['Labels'] = best_classifier.labels_
    centroids = np.array([(points.loc[points['Labels'] == i].mean(axis=0)[:-1]) for i in range(cluster_num)])

    a = train_pca
    b = centroids[best_classifier.labels_]

    # For Spectral Clustering
    k_number = 10  # number of cluster pictures to show

    # Amount condition for images visualization.
    smallest_cluster_size = list(objects['PCA labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    objects['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))
    similar_pics_idx = [
        (objects.loc[objects['PCA labels'] == i]['Clustering center distance']).nsmallest(k_number).index for i
        in range(len(centroids))]

    for i in range(len(similar_pics_idx)):
        print("Label number: %s" % i)

        # Plot figure with subplots of different sizes
        fig = plt.figure()
        # set up subplot grid
        gridspec.GridSpec(3, 2)
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=objects['PCA labels'] == i)

        for i_i in range(k_number):
            object_id = similar_pics_idx[i][i_i]
            site_nr = str(int(objects.iloc[object_id]['site nr'])) + '_' + str(
                int(objects.iloc[object_id]['Object in image nr']))
            img = skimage.io.imread(directory_cropped_image + "cropped_site{}.png".format(site_nr))
            plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.title("Spectral clustering")
            plt.imshow(img)

            img = skimage.io.imread(
                "{}site{}.jpg".format(directory_image, str(int(objects.iloc[object_id]['site nr']))))
            plt.subplot(222)
            plt.axhline(objects.iloc[object_id]['Centroids row'])
            plt.axvline(objects.iloc[object_id]['Centroids column'])
            plt.imshow(img)

    return objects



def k_means_code(train_pca, objects, directory_cropped_image, directory_image,
                 TwoD=True, ThreeD=False, silhouette_threshold=0.48, force=False, k_number=10):
    '''
    K-means clustering after PCA analysis
    Parameters
    ----------
    train_pca: input data
    objects: DataFrame of objects of interest
    directory_cropped_image: path to cropped images directory
    directory_image: path to image directory
    TwoD:  if True, plot clustering in 2D
    ThreeD: if True, plot clustering in 3D
    silhouette_threshold
    force: if True, use clusters through silhouette method
    k_number: number of clusters in K-means

    Returns
    -------
    objects: DataFrame with added labels
    '''

    k_min = 2
    k_max = 5

    classifiers = [sklearn.cluster.KMeans(n_clusters=n).fit(train_pca) for n in np.arange(k_min, k_max)]

    # Find best classifier according to silhouette score
    sillhouette = [sklearn.metrics.silhouette_score(train_pca, clas.labels_, metric='euclidean') for clas in
                   classifiers]
    print("Silhouette scores:%s" % sillhouette)
    if (np.amax(sillhouette) > silhouette_threshold) or (force != False):
        best_classifier = classifiers[np.argmax(sillhouette)]
    else:
        best_classifier = sklearn.cluster.KMeans(n_clusters=1).fit(train_pca)

    if ThreeD:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(objects['Red part'], objects['Green part'], objects['Blue part'],
                   c=best_classifier.labels_, alpha=0.1)
        for i in best_classifier.cluster_centers_:
            ax.scatter(i[0], i[1], i[2], c=['r'], alpha=1.0, s=10)

    if TwoD:
        plt.figure("K-Means colors")
        plt.title("KMeans clustering")
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_)

    objects['PCA labels'] = best_classifier.labels_
    a = train_pca
    b = best_classifier.cluster_centers_[best_classifier.labels_]

    # Amount condition for images visualization.
    smallest_cluster_size = list(objects['PCA labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    objects['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))
    similar_pics_idx = [
        (objects.loc[objects['PCA labels'] == i]['Clustering center distance']).nsmallest(k_number).index for i
        in range(len(best_classifier.cluster_centers_))]

    for i in range(len(similar_pics_idx)):
        print("Label number: %s" % i)

        # Plot figure with subplots of different sizes
        fig = plt.figure()
        # set up subplot grid
        gridspec.GridSpec(3, 2)
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_ == i)
        for i_i in range(k_number):
            object_id = similar_pics_idx[i][i_i]
            site_nr = str(int(objects.iloc[object_id]['site nr'])) + '_' + str(
                int(objects.iloc[object_id]['Object in image nr']))
            img = skimage.io.imread(directory_cropped_image + "cropped_site{}.png".format(site_nr))

            plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.title("KMeans Clustering")
            plt.imshow(img)

            img = skimage.io.imread(
                "{}site{}.jpg".format(directory_image, str(int(objects.iloc[object_id]['site nr']))))

            plt.subplot(222)
            plt.axhline(objects.iloc[object_id]['Centroids row'])
            plt.axvline(objects.iloc[object_id]['Centroids column'])
            plt.imshow(img)

    return objects


def PCA(X_train, plot=False):

    X_train_pca= X_train.values
    sc = sklearn.preprocessing.StandardScaler()
    X_train_std = sc.fit_transform(X_train_pca)


    pca = sklearn.decomposition.PCA()
    X_train_pca = pca.fit_transform(X_train_std)


    # Take the number of PC  corresponding to 80percent of variance
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pca = len(cum[cum < 0.80])

    if n_pca <=1:
        n_pca += 1

    print("PCA #components: %s, variance: %s" % (n_pca,round(cum[n_pca],2)))
    pca2 = sklearn.decomposition.PCA(n_components=n_pca)
    X_train_pca = pca2.fit_transform(X_train_std)

    if plot:
        plt.figure()
        plt.bar(range(1, X_train.shape[1] + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
        plt.step(range(1, X_train.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.show()

        # For 2PC
        if n_pca > 1:
            plt.figure()
            plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.show()

    return X_train_pca,pca2.components_,pca2.explained_variance_ratio_


def spectral_clustering(train_pca, objects, directory_cropped_image, directory_image,  cluster_num=3):
    '''
    Perform spectral clustering on input.
    Parameters
    ----------
    train_pca: input data
    objects: DataFrame containing object information
    directory_cropped_image: path to directory cropped images are located
    directory_image: path to directory where original images are located
    cluster_num: initial number of clusters

    Returns
    -------
    objects: DataFrame with added labels
    '''

    best_classifier = sklearn.cluster.SpectralClustering(n_clusters=cluster_num, eigen_solver='arpack',
                                                         affinity="nearest_neighbors").fit(train_pca)
    objects['PCA labels'] = best_classifier.labels_
    plt.figure()
    plt.title("Spectral clustering")
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_)

    points = pd.DataFrame(train_pca)
    points['Labels'] = best_classifier.labels_
    centroids = np.array([(points.loc[points['Labels'] == i].mean(axis=0)[:-1]) for i in range(cluster_num)])

    a = train_pca
    b = centroids[best_classifier.labels_]

    # For Spectral Clustering
    k_number = 10  # number of cluster pictures to show

    # Amount condition for images visualization.
    smallest_cluster_size = list(objects['PCA labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    objects['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))
    similar_pics_idx = [
        (objects.loc[objects['PCA labels'] == i]['Clustering center distance']).nsmallest(k_number).index for i
        in range(len(centroids))]

    for i in range(len(similar_pics_idx)):
        print("Label number: %s" % i)

        # Plot figure with subplots of different sizes
        fig = plt.figure()
        # set up subplot grid
        gridspec.GridSpec(3, 2)
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=objects['PCA labels'] == i)

        for i_i in range(k_number):
            object_id = similar_pics_idx[i][i_i]
            site_nr = str(int(objects.iloc[object_id]['site nr'])) + '_' + str(
                int(objects.iloc[object_id]['Object in image nr']))
            img = skimage.io.imread(directory_cropped_image + "cropped_site{}.png".format(site_nr))
            plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.title("Spectral clustering")
            plt.imshow(img)

            img = skimage.io.imread(
                "{}site{}.jpg".format(directory_image, str(int(objects.iloc[object_id]['site nr']))))
            plt.subplot(222)
            plt.axhline(objects.iloc[object_id]['Centroids row'])
            plt.axvline(objects.iloc[object_id]['Centroids column'])
            plt.imshow(img)

    return objects


def mean_shift(train_pca, objects, directory_cropped_image, directory_image, quant=0.20, k_number=10):
    '''
    Mean shift clustering implementation.
    Parameters
    ----------
    train_pca: input data
    objects: DataFrame containing object information
    directory_cropped_image: directory where cropped images are located
    directory_image: directory where original images are located
    quant: mean shift internal parameter
    k_number: number of initial clusters

    Returns
    -------
    objects: DataFrame with object labels
    '''

    # The following bandwidth can be automatically detected using
    bandwidth = sklearn.cluster.estimate_bandwidth(train_pca, quantile=quant, n_samples=500)

    best_classifier = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(train_pca)
    # ms = sklearn.cluster.MeanShift(bin_seeding=True).fit(X_train_pca)
    labels = best_classifier.labels_
    cluster_centers = best_classifier.cluster_centers_

    n_clusters_ = len(np.unique(labels))

    print("number of estimated clusters : %d" % n_clusters_)

    plt.figure()
    plt.title("Mean_shift")
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_)

    objects['PCA labels'] = best_classifier.labels_

    a = train_pca
    b = best_classifier.cluster_centers_[best_classifier.labels_]

    # Amount condition for images visualization.
    smallest_cluster_size = list(objects['PCA labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    objects['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))
    similar_pics_idx = [
        (objects.loc[objects['PCA labels'] == i]['Clustering center distance']).nsmallest(k_number).index for i
        in range(len(best_classifier.cluster_centers_))]

    for i in range(len(similar_pics_idx)):
        print("Label number: %s" % i)

        # Plot figure with subplots of different sizes
        fig = plt.figure()
        # set up subplot grid
        gridspec.GridSpec(3, 2)
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_ == i)
        for i_i in range(k_number):
            object_id = similar_pics_idx[i][i_i]
            site_nr = str(int(objects.iloc[object_id]['site nr'])) + '_' + str(
                int(objects.iloc[object_id]['Object in image nr']))
            img = skimage.io.imread(directory_cropped_image + "cropped_site{}.png".format(site_nr))

            plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.title("Mean_shift")
            plt.imshow(img)

            # img = skimage.io.imread("/Users/macbook/Desktop/D6_COPY/Site{}.jpg".format(str(int(all_objects.iloc[object_id]['site nr']))))
            img = skimage.io.imread(
                "{}site{}.jpg".format(directory_image, str(int(objects.iloc[object_id]['site nr']))))

            plt.subplot(222)
            plt.axhline(objects.iloc[object_id]['Centroids row'])
            plt.axvline(objects.iloc[object_id]['Centroids column'])
            plt.imshow(img)

    return objects


def DBSCAN_get_outliers(train_pca, objects, eps=1.6, min_samples = 3):
    '''
    DBSCAN clustering implementation.
    Parameters
    ----------
    train_pca: input data
    objects: DataFrame with object information
    eps: DBSCAN internal parameter
    min_samples: DBSCAN internal parameter

    Returns
    -------
    objects: DataFrame containing given labels
    '''

    best_classifier = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(train_pca)
    objects['PCA labels'] = best_classifier.labels_
    outliers_address = []

    if objects[objects['PCA labels'] == -1].shape[0] != 0:

        # Outliers exist -> let's get its addresses

        outliers = objects.loc[objects['PCA labels'] == -1]
        for i in range(outliers.shape[0]):
            object_id = outliers.index[i]
            outliers_address.append(str(int(objects.iloc[object_id]['site nr'])) + '_' + str(int(objects.iloc[object_id]['Object in image nr'])))

        outliers_index = outliers.index
    else:
        # Outliers do not exsist
        outliers = None
        outliers_address = []
        outliers_index = []

    return outliers_index, objects, outliers_address


def DBSCAN_show_outliers(train_pca, objects, directory_cropped_image, directory_image, eps=1.6, min_samples = 3, plot=False):
    '''  Plot DBSCAN outliers. '''

     best_classifier = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples,metric='euclidean').fit(train_pca)
     objects['PCA labels'] = best_classifier.labels_

     if plot:
        plt.figure()
        plt.title("DBSCAN outliers")
        plt.scatter(train_pca[:, 0], train_pca[:, 1], c=best_classifier.labels_, cmap='Set1')

     print("Labeled as ouliers/total objects: %s/%s, eps value: %s" % ((objects[objects['PCA labels'] == -1].shape[0]),objects.shape[0],eps))

     outliers_sites_nr = []
     if objects[objects['PCA labels'] == -1].shape[0] != 0:
         outliers = ([(objects.loc[objects['PCA labels'] == -1])][0])
         for i in range(outliers.shape[0]):
             object_id = outliers.index[i]
             outliers_sites_nr.append(object_id)
             site_nr = str(int(objects.iloc[object_id]['site nr'])) + '_' + str(
                 int(objects.iloc[object_id]['Object in image nr']))

             if plot:
                img = skimage.io.imread(directory_cropped_image + "cropped_site{}.png".format(site_nr))
                plt.figure(figsize=(6, 6))
                plt.subplot(221)
                plt.title("DBSCAN outliers")
                plt.suptitle(str(site_nr) + " | " + str(i))
                plt.imshow(img)

                img = skimage.io.imread(
                    "{}site{}.jpg".format(directory_image, str(int(objects.iloc[object_id]['site nr']))))

                plt.subplot(222)
                plt.axhline(objects.iloc[object_id]['Centroids row'])
                plt.axvline(objects.iloc[object_id]['Centroids column'])
                plt.imshow(img)

     return outliers_sites_nr, objects


def DBSCAN_label_outliers(outliers_sites, objects, damaged_outliers):
    ''' Evaluate if DBSCAN outliers are actual damages.'''

    damaged_outliers_real_idx = [outliers_sites[i] for i in damaged_outliers]
    label_num = np.unique(objects['PCA labels'])[-1]
    # What if damaged_outliers is empty list? We need to add +1 label, not +2
    pics_different_clusters = [i for i in outliers_sites if i in damaged_outliers_real_idx]
    if (len(pics_different_clusters) != 0) and (len(outliers_sites) != len(damaged_outliers)):
        # Case 1: There are damaged and not damaged outliers
        for i in outliers_sites:
            if i in damaged_outliers_real_idx:
                # -> outlier is marked as damaged
                objects.loc[int(i), 'PCA labels'] = label_num + 2
            else:
                # -> outlier is marked as garbage
                objects.loc[int(i), 'PCA labels'] = label_num + 1

    elif (len(outliers_sites) == len(damaged_outliers)):

        # Case 2: All outliers are marked as damaged
        for i in outliers_sites:
            objects.loc[int(i), 'PCA labels'] = label_num + 1
    else:

        # Case 3: No damaged outliers
        # if there is no damaged outliers - damaged_outliers = []

        for i in outliers_sites:
            objects.loc[int(i), 'PCA labels'] = label_num + 1


    return objects


def updated_DBSCAN_clustering(train_pca, objects, k_number=5):
    ''' Redo DBSCAN clustering. '''

    points = pd.DataFrame(train_pca)
    points['Labels'] = list(objects['PCA labels'])

    centroids = np.array([(points.loc[points['Labels'] == i].mean(axis=0)[:-1]) for i in (list(np.unique(points['Labels'])))])

    # For operation below (b=centroids[points['Labels']]) we should make sure that points['Labels'] contains elements starting from 0.
    # centroids[0] is outliers -> this means that points['Labels']=0 should also mark outliers.
    # We can add +1 to all points['Labels'] elements and later subtract 1.

    if -1 in list(points['Labels'].value_counts().index):
        points['Labels'] += 1
        a = train_pca
        b = centroids[points['Labels']]
        points['Labels'] -= 1

    else:
        # if there is no outliers, just leave it
        a = train_pca
        b = centroids[points['Labels']]


    # Amount condition for images visualization.
    smallest_cluster_size = list(points['Labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    # For DBSCAN
    points['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))

    if -1 in list(points['Labels'].value_counts().index): # If outliers exist

        similar_pics_idx = [(points.loc[points['Labels'] == i]['Clustering center distance']).nsmallest(k_number).index
                        for i in range(-1,len(centroids)-1,1)]

    else:

        similar_pics_idx = [(points.loc[points['Labels'] == i]['Clustering center distance']).nsmallest(k_number).index
                        for i in range(0,len(centroids))]

    if np.unique(points['Labels'])[0] == -1:
        # the last element should be outliers
        similar_pics_idx += [similar_pics_idx.pop(0)]

    return objects, similar_pics_idx, smallest_cluster_size

def DBSCAN_clustering(train_pca, objects, directory_cropped_image, directory_image, k_number=5):
    ''' Perform DBSCAN clustering. '''

    points = pd.DataFrame(train_pca)
    points['Labels'] = list(objects['PCA labels'])
    centroids = np.array([(points.loc[points['Labels'] == i].mean(axis=0)[:-1]) for i in (list(np.unique(objects['PCA labels'])))])

    ## if all outliers are damages, 7->6, nes the one before last is for trashes
    start, end = np.unique(objects['PCA labels'])[0], np.unique(objects['PCA labels'])[-1]
    L = list(range(start, end + 1, 1))
    missing_element = sorted(set(range(start, end + 1)).difference(np.unique(objects['PCA labels'])))
    if len(missing_element) > 0:  # if there is no undamaged outliers -> change label to -1,because max is reserved for damages
        rows = objects[objects['PCA labels'] == max(np.unique(objects['PCA labels']))]
        for obj in list(rows.index):
            objects.loc[obj, 'PCA labels'] = max(np.unique(objects['PCA labels'])) - 1

    a = train_pca
    b = centroids[objects['PCA labels']]
    # Amount condition for images visualization.
    smallest_cluster_size = list(objects['PCA labels'].value_counts())[-1]
    if smallest_cluster_size < k_number:
        k_number = smallest_cluster_size

    # For DBSCAN
    objects['Clustering center distance'] = np.sqrt(np.sum((np.absolute(np.subtract(a, b)) ** 2), axis=1))

    similar_pics_idx = [(objects.loc[objects['PCA labels'] == i]['Clustering center distance']).nsmallest(k_number).index for i
        in range(-1,len(centroids)-1,1)]

    print("Similar pics idx", similar_pics_idx)

    if np.unique(objects['PCA labels'])[0] == -1:
        # if 'outliers' exist, change it's indices from the first ones to the last one to preserve the future code
        # the last indices are reserved for the outliers
        similar_pics_idx += [similar_pics_idx.pop(0)]

    print("Similar pics idx2", similar_pics_idx)


    return objects, similar_pics_idx


def eq_hist_check(all_objects_eq_hist, excel_file, radius, unwanted_features=['Unnamed: 0',
                                                'Red part', 'Blue part', 'Green part', 'Entropy min', 'Entropy max',
                                                'Entropy mean', 'Entropy std'],pca=True, plot=False):


    # Damages mostly in the centers - limit the possible coordinates by cutting the edges of graph
    if plot:
        plt.figure()
        plt.title("Before X,Y coordinates limit")
        plt.scatter(all_objects_eq_hist['Centroids row'], all_objects_eq_hist['Centroids column'],
                    s=all_objects_eq_hist['Shapes Area'] * 10 / all_objects_eq_hist['Shapes Area'].max())

    column =all_objects_eq_hist['Centroids column'] - all_objects_eq_hist['Centroids column'].max()/2
    row = all_objects_eq_hist['Centroids row'] - all_objects_eq_hist['Centroids row'].max()/2
    all_objects_eq_hist['dist'] = np.sqrt(column**2 + row**2)

    all_objects_eq_hist = all_objects_eq_hist[all_objects_eq_hist['dist'] < radius]

    del all_objects_eq_hist['dist']

    # Damages mostly in the centers - limit the possible coordinates by cutting the edges of graph
    if plot:
        plt.figure()
        plt.title("After X,Y coordinates limit")
        plt.scatter(all_objects_eq_hist['Centroids row'], all_objects_eq_hist['Centroids column'],
                    s=all_objects_eq_hist['Shapes Area'] * 10 / all_objects_eq_hist['Shapes Area'].max())

    df2 = pd.read_csv(excel_file, decimal=',', sep='\t')

    # Delete elements with -2 or -1 'Insp. status' and reindex all_objects after
    # Delete 'Insp. Status' == -2 rows from both (df and all_objects)
    df2 = df2[df2['Insp. Status'] != -2]
    site_nr = df2.loc[df2['Insp. Status'] != -2]['Site Nr.'].tolist()
    all_objects_eq_hist = all_objects_eq_hist[all_objects_eq_hist['site nr'].astype(int).isin(site_nr)]
    all_objects_eq_hist = all_objects_eq_hist.reset_index(drop=True)

    df2 = df2[df2['Insp. Status'] != -1]
    site_nr = df2.loc[df2['Insp. Status'] != -1]['Site Nr.'].tolist()
    all_objects_eq_hist = all_objects_eq_hist[all_objects_eq_hist['site nr'].astype(int).isin(site_nr)]
    all_objects_eq_hist = all_objects_eq_hist.reset_index(drop=True)

    # Set row index starting to - to coincide with 'Site Nr.'
    if df2.index[0] != min(df2['Site Nr.'].astype(int)):
        df2.index = df2.index + 1

    a1 = df2.loc[:, ['Site Nr.', 'Insp. Status']]
    a1.columns = ['site nr', 'Insp. Status']
    all_objects_eq_hist['site nr'] = pd.to_numeric(all_objects_eq_hist['site nr'])
    all_objects_eq_hist = pd.merge(a1, all_objects_eq_hist, on='site nr')

    X_train = all_objects_eq_hist[all_objects_eq_hist.columns.difference(['site nr', 'Min row', 'Max row', 'Max column',
                                                                          'Min column', 'Object in image nr',
                                                                          'RGB clusters number','Insp. Status', 'Centroids column', 'Centroids row',
                                                                          'Red part thumb', 'Green part thumb','Blue part thumb'])]

    if pca:
        X_train_std, pca_components, _ =  PCA(X_train,plot=True)
        return X_train_std, all_objects_eq_hist

    elif pca == False:
        sc = sklearn.preprocessing.StandardScaler()
        X_train_std = sc.fit_transform(X_train.values)
        return X_train_std, all_objects_eq_hist


def saving_cluster(all_objects, directory_cropped_image):
    ''' Create directory to save objects wrt the given class by clustering algorithm. '''

    warnings.filterwarnings("ignore")
    directory_save_cluster = os.path.join(directory_cropped_image,'Cluster')

    for i in list(np.unique(all_objects['PCA labels'])):
        print("Label being saved now: %s" % (i))
        directory_save_cluster_cluster = directory_save_cluster + "%s/" % i

        try:
            os.makedirs(directory_save_cluster_cluster)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise