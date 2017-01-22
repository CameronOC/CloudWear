from os import listdir, makedirs
from os.path import join, exists, isdir
from time import time

import numpy as np
import sys
import shutil
import os
import json
import hashlib

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as pl
from scipy.misc import imread
from PIL import Image, ImageOps
from bunch import Bunch

CLOTHE_PATH = './clothe'
CLOTHE_PROC_PATH = './clothe_processed'
CLOTHE_TEST_PATH = './clothe_test'
#CLOTHE_TEST_PROC_PATH = './clothe_test_processed'
CLOTHE_TEST_PROC_PATH = './clothe_test_processed'

saved_stdout = sys.stdout

def get_clothe(h=200, w=200):
    # processing data: make all images h X w in size and gray scale, save them in diff folder
    if not exists(CLOTHE_PROC_PATH):
        makedirs(CLOTHE_PROC_PATH)

        for clothe_name in sorted(listdir(CLOTHE_PATH)):
            folder = join(CLOTHE_PATH, clothe_name)
            if not isdir(folder):
                continue

            # make new dir for the clothe
            new_folder = join(CLOTHE_PROC_PATH, clothe_name)
            makedirs(new_folder)

            # iterate over existing clothe's pictures and process each one
            paths = [join(folder, f) for f in listdir(folder) if f != '.DS_Store']
            ###print("printing folder:" + str(folder))
            ###print("printing paths" + str(paths))
            for i, path in enumerate(paths):
                img = Image.open(path).convert('RGB')
                img = ImageOps.fit(img, (w, h), Image.ANTIALIAS, (0.5, 0.5))
                img = ImageOps.grayscale(img)

                new_path = join(CLOTHE_PROC_PATH, clothe_name, str(i)) + '.jpg'
                ###print("new_path :" + str(new_path))
                img.save(new_path)

    if exists(CLOTHE_TEST_PROC_PATH):
        shutil.rmtree(CLOTHE_TEST_PROC_PATH)
    if not exists(CLOTHE_TEST_PROC_PATH):
        makedirs(CLOTHE_TEST_PROC_PATH)


        #for clothe_name in sorted(listdir(CLOTHE_TEST_PATH)):
            #test_folder = join(CLOTHE_TEST_PATH,clothe_name)
        test_folder = CLOTHE_TEST_PATH
        #print("test_folder: " + str(test_folder))

        #if not exists(test_folder):
        #    continue

        #new_test_folder = join(CLOTHE_TEST_PROC_PATH, clothe_name)
        #print("new_test_folder"+ str(new_test_folder))
        #makedirs(new_test_folder)


        test_paths = [join(test_folder,f) for f in listdir(test_folder) if f!='.DS_Store']
        #print("test paths: " + str(test_paths))
        for i, test_path in enumerate(test_paths):
            test_img = Image.open(test_path).convert('RGB')
            test_img = ImageOps.fit(test_img,(w,h), Image.ANTIALIAS, (0.5, 0.5))
            test_img = ImageOps.grayscale(test_img)
            #print("test_img" + str(type(test_img)))
            #print("test_img shape" + str(test_img.size))
            #print("printing i :" + str(i))
            new_test_path = join(CLOTHE_TEST_PROC_PATH,str(i)) + '.jpg'
            test_img.save(new_test_path)

        new_test_paths = [join(CLOTHE_TEST_PROC_PATH,f) for f in listdir(test_folder) if f!='.DS_Store']



    # read clothe names and paths
    clothe_names, clothe_paths = [], []
    for clothe_name in sorted(listdir(CLOTHE_PROC_PATH)):
        folder = join(CLOTHE_PROC_PATH, clothe_name)
        if not isdir(folder):
            continue
        paths = [join(folder, f) for f in listdir(folder) if f != '.DS_Store']
        n_images = len(paths)
        clothe_names.extend([clothe_name] * n_images)
        clothe_paths.extend(paths)

    #print("clothe paths" + str(clothe_paths))

    n_clothe = len(clothe_paths)
    target_names = np.unique(clothe_names)
    target = np.searchsorted(target_names, clothe_names)

    # read data
    clothes = np.zeros((n_clothe, h, w), dtype=np.float32)
    for i, clothe_path in enumerate(clothe_paths):
            img = imread(clothe_path)
            clothe = np.asarray(img, dtype=np.uint32)
            clothes[i, ...] = clothe

    #print("clothes" + str(type(clothes)))
    #print("clothes shape" + str(clothes.shape))
    test_img = imread(new_test_paths[0])
    test_clothe = np.asarray(test_img, dtype = np.uint32)



    # shuffle clothe
    indices = np.arange(n_clothe)
    np.random.RandomState(42).shuffle(indices)
    clothes, target = clothes[indices], target[indices]

    #print("clothes" + str(type(clothes)))
    #print("clothes shape" + str(clothes.shape))

    #print("test clothe" + str(type(test_clothe)))
    #print("test clothe shape" + str(test_clothe.shape))

    #

    return Bunch(data=clothes.reshape(len(clothes), -1), test_clothe=test_clothe,
                 images=clothes,
                 target=target, target_names=target_names,
                 DESCR="Pokemon dataset")


def main():
    t0 = time()
    np.random.seed(3)
    clothe = get_clothe()

    #h, w = clothe.images[0].shape

    X = clothe.data
    y = clothe.target
    test = clothe.test_clothe




    #n_classes = clothe.target_names.shape[0]

    #kf = KFold(len(y), n_folds=4, shuffle=True)
    scores = 0.0

    #print(kf)

    X_train = np.array([X[i] for i in range(X.shape[0])])
    y_train = np.array([y[i] for i in range(y.shape[0])])
    test = np.array([test])
    test = test.reshape((1,40000))
    #test =    np.array([test[i] for i in range(test.shape[0])])



    ###############################################################################
    # Apply PCA
    n_components = 18  # 80%
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    # eigenclothes = pca.components_.reshape((n_components, h, w))

    # print ("Projecting the input data on the eigenclothe orthonormal basis")
    X_train_pca = pca.transform(X_train)
    test_pca = pca.transform(test)

    # reconstruction = pca.inverse_transform(X_train_pca[1])
    # im = Image.fromarray(reconstruction.reshape(h,w))
    # im.show()

    ###############################################################################

    ###############################################################################
    # Train an SVM classification model
    #print ("Fitting the classifier to the training set")






    param_grid = {
        'n_neighbors': list(xrange(1, 15)),
    }

    clf = GridSearchCV(KNeighborsClassifier(), param_grid)
    clf = clf.fit(X_train_pca, y_train)





    ###############################################################################
    # Quantitative evaluation of the model quality on the test set
    #print ("Predicting clothe names on the testing set")
    y_pred = clf.predict(test_pca)

    #sys.stdout = sys.stderr = open(os.devnull, "w")
    #print("y_pred : " +str(y_pred))
    #sys.stdout = saved_stdout
    #print ("Computed in %0.3fs" % (time() - t0))

    #os.system('mkdir final_images')
    os.system('cp ./clothe_test/0.jpg ./final_images/')

    hasher = hashlib.md5()
    with open('./final_images/0.jpg', 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)

    os.system('mv ./final_images/0.jpg ./final_images/'+hasher.hexdigest()+'.jpg')
    out_image = {}
    out_image['category'] = [str(y_pred)]
    out_image['image'] = ['./final_images/'+hasher.hexdigest()+'.jpg']

    print(json.dumps(out_image))

if __name__ == "__main__":
    main()

    '''
    print (classification_report(y_test, y_pred, target_names=clothe.target_names))
    print (confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    scores += clf.score(X_test_pca, y_test)

    ###############################################################################
    # View results
    prediction_titles = [title(y_pred, y_test, clothe.target_names, i)
                         for i in range(y_pred.shape[0])]
    # eigenclothes_titles = ["eigenclothe %d" % i for i in range(eigenclothes.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)
    # plot_gallery(eigenclothes, eigenclothes_titles, h, w)
    pl.show()


    for train_index, test_index in kf:
        #print(train_index)
        #print(test_index)
        #
        X_train = np.array([X[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_train = np.array([y[i] for i in train_index])
        y_test = np.array([y[i] for i in test_index])

        print("in for : X_train" + str(type(X_train)))
        print("in for : X_train" + str(X_train.shape))

        print("in for : y_train" + str(type(y_train)))
        print("in for : y_train" + str(y_train.shape))

        print("in for : X_test" + str(type(X_test)))
        print("in for : X_test" + str(X_test.shape))

        print("in for : y_test" + str(type(y_test)))
        print("in for : y_test" + str(y_test.shape))


        sys.exit()
        ###############################################################################
        # Apply PCA
        n_components = 18  # 80%
        pca = PCA(n_components=n_components, whiten=True).fit(X_train)
        #eigenclothes = pca.components_.reshape((n_components, h, w))

        #print ("Projecting the input data on the eigenclothe orthonormal basis")
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # reconstruction = pca.inverse_transform(X_train_pca[1])
        # im = Image.fromarray(reconstruction.reshape(h,w))
        # im.show()

        ###############################################################################
        # Train an SVM classification model
        print ("Fitting the classifier to the training set")
        param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }
        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, y_train)

        # print "Fitting the classifier to the training set"
        # param_grid = {
        #         'n_neighbors': list(xrange(1, 15)),
        # }
        # clf = GridSearchCV(KNeighborsClassifier(), param_grid)
        # clf = clf.fit(X_train, y_train)

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set
        print ("Predicting clothe names on the testing set")
        y_pred = clf.predict(X_test_pca)

        print (classification_report(y_test, y_pred, target_names=clothe.target_names))
        print (confusion_matrix(y_test, y_pred, labels=range(n_classes)))
        scores += clf.score(X_test_pca, y_test)

        ###############################################################################
        # View results
        prediction_titles = [title(y_pred, y_test, clothe.target_names, i)
                             for i in range(y_pred.shape[0])]
        #eigenclothes_titles = ["eigenclothe %d" % i for i in range(eigenclothes.shape[0])]

        plot_gallery(X_test, prediction_titles, h, w)
        #plot_gallery(eigenclothes, eigenclothes_titles, h, w)
        pl.show()

    print ("Computed in %0.3fs" % (time() - t0))
    print ('AVG score = %0.3f' % (scores/len(kf)))
    '''

    '''
    # original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            pl.subplot(n_row, n_col, i + 1)
            pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
            pl.title(titles[i], size=12)
            pl.xticks(())
            pl.yticks(())


    # original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
    def title(y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    '''

    '''
        return Bunch(data=clothes.reshape(len(clothes), -1),test_clothe = test_clothe.reshape((1,200,200)), images=clothes,
                     target=target, target_names=target_names,
                     DESCR="Pokemon dataset")
        '''

    '''
        print("in main : X_train" + str(type(X_train)))
        print("in main : X_train" + str(X_train.shape))

        print("in main : y_train" + str(type(y_train)))
        print("in main : y_train" + str(y_train.shape))
        print("testttttt :: " + str(type(test)))
        print("testttttt :: " + str(test.shape))
        '''

    '''
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }

        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, y_train)


        '''
