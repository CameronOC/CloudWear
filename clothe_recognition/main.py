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
            for i, path in enumerate(paths):
                img = Image.open(path).convert('RGB')
                img = ImageOps.fit(img, (w, h), Image.ANTIALIAS, (0.5, 0.5))
                img = ImageOps.grayscale(img)

                new_path = join(CLOTHE_PROC_PATH, clothe_name, str(i)) + '.jpg'
                img.save(new_path)

    if exists(CLOTHE_TEST_PROC_PATH):
        shutil.rmtree(CLOTHE_TEST_PROC_PATH)
    if not exists(CLOTHE_TEST_PROC_PATH):
        makedirs(CLOTHE_TEST_PROC_PATH)



        test_folder = CLOTHE_TEST_PATH
        test_paths = [join(test_folder,f) for f in listdir(test_folder) if f!='.DS_Store']



        for i, test_path in enumerate(test_paths):
            test_img = Image.open(test_path).convert('RGB')
            test_img = ImageOps.fit(test_img,(w,h), Image.ANTIALIAS, (0.5, 0.5))
            test_img = ImageOps.grayscale(test_img)
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

    n_clothe = len(clothe_paths)
    target_names = np.unique(clothe_names)
    target = np.searchsorted(target_names, clothe_names)

    # read data
    clothes = np.zeros((n_clothe, h, w), dtype=np.float32)
    for i, clothe_path in enumerate(clothe_paths):
            img = imread(clothe_path)
            clothe = np.asarray(img, dtype=np.uint32)
            clothes[i, ...] = clothe


    test_img = imread(new_test_paths[0])
    test_clothe = np.asarray(test_img, dtype = np.uint32)



    # shuffle clothe
    indices = np.arange(n_clothe)
    np.random.RandomState(42).shuffle(indices)
    clothes, target = clothes[indices], target[indices]

    return Bunch(data=clothes.reshape(len(clothes), -1), test_clothe=test_clothe,
                 images=clothes,
                 target=target, target_names=target_names,
                 DESCR="Pokemon dataset")


def main():
    t0 = time()
    np.random.seed(3)
    clothe = get_clothe()
    X = clothe.data
    y = clothe.target
    test = clothe.test_clothe

    X_train = np.array([X[i] for i in range(X.shape[0])])
    y_train = np.array([y[i] for i in range(y.shape[0])])
    test = np.array([test])
    test = test.reshape((1,40000))

    # Apply PCA
    n_components = 18  # 80%
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    test_pca = pca.transform(test)


    param_grid = {
        'n_neighbors': list(xrange(1, 15)),
    }

    clf = GridSearchCV(KNeighborsClassifier(), param_grid)
    clf = clf.fit(X_train_pca, y_train)
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
