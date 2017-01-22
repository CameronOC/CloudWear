import os
import sys


def main(argv):
    s = argv[1]
    os.system('./rem_bkg '+ s)
    os.system('cp 0.jpg ../clothe_recognition/clothe_test')
    #os.system('cd ../clothe_recognition/') 
    #os.system('python ../clothe_recognition/main.py >> out_images.json')


if __name__ == '__main__':
    main(sys.argv)


