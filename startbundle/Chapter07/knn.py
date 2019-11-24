
#useage: python knn.py --dataset ../dataset/animals

#import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import sys
sys.path.append("..")
from startbundle.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from startbundle.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor

def knn(dataset,neighbors,jobs):
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset))


    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    '''
    3072=32*32*3(RGB)
    '''
    data = data.reshape((data.shape[0], 3072))

    print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("[INFO evaluating k-NN classifier...]")
    model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
    model.fit(trainX, trainY)
    print(classification_report(testY, model.predict(testX), target_names=le.classes_))
    print("Done.")

if __name__ == '__main__':
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of jobs for k-NN distance (-1 uses all avilable cores)")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    neighbors = args["neighbors"]
    jobs = args["jobs"]
    '''
    '''
    windows 用\\，linux用/
    '''
    dataset = 'c:\\kuaipan\\cv-data\\dogs-vs-cats\\'
    neighbors = 1
    jobs = 1

    knn(dataset,neighbors,jobs)

