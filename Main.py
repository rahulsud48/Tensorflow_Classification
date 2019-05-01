from Data import images, Preprocessing
from TF_Model import CNN_Model
from Report import Report_Generation
import sys

#DataSet Path
path = 'data/Images'

#Reading the Images and Classes
Img = images(path)
DataSet = Img.read_images()
Prep = Preprocessing(DataSet)
X_train, X_test, y_Train, y_test = Prep.train_test()

#TensorFlow CNN Model
Model = CNN_Model(X_train, y_Train, X_test, y_test)
y_pred,y_true =  Model.Training()

#Classification Report
classification_report = Report_Generation(y_pred,y_true)
message = classification_report.Full_classification_Report()
print(message)