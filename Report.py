from sklearn.metrics import classification_report
import numpy as np 

class Report_Generation:

    def __init__(self,pred,true):
        self.pred = pred
        self.true = true
        return None

    def Full_classification_Report(self):
        print(classification_report(np.argmax(self.true,axis=1),np.argmax(self.pred, axis=1).reshape(-1,1)))
        return "Program Completed"