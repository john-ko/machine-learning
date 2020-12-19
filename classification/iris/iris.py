from  sklearn import  datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score

def classify():
  iris = datasets.load_iris()

  # iris.data contains features
  x = iris.data
  
  # iris.target contains labels
  y = iris.target

  # split data to training set and testing set
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

  # set the classifier
  classifier=tree.DecisionTreeClassifier()
  # classifier=neighbors.KNeighborsClassifier()

  # train the model
  classifier.fit(x_train, y_train)

  # predict
  predictions=classifier.predict(x_test)

  # check prediction score
  print(accuracy_score(y_test,predictions))


if __name__ == "__main__":
  classify()
