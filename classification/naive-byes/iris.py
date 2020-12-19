from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def naive_bayes(x, y):
  # split data to training set and testing set
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

  gnb = train(x, y)
  
  y_pred = predict(gnb, x_test)
  
  print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))

def train(x, y):
  gnb = GaussianNB()

  gnb.fit(x, y)

  return gnb

def predict(gnb, x):
  return gnb.predict(x)

if __name__ == "__main__":
  iris = load_iris()

  # iris.data contains features
  x = iris.data
  
  # iris.target contains labels
  y = iris.target

  naive_bayes(x, y)