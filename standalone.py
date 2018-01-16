from sklearn.externals import joblib
import numpy as np # numerical python lib
from numpy import genfromtxt # get text from file lib
import platform # check for 64-bit

if platform.architecture()[0] != '64bit':
    print()
    print("Please run with Python 3.6.4 on a 64 bit machine!")
    print()
    quit()

def read_in_file(s_path):
    """
     Args:
        s_path: string with full path to be readin        
     Return:
        numpy array with readin data
    """
    if len(s_path)==0:
        return 0    
    with open(s_path, 'r') as csvfile:
        return genfromtxt(s_path, delimiter=' ')
    
def get_test_data():
    """
     Return:
        numpy arrays with readin data (test data)
    """
    m_X_test = read_in_file('comp_testX.dat')
    return m_X_test
   
best_est = joblib.load('best_est.pkl')
m_X_test = get_test_data() # Call the function to get the data
a_y = best_est.predict(m_X_test)
np.savetxt('comp_testY.dat', a_y, fmt='%.5f')
print("Written to file 'comp_testY.dat'.")
print(a_y)