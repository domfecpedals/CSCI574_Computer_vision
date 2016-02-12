import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotObject():
    # result coordinates from the openCV program
    x=np.array([6.980295287238382, 7.370532315229015, -7.522391332331264, -7.833903258804979, 7.443727130264914, 7.913331234130481, -6.982199489363383, -7.369391886363174])
    y=np.array([-3.593939139491011, 5.39927523426932, 5.756593185408509, -3.444498042984034, -5.567604838717801, 3.423315248904341, 3.677127319779906, -5.650268967169233])
    z=np.array([4.627164749097506, 2.961825591230027, 2.884070077124143, 4.467784989202845, -2.61421804849612, -4.380421243641753, -4.828000785343495, -3.118205329173152])

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    # Plot 8 corners
    ax.scatter(x,y,z, c='r', marker='*')
    # Connect adjuscent pairs
    ax.plot(x[4:],y[4:],z[4:])
    ax.plot(x[:4],y[:4],z[:4])
    
    # Print out the direction vector of each edge
    print x[1]-x[0],y[1]-y[0],z[1]-z[0]
    print x[2]-x[3],y[2]-y[3],z[2]-z[3]
    print x[2]-x[1],y[2]-y[1],z[2]-z[1]
    print x[3]-x[0],y[3]-y[0],z[3]-z[0]

    print x[5]-x[4],y[5]-y[4],z[5]-z[4]
    print x[6]-x[7],y[6]-y[7],z[6]-z[7]
    print x[6]-x[5],y[6]-y[5],z[6]-z[5]
    print x[7]-x[4],y[7]-y[4],z[7]-z[4]

    ax.plot(np.array([x[0],x[3]]),np.array([y[0],y[3]]),np.array([z[0],z[3]]))
    ax.plot(np.array([x[4],x[7]]),np.array([y[4],y[7]]),np.array([z[4],z[7]]))
    
    for idx in range(0,4):
        ax.plot(np.array([x[0+idx],x[4+idx]]),np.array([y[0+idx],y[4+idx]]),np.array([z[0+idx],z[4+idx]]))
        print x[4+idx]-x[0+idx],y[4+idx]-y[0+idx],z[4+idx]-z[0+idx]





    plt.show()

if __name__ == '__main__':
    plotObject()

