import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 


np.random.seed(20)
x=5*np.random.random(100)
y=2+4*x + np.random.randn(100)

# re=2+4*x
# plt.figure(figsize=(10,5))  
# plt.scatter(x,y)
# plt.plot(x,re,color='r')

b=np.arange(-15,15,0.5)
m=np.arange(-2.5,11.5,0.5)
# def mse(x,y):
#     # b = b[:, np.newaxis]
#     mse_b=1/n*np.sum((y-(b.reshape(-1,1)+4*x))**2,axis=1)
#     mse_m=1/n*np.sum((y-(2+m.reshape(-1,1)*x))**2,axis=1)
#     return [mse_b,mse_m]
# mse_b=mse(x,y)[0]
# mse_m=mse(x,y)[1]

# ax1,ax2=plt.figure(figsize=(10,5)).subplots(1,2)
# ax1.set_title("MSE vs Intercept(b)")
# ax2.set_title("MSE vs Slope(m)")
# ax1.set_xlabel("Intercept(b)")
# ax2.set_xlabel('Slope "m"')
# ax1.plot(b,mse_b)
# ax2.plot(m,mse_m)
# plt.show()


# from sklearn.linear_model import LinearRegression
# md=LinearRegression()
# md.fit(x,y)
# print(md.coef_,md.intercept_,sep='\n')

# from scipy.stats import linregress
# r=linregress(x[:,0],y[:,0])
# print(r.slope,r.intercept,sep='\n')



b2,m2=np.meshgrid(b,m)

n=len(x)

mse=np.zeros([len(m),len(b)])

for i in range(len(m)):
    for j in range(len(b)):
        y_hat=b2[i,j] + m2[i,j]*x
        mse[i,j]=1/n*np.sum((y - y_hat)**2)

min_val=np.argmin(mse)
r,c=np.unravel_index(min_val,mse.shape)

print(f"Lowest MSE loss is: ",{np.min(mse)},"Best Slope: ", {m2[r][c]},"Best Intercept: ",{b2[r][c]},sep='\n')
# OUTPUT:
# Lowest MSE loss is:
# {1.1179640956873431}
# Best Slope:
# {4.0}
# Best Intercept:
# {2.0}

#--------PLOT OPTIMAL COEFFICIENTS---------
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
surf=ax.plot_surface(m2,b2,mse,cmap='gnuplot2')
ax.scatter(m2[r,c],b2[r,c],np.min(mse),color='r',marker='x')
ax.set_xlabel("Slope")
ax.set_ylabel("Intercept")
plt.title('MSE as Loss Function')

fig.colorbar(surf,shrink=0.8)
plt.show()


#-----------------PART II : PLOT GRADIENT --------------------------

#---------------------------------------------------------------#
def f(m,b):
    points=b+m*x
    loss=1/n*np.sum((y-points)**2)
    return loss
#---------------------------------------------------------------#
def grad(m,b):
    der_m = -2/n*np.sum(x*(y-(b+m*x)))
    der_b = -2/n*np.sum(y-(b+m*x))
    grad=np.array([der_m,der_b])
    return grad
#--------------------------------------------------------------#
l_rate=0.01
start=np.array([8,8,f(8,8)])
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)

for i in range(1000):
    m_new= start[0] - l_rate*grad(start[0],start[1])[0]
    b_new= start[1] - l_rate*grad(start[0],start[1])[1]
    start=np.array([m_new,b_new,f(m_new,b_new)])
    
    surf=ax.plot_surface(m2,b2,mse,cmap='gnuplot2')
    ax.scatter(m2[r,c],b2[r,c],np.min(mse),color='r',marker='x',label='Lowest Point')
    ax.scatter(start[0],start[1],start[2],color='g',marker='o',label='Gradient moving')
    plt.legend()
    
    plt.pause(0.001)
    ax.clear()

print(f"Now m is: ",{start[0]},"b is: ",{start[1]}, "and final loss: ",{start[2]},sep="\n")








