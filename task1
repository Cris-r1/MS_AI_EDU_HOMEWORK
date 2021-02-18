import numpy as np
import matplotlib.pyplot as plt

def standardize(X):
    m,n=X.shape
    values={}  # 保存每一列的mean和std，便于对预测数据进行标准化
    for i in range(n):
        features=X[:,i]
        meanVal=features.mean(axis=0)
        stdVal=features.std(axis=0)
        values[i]=[meanVal, stdVal]
        if stdVal!=0:
            X[:,i]=(features - meanVal)/stdVal
        else:
            X[:,i]=0
    return X,values
    
def F(W,X):
    return np.dot(X,W)

def J(W,X,Y):
    m=len(X)
    return np.sum(np.dot((F(W,X)-Y).T,F(W,X)-Y)/(2*m))

def BGD(X,Y,step,maxloop,eps):
    m,n=X.shape   #m个样本，n=3
    W=np.zeros((n,1))
    count=0       #迭代次数
    cost=np.inf   #代价
    cost_s=[J(W, X, Y),] #储存cost更新过程
    W_s={}#储存W更新过程
    for i in range(n):
        W_s[i]=[W[i,0],]
    
    while(count<=maxloop):
        count+=1
        W=W-step*1.0/m*np.dot(X.T,F(W,X)-Y)
        
        for i in range(n):
            W_s[i].append(W[i,0])
        cost=J(W,X,Y)
        cost_s.append(cost)
        if(abs(cost_s[-1]-cost_s[-2])<eps):
            break
    
    return W,W_s,cost_s

data=np.loadtxt('mlm.csv',delimiter=',',skiprows=1)
originX=data[:,:2]
Y=data[:,2:]
m,n=originX.shape
X,values=standardize(originX.copy())
X=np.concatenate((np.ones((m,1)), X), axis=1)

step=1
maxloop=100000 #最大迭代次数
eps=0.001      #收敛边界
result=BGD(X,Y,step,maxloop,eps)
W,W_s,cost_s=result
print(cost_s[-1])

ax = plt.gca(projection='3d')
point_x = data[:, 0]
point_y = data[:, 1]
point_z = data[:, 2]
xa = np.linspace(0, 100, 5)
ya = np.linspace(0, 100, 5)
sur_x, sur_y = np.meshgrid(xa, ya)
ax.plot_surface(sur_x, sur_y, sol.W[0]+sur_x*sol.W[1]+sur_y*sol.W[2], color='yellow', alpha=0.6)
ax.scatter3D(point_x, point_y, point_z, cmap='Blues')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
