#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


###### PART 1 ############
#__________________________

x = np.array([1,2])
y = np.array([-2,1])
a = np.dot(x,y)
print(a)

b = np.array([1,2])
c = np.sqrt(x[0]**2+x[1]**2)
print(b,c)


# In[3]:


theta = np.arccos( np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(x)) )
print(theta*180/np.pi)


# In[4]:


B = np.array([[3,2,1],[2,6,5],[1,5,9]], dtype=float)
print(B)
print(B - B.T)


# In[5]:


z = np.random.rand(3)
v = B @ z
print(v.shape)
print(v)

print(z.T @ B @ z)


# In[6]:


print(np.trace(B))
print(np.linalg.det(B))
print(np.linalg.inv(B) @ B)


# In[7]:


D, U = np.linalg.eig(B)
print(D)
print(U)
print(np.dot(U[:,0], U[:,1]))
print(U @ U.T) # U * U^T is identity matrix


# In[8]:


U[:,0]


# In[9]:


###### PART 2 ############
#__________________________

x = np.random.rand(10000,1)
plt.figure(figsize=(3,3))
n, bins, patches = plt.hist(x, bins=100, color="r", alpha=0.7, rwidth=0.8)
print(n)
print(bins)

ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.savefig("part2-uniform-100bins.svg", format="svg")


# In[10]:


# Histogram of random numbers following Gaussian distribution
MaxTrials = 10
NumSamples = 200
NumBins = 20
for trial in range(MaxTrials):
    x = np.random.randn(1000,1)
    counts,bins,patches = plt.hist(x, NumBins)
    plt.clf
    print("Variation within bin counts: ", np.var(counts/NumSamples))
    

ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.savefig("part2-normal2-20bins.svg", format="svg")


# In[11]:


N = 30000
x1 = np.zeros(N)
for n in range(N):
    x1[n] = np.sum(np.random.rand(2,1)) - np.sum(np.random.rand(120,1))
plt.hist(x1,40,color="b",alpha=0.8,rwidth=0.8)
plt.xlabel("Bin", FontSize=16)
plt.ylabel("Counts", FontSize=16)
plt.grid(True)
plt.title("Histogram", FontSize=16)
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


plt.savefig("part2-diffofuniform-2-120.svg", format="svg")


# In[12]:


sampleSizeRange = np.linspace(100,200,40)
plotVar = np.zeros(len(sampleSizeRange))
for sSize in range(len(sampleSizeRange)):
    numSamples = np.int(sampleSizeRange[sSize])
    MaxTrial = 2000
    vStrial = np.zeros(MaxTrial)
    for trial in range(MaxTrial):
        xx = np.random.randn(numSamples,1)
        vStrial[trial] = np.var(xx)
    plotVar[sSize] = np.var(vStrial)
    
plt.plot(sampleSizeRange, plotVar, c="g")
plt.xlabel("Sample Size", FontSize=16)
plt.ylabel("Variance", FontSize=16)
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig("part3-line.svg", format="svg")


# In[13]:


def gauss2D(x,m,C):
    Ci = np.linalg.inv(C)
    dC = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x-m).T, np.dot(Ci, (x-m))))
    den = 2 * np.pi * dC
    
    return num/den

def twoDGaussianPlot(nx,ny,m,C):
    x = np.linspace(-5,5,nx)
    y = np.linspace(-5,5,ny)
    X,Y = np.meshgrid(x,y,indexing="ij")
    
    Z = np.zeros([nx,ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec,m,C)
    return X,Y,Z

nx, ny = 50,40
plt.figure(figsize=(6,6))

m1 = np.array([0,2])
C1 = np.array([[2,1],[1,2]], np.float32)
Xp,Yp,Zp = twoDGaussianPlot(nx,ny,m1,C1)
plt.contour(Xp,Yp,Zp,3)
plt.grid(True)

m2 = np.array([2,0])
C2 = np.array([[2,-1],[-1,2]], np.float32)
Xp2,Yp2,Zp2 = twoDGaussianPlot(nx,ny,m2,C2)
plt.contour(Xp2,Yp2,Zp2,3)

m3 = np.array([-2,-2])
C3 = np.array([[2,0],[0,2]], np.float32)
Xp3,Yp3,Zp3 = twoDGaussianPlot(nx,ny,m3,C3)
plt.contour(Xp3,Yp3,Zp3,3)

ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig("part4-contour.svg", format="svg")


# In[14]:


C=[[2,1],[1,2]]
print("C=",C)
A = np.linalg.cholesky(C)
print("A=\n",A)
print("A*A.T=\n",A @ A.T)

X = np.random.randn(10000,2)
Y = X @ A.T
print("X.shape=",X.shape)
print("Y.shape=",Y.shape)

plt.scatter(Y[:,0], Y[:,1], s=3,c="m")
plt.scatter(X[:,0], X[:,1], s=3,c="c")
plt.grid(True)
plt.title("Scatter of Isotropic and Correlated Gaussian Densities")
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig("part5-scatter.svg", format="svg")


# In[15]:


theta = np.pi/3
u =[round(np.cos(theta),4),np.sin(theta)]
print("The vector:",u)
print("Sum of squares:", round(u[0]**2 +u[1]**2))
print("Degrees:",round(theta*180/np.pi))

yp = Y @ u
print(yp.shape)
print("Projected Variance:",np.var(yp))


# In[33]:


nPoints = 160
pVars = np.zeros(nPoints)
thRange = np.linspace(0, 2*np.pi, nPoints)
for n in range(nPoints):
    theta = thRange[n]
    u = [np.cos(theta), np.sin(theta)]
    pVars[n] = np.var(Y @ u)
    

plt.plot(pVars,c="#c20c06",alpha=0.9)

xlabels =["0",r"$\frac{\pi}{4}$",r"$\frac{\pi}{2}$",r"$\frac{3\pi}{4}$",r"$\pi$",r"$\frac{5\pi}{4}$",r"$\frac{3\pi}{2}$",r"$\frac{7\pi}{4}$",r"$2\pi$"]
plt.xticks(np.arange(0, nPoints+1, nPoints/8),xlabels,fontsize=14)
#plt.yticks(np.arange(1, 3+0.25, 0.25))
plt.grid(True)
plt.xlabel(r"$\theta$ (Radians)", fontsize=14)
plt.ylabel("Variance of Projections", fontsize=16)

ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.gcf().subplots_adjust(bottom=0.15)
#plt.savefig("part6-sine.svg", format="svg")


# In[17]:


D1, U1 = np.linalg.eig(C)
print(D1)
print(U1)


# In[18]:


C2 = [[2,-1],[-1,2]]
A2 = np.linalg.cholesky(C2)
print("A2=\n",A2)
print("A2*A2.T=\n",A2 @ A2.T)

X2 = np.random.randn(10000,2)
Y2 = X2 @ A2
print("X2.shape=",X2.shape)
print("Y2.shape=",Y2.shape)

nPoints = 50
pVars = np.zeros(nPoints)
thRange = np.linspace(0, 2*np.pi, nPoints)
for n in range(nPoints):
    theta = thRange[n]
    u = [np.sin(theta), np.cos(theta)]
    pVars[n] = np.var(Y2 @ u)
    
plt.plot(pVars,c="r",alpha=0.9)
plt.grid(True)
plt.xlabel("Direction", fontsize=14)
plt.ylabel("Variance of Projections", fontsize=14)

ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig("part6-sine2.svg", format="svg")


# In[19]:


D2, U2 = np.linalg.eig(C2)
print(D2)
print(U2)


# In[ ]:




