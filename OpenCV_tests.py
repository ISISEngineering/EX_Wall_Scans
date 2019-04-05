import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

guage=[300,300]
pinDia=300
omega=0    #angle
Diagonal=int(2*np.ceil((np.sqrt(guage[0]**2+guage[1]**2))/2)) ## make even number
mu=0.005  #absorption


#Import image
Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pinDia,pinDia))
Rect = cv2.getStructuringElement(cv2.MORPH_RECT,(guage[0],guage[1]))

MtransEll = np.float32([[1,0,guage[0]+pinDia/2],[0,1,pinDia/2]])
transEll = cv2.warpAffine(Ellipse,MtransEll,(pinDia*2+guage[0]*2,pinDia*2))

MtransRect = np.float32([[1,0,(Diagonal-guage[0])/2],[0,1,(Diagonal-guage[1])/2]])
Rect = cv2.warpAffine(Rect,MtransRect,(Diagonal,Diagonal))

Mrot = cv2.getRotationMatrix2D((int(Diagonal/2),int(Diagonal/2)),omega,1)
Rect = cv2.warpAffine(Rect,Mrot,(Diagonal,Diagonal))


A=np.zeros(np.shape(transEll))

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i,j]=np.sum(transEll[:i,j])
AbsArray=np.exp(-mu*A)

AbsEll=np.multiply(transEll,AbsArray)



conv2 = signal.fftconvolve(Rect,AbsEll)
entryCurve0=np.sum(conv2, axis=0) # X-scan
entryCurve1=np.sum(conv2, axis=1) # Y-scan



#Show the image with matplotlib
fig, axs = plt.subplots(3, 2,figsize=(20,20))

axs[0,0].imshow(Rect);axs[0,1].imshow(transEll);

axs[1,0].contourf(AbsEll,cmap='jet'); axs[1,1].contourf(conv2,cmap='jet')

axs[2,0].plot(entryCurve0);axs[2,1].plot(entryCurve1)
plt.show()