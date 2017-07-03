import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

import time
import os

os.chdir('/Users/thomasschmitt/Desktop/MBI')

cwd = os.getcwd()
print(cwd)



Npts = 40


final_arr = np.loadtxt('clean_data.dat')


finalsorted_arr = final_arr[np.argsort(final_arr[:,0]),:]

finalsplit_arr = np.array_split(finalsorted_arr, Npts)

##finalsplit1_arr = np.mean(finalsplit_arr[0],axis=0).astype('int')

def group_delay(p=None):
    delay_avg = np.mean(finalsplit_arr[p],axis=0)
    phts_sum = np.sum(finalsplit_arr[p],axis=0).astype('int')
    dii = 2*(finalsplit_arr[p][:,1]-finalsplit_arr[p][:,2])/(finalsplit_arr[p][:,1]+finalsplit_arr[p][:,2])
    return np.array([delay_avg[0],dii.mean(),dii.var()**0.5/(len(dii)**0.5)])
    
def group_delay2(g=None):
    delay_avg2 = np.mean(finalsplit_arr[g],axis=0)
    phts_sum2 = np.sum(finalsplit_arr[g],axis=0).astype('int')
    return np.array([delay_avg2[0],phts_sum2[1],phts_sum2[2]])
    
group_arr = np.zeros((Npts,3))
group_arr2 = np.zeros((Npts,3))

for vvv in range (Npts):
    group_arr[vvv] = group_delay(p=vvv)
    
for kkk in range (Npts):
    group_arr2[kkk] = group_delay2(g=kkk)


group_min = 1665000
group_max = 1680000

plt.ion()

fig1 = plt.figure(1, figsize=(9.6,8))
fig1.clf()
ax1 = plt.subplot2grid((2, 1), (0, 0))  #rowspan=5
ax2 = plt.subplot2grid((2, 1), (1, 0))  #rowspan=5
ax1.plot(-(group_arr2[:,0]-1673770),group_arr2[:,1] , 'b-')
ax1.plot(-(group_arr2[:,0]-1673770),group_arr2[:,2] , 'r-')


totaldIoverI = -(2*((finalsorted_arr[:,1]-finalsorted_arr[:,2])/(finalsorted_arr[:,2]+finalsorted_arr[:,1])))
sortedtotaldIoverI = np.sort(totaldIoverI)
splitdIoverI = np.array_split(sortedtotaldIoverI, Npts)

def split_array(e=None):
    split_avg = np.mean(splitdIoverI[e], axis=0)
    return np.array(split_avg)
 
split_arr = np.zeros((Npts,))

for uuu in range (Npts):
    split_arr[uuu] = split_array(e=uuu)

dIoverI = -(2*((group_arr2[:,1]-group_arr2[:,2])/(group_arr2[:,2]+group_arr2[:,1])))
delay =  -(group_arr[:,0]-1673770)

ax2.plot(delay,dIoverI , 'b-')



plt.tight_layout(pad=0.0)


def func3(x, a, x0, b, c, d):
    return a * (erf((x-x0)/b)+1)/2 + c + d*(x-x0)*(.5 * (np.sign(x-x0)+1))

p0 =  [-0.15, 300,200,0, .000025]

popt, pcov = curve_fit(func3,delay, dIoverI,p0=p0)
perr = np.sqrt(np.diag(pcov))

xinput = np.arange(-3000, 5000, 10)
yinput = func3(xinput,*popt)
ax2.plot(xinput,yinput, color='red',ls='--')





dIIerr = (np.std(split_arr))/(np.sqrt(len(split_arr)))


ax2.errorbar(-(group_arr[:,0]-1.67377*10**6), -group_arr[:,1], yerr=group_arr[:,2],fmt='o')




print('Optimized parameters are ',popt)
print('Errors are',perr )




