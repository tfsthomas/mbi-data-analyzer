import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

import time
import os

os.chdir('/Users/thomasschmitt/Desktop/MBI')

cwd = os.getcwd()
print(cwd)


##data = np.genfromtxt('170627_Bi111_5_rot000_7.dat',)
####print(data)
##
##
##
##def get_phts(x=None):
##    if (x > 0.10) and (x < 0.125):
##        return 0
##    elif (x > 0.125) and (x <0.16):
##        return 1
##    elif (x>.16) and (x<.19):
##        return 2
##    elif (x >.19) and (x<.22):
##        return 3
##    elif (x >.22) and (x<.25):
##        return 4
##    else:
##        return 0
##
##result_arr = np.zeros((50,5000))
##
##for iii in range(50):
##    for jjj,el in enumerate(data[iii,1:5000]):
##        result_arr[iii,jjj] = get_phts(x=el)
##
##def open_closed (y=None):
##    if (y>2.5):
##        return 200
##    if (y<2.5):
##        return 100
##
##return_arr = np.zeros((50,5000))
##
##for aaa in range(50):
##    for bbb, el in enumerate(data[aaa,5001:10001]):
##        return_arr[aaa,bbb] = open_closed(y=el)
##
##def analyse_delay(n=None):
##    return np.array([data[n,0],np.sum(np.column_stack((return_arr[n,:],result_arr[n,:]))[np.argsort(np.column_stack((return_arr[n,:],result_arr[n,:]))[:,0]),:][2500:5000,:],axis=0)[:][1],np.sum(np.column_stack((return_arr[n,:],result_arr[n,:]))[np.argsort(np.column_stack((return_arr[n,:],result_arr[n,:]))[:,0]),:][0:2500,:],axis=0)[:][1]]).astype('int')
##
##
##
##final_arr = np.zeros((50,3)).astype('int')
##
##for yyy in range(50):
##    final_arr[yyy] = analyse_delay(n=yyy)
##
##final_arr = np.array(final_arr)


Npts = 40


final_arr = np.loadtxt('clean_data.dat')


finalsorted_arr = final_arr[np.argsort(final_arr[:,0]),:]

finalsplit_arr = np.array_split(finalsorted_arr, Npts)

##finalsplit1_arr = np.mean(finalsplit_arr[0],axis=0).astype('int')

def group_delay(p=None):
    delay_avg = np.mean(finalsplit_arr[p],axis=0)
##    delay_err = np.std(finalsplit_arr[p],axis=0)
    phts_sum = np.sum(finalsplit_arr[p],axis=0).astype('int')
    dii = 2*(finalsplit_arr[p][:,1]-finalsplit_arr[p][:,2])/(finalsplit_arr[p][:,1]+finalsplit_arr[p][:,2])
##    print(len(finalsplit_arr[p][:,1]))
##    return np.array([delay_avg[0],phts_sum[1],phts_sum[2]])
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

##hist, bin_edges = np.histogram(data[0,1:5000],bins = 1024, range = (group_min, group_max))
##print(hist)                               
##print (bin_edges)
plt.ion()

fig1 = plt.figure(1, figsize=(9.6,8))
fig1.clf()
ax1 = plt.subplot2grid((2, 1), (0, 0))  #rowspan=5
ax2 = plt.subplot2grid((2, 1), (1, 0))  #rowspan=5
ax1.plot(-(group_arr2[:,0]-1673770),group_arr2[:,1] , 'b-')
ax1.plot(-(group_arr2[:,0]-1673770),group_arr2[:,2] , 'r-')
####ax2.plot(data2, 'ro')

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
##ax2.plot(delay,1/np.sqrt((group_arr[:,1]+group_arr[:,2])) , 'r-')



##1.67377*e6 = time_delay_zero


plt.tight_layout(pad=0.0)

##
##def func(x , a, x0, w):
##    return a * (-(x-x0)/w)
##
##xdata = np.arange(-3000,5000,500)
##ydata = func(xdata, -60,0, 250)
##ax2.plot(xdata, (ydata/10000)-.28, color='orange',ls='-')

##popt, pcov = curve_fit(func, -(group_arr[:,0]-1673770), -(2*((group_arr[:,1]-group_arr[:,2])/(group_arr[:,2]+group_arr[:,1]))))
##ax2.plot(-(group_arr[:,0]-1673770), func(-(group_arr[:,0]-1673770), *popt),'r-')


##def func2(x, a,x0, b, c):
##    return a * erf((x-x0)/b)+c
##
##xinput = np.arange(-3000, 5000, 10)
##yinput = func2(xinput, -1600, 300,200,500)
##ax2.plot(xinput,(yinput/10000)-.21, color='green')

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
##plt.errorbar(delay, dIoverI, yerr=dIIerr,fmt='o')



print('Optimized parameters are ',popt)
print('Errors are',perr )





##ax2.plot(xinput, 0.5*(np.sign(-(group_arr[:,0]-1673770))+1), color='yellow')

##def heaviside(x):
##    if x==0:
##        return 0
##    return 0 if x<0 else (func3(x, -1600, 300,200,500, .25)/10000)-.16
##
##yinput_arr = np.zeros((800))
##
##for ddd in range (-3000, 5000, 10):
##    yinput_arr[ddd] = heaviside(x=ddd)
##
##.5 * (np.sign(x-x0)*1)
##
##ax2.plot(xinput, yinput_arr, color='brown')    

##plt.show()
##
##S*(-(x-x0)/w
##x,s,x0,w, bg, a
##w=250


##plt.ion()
##
##fig1 = plt.figure(1, figsize=(9.6,8))
##fig1.clf()
##ax1 = plt.subplot2grid((2, 1), (0, 0))  #rowspan=5
##ax2 = plt.subplot2grid((2, 1), (1, 0))  #rowspan=5
##
##data1 = np.arange(1,5,0.1)
##data2 = np.arange(0,5,1)
##ax1.plot(data1, 'bo')
##ax2.plot(data2, 'ro')
##
##plt.tight_layout(pad=0.0)
##plt.show()
##
##
##for nnn in range (51):
##    for ooo, el in enumerate(data [nnn, 5001:10001]):
##        final_arr[nnn,ooo] = analyse_delay(n=el)


##row1_arr = np.column_stack((return_arr[0,:],result_arr[0,:]))    
##row2_arr = np.column_stack((return_arr[1,:],result_arr[1,:]))
##row3_arr = np.column_stack((return_arr[2,:],result_arr[2,:]))
##
##row1sort_arr = row1_arr[np.argsort(row1_arr[:,0]),:]
##row2sort_arr = row2_arr[np.argsort(row2_arr[:,0]),:]
##
##row1closed_arr = row1sort_arr[0:2500,:]
##row1open_arr = row1sort_arr[2500:5000,:]
##row2closed_arr = row2sort_arr[0:2500,:]
##row2open_arr = row2sort_arr[2500:5000,:]
##
##row1closedphts_arr = np.sum(row1closed_arr,axis=0)
##row1openphts_arr = np.sum(row1open_arr,axis=0)
##row2closedphts_arr = np.sum(row2closed_arr,axis=0)
##row2openphts_arr = np.sum(row2open_arr,axis=0)
##
##row1unpumped = row1closedphts_arr[:][1]
##row1pumped = row1openphts_arr[:][1]
##row2unpumped = row2closedphts_arr[:][1]
##row2pumped = row2openphts_arr[:][1]
##
##
##row1 = np.array([data[0,0],row1pumped,row1unpumped]).astype('int')
##row2 = np.array([data[1,0],row2pumped,row2unpumped]).astype('int')
##
##results = np.vstack((row1,row2))


##
##row1_arr = np.column_stack((return_arr[n,:],result_arr[n,:]))
##    row1sort_arr = row1_arr[np.argsort(row1_arr[:,0]),:]
##    row1closed_arr = row1sort_arr[0:2500,:]
##    row1open_arr = row1sort_arr[2500:5000,:]
##    row1closedphts_arr = np.sum(row1closed_arr,axis=0)
##    row1openphts_arr = np.sum(row1open_arr,axis=0)
##    row1unpumped = row1closedphts_arr[:][1]
##    row1pumped = row1openphts_arr[:][1]
##    row1 = np.array([data[n,0],np.sum(row1_arr[np.argsort(np.column_stack((return_arr[n,:],result_arr[n,:]))[:,0]),:][2500:5000,:],axis=0)[:][1],np.sum(row1_arr[np.argsort(np.column_stack((return_arr[n,:],result_arr[n,:]))[:,0]),:][0:2500,:],axis=0)[:][1]]).astype('int')
##    np.vstack((row1))
##    (
