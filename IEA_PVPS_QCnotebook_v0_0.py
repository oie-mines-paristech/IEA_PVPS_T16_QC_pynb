# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
#%%
import tabulate
table0 = [["Sun",696000,1989100000],
         ["Earth",6371,5973.6],
         ["Moon",1737,73.5],
         ["Mars",3390,641.85]]
table1 = [
    ['field-name','time','time_bnds','Lat','Lon','elevation','BNI','DHI','GHI'],
    ['long_name',"Time of the end of period","Integration period",'Latitude','Longitude','Elevation above mean sea level', 'Beam Normal Irradiance',"Diffuse Horizontal Irradiance","Global Horizontal Irradiance"],
    ['standard_name',"Time","-",'Latitude','Longitude','Elevation above mean sea level','surface_direct_downwelling_ shortwave_flux_in_air',"diffuse_downwelling_ shortwave_flux_in_air","surface_downwelling_ shortwave_flux_in_air"],
    ['units',"days since 1970-01-01","-","degrees_north","degrees_east","m",'W/m^2',"W/m^2","W/m^2"],
    ['dimensions','(T)','(T)','(Y)','(X)','(Z)','(T,Y,X,Z)',"(T,Y,X,Z)","(T,Y,X,Z)"],
    ['missing_value','n.a.','n.a.','n.a.','n.a.','n.a.','NaN','NaN','NaN'],
    ['instrument','n.a.','n.a.','n.a.','n.a.','n.a.','instrument3','instrument2','instrument1'],
    ['comment','-','-','-','-','-','BNI set to 0 for SZA >= 97 deg','DHI set to 0 for SZA >= 97 deg','GHI set to 0 for SZA >= 97 deg']]
table2=[
    ['field-name','instrument1','instrument1_calibration','instrument2','instrument2_calibration','instrument3','instrument3_calibration'],
    ['long_name','Thermopile pyranometer',"n.a.","Thermopile pyranometer with a shadding ball","n.a.","Thermopile pyrheliometer","n.a."],
    ['text',"-","-","-","-","-","-"],
    ['calibration',"instrument1_calibration","n.a.","instrument2_calibration","n.a.","instrument3_calibration","n.a."],
    ['precision',"","n.a.","","n.a.","","n.a."],
    ['zenithDegr',"","n.a.","","n.a.","","n.a."],
    ['azimuthDegr',"","n.a.","","n.a.","","n.a."],
    ['date',"n.a.","","n.a.","","n.a.",""],
    ['responsivity',"n.a.","","n.a.","","n.a.",""],
    ['unit',"n.a.","","n.a.","","n.a.",""],
    ['comment',"","","","","",""],
    ]
print(tabulate.tabulate(table2, tablefmt='html'))

#%% Import of the solar radiation measurements from the Thredds data server
import netCDF4 as nc
import time as timelib
import datetime as dt
import pandas as pd
import numpy as np

dns='BSRN_PAY_2004_2018.nc' # name of the file containing the measurements of the station of interest
t_start=0                   # first time step considered in the QC
t_end=t_start+3*365*24*60   # last time step of the three year data

startTime = timelib.time()
data_nc = nc.Dataset('http://bsrn:bsrnbsrn@tds.webservice-energy.org/thredds/dodsC/bsrn-stations/'+dns,'r')
lat  = data_nc.variables['lat'][:].data
lon  = data_nc.variables['lon'][:].data
elev = data_nc.variables['elevation'][:].data

# loading and conversion of the time vector 
time0=data_nc.variables['time'][t_start:t_end]
units    = data_nc.variables['time'].units
calendar = data_nc.variables['time'].calendar
ncdate0    = nc.num2date(0,units,calendar=calendar)
date0 = dt.datetime(ncdate0.year,ncdate0.month,ncdate0.day,ncdate0.hour,ncdate0.minute)
ncdate1    = nc.num2date(1,units,calendar=calendar)
date1 = dt.datetime(ncdate1.year,ncdate1.month,ncdate1.day,ncdate1.hour,ncdate1.minute)
unit_deltatime_min=(date1-date0).total_seconds()/60
deltatime_min=(np.round(time0.data[:]*unit_deltatime_min)).astype('timedelta64[m]')
time=(np.datetime64(date0)+deltatime_min).astype(dt.datetime)

# download measurements of the three components of the solar irradiance
GHI = data_nc.variables['GHI'][t_start:t_end].data
DHI = data_nc.variables['DHI'][t_start:t_end]
BNI = data_nc.variables['BNI'][t_start:t_end]

print("Elapsed time is " + str(timelib.time()-startTime) + " seconds.") # 10+15 sec

#%% Calculation of the sun position using the webservice wps_SG2
import sys
sys.path.append('C:\DossierYMSD\jupyter_notebooks\Solar_lab-master\python_functions') # Directory containing le library lib_img_SOMFY.py
import wps as wps
import importlib
importlib.reload(wps)
import pytz

startTime = timelib.time()

location=[lat, lon, elev]
tzinfo=pytz.timezone('utc')
date_begin=dt.datetime(time.min().year,1,1,0,0,tzinfo=tzinfo)
date_end=dt.datetime(time.max().year,12,31,23,59,tzinfo=tzinfo)
SG2=wps.wps_SG2(location,date_begin,date_end,1/60.)

print("Elapsed time is " + str(timelib.time()-startTime) + " seconds.") # 31 sec

#%% Merge BSRN and SG2 datasets in a pandas dataframe 

startTime = timelib.time()

QC_df=pd.DataFrame({'Time': time,'GHI': GHI,'BNI': BNI,'DHI':DHI})
QC_df.set_index('Time',inplace=True)
complete_index=pd.date_range(date_begin,date_end,freq='min',tz='utc')
QC_df.reindex(complete_index)
QC_df=QC_df.merge(SG2, how='left',left_index=True, right_index=True)

print("Elapsed time is " + str(timelib.time()-startTime) + " seconds.") # 17 sec
#%% Visualisation of the content of the dataframe so far
QC_df
#%% Plotting the data
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,8))

ax11 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
QC_df[['TOA','GHI','DHI']].plot(title='Whole period',ax=ax11)
plt.ylabel('W/m2')

ax12 = plt.subplot2grid((2, 3), (0, 2))
QC_df[['TOA','GHI','DHI']].loc[(QC_df.index >dt.datetime(2005,7,1)) & (QC_df.index <= dt.datetime(2005,7,10))].plot(title='10 days in summer',ax=ax12)
plt.ylabel('W/m2')

ax21 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
QC_df[['BNI']].plot(title='Whole period',ax=ax21)
plt.ylabel('W/m2')

ax22 = plt.subplot2grid((2, 3), (1, 2))
QC_df[['BNI']].loc[(QC_df.index >dt.datetime(2005,7,1)) & (QC_df.index <= dt.datetime(2005,7,10))].plot(title='10 days in summer',ax=ax22)
plt.ylabel('W/m2')

plt.tight_layout()

#%% Control of the time system using a two-dimensional time representation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import datetime as dt



# generation of a colormap with light colours for the night
nt_nb=(6,128)
top = cm.get_cmap('Blues', nt_nb[0])
bottom = cm.get_cmap('jet', nt_nb[1])
newcolors = np.vstack((top(np.linspace(0, 1, nt_nb[0])),bottom(np.linspace(0, 1, nt_nb[1]))))
newcmp = ListedColormap(newcolors, name='jet_ymsd')


x_lims = [QC_df.index[0].date(),QC_df.index[-1].date()]
x_lims = mdates.date2num(x_lims)
y_lims = [0, 24]
nb_min=24*60
nb_days=np.int(QC_df.shape[0]/nb_min)

x_timeidx=mdates.date2num(QC_df.index)


fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15,15))

im1=ax1.imshow(np.reshape(QC_df.GHI.values,(nb_days,nb_min)).T,
           extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]],
           aspect='auto',cmap=newcmp)
ax1.xaxis_date()
ax1.set_xlabel('Days')
ax1.set_yticks(np.arange(0,25,3))
ax1.set_ylabel('Time of the day')
ax1.plot(x_timeidx,QC_df.SR_h,'r--')
ax1.plot(x_timeidx,QC_df.SS_h,'r--')
cbar =fig.colorbar(im1, ax=ax1, orientation='vertical')
cbar.set_label('GHI', rotation=270)
ax1.set_title('GHI')


im2=ax2.imshow(np.reshape(QC_df.BNI.values,(nb_days,nb_min)).T,
           extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]],
           aspect='auto',cmap=newcmp)
ax2.xaxis_date()
ax2.set_xlabel('Days')
ax2.set_yticks(np.arange(0,25,3))
ax2.set_ylabel('Time of the day')
ax2.plot(x_timeidx,QC_df.SR_h,'r--')
ax2.plot(x_timeidx,QC_df.SS_h,'r--')
cbar =fig.colorbar(im2, ax=ax2, orientation='vertical')
cbar.set_label('BNI', rotation=270)
ax2.set_title('BNI')


im3=ax3.imshow(np.reshape(QC_df.DHI.values,(nb_days,nb_min)).T,
           extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]],
           aspect='auto',cmap=newcmp)
ax3.xaxis_date()
ax3.set_xlabel('Days')
ax3.set_yticks(np.arange(0,25,3))
ax3.set_ylabel('Time of the day')
cbar =fig.colorbar(im3, ax=ax3, orientation='vertical')
cbar.set_label('DHI', rotation=270)
ax3.set_title('DHI')
ax3.plot(x_timeidx,QC_df.SR_h,'r--')
ax3.plot(x_timeidx,QC_df.SS_h,'r--')

fig.tight_layout()

#%% Analysis of the shadow surrounding the station

vSEA = QC_df.loc[QC_df.GAMMA_S0>1/50,'GAMMA_S0']
vSAA = QC_df.loc[QC_df.GAMMA_S0>1/50,'ALPHA_S']
vBNI = QC_df.loc[QC_df.GAMMA_S0>1/50,'BNI']


idx_sort = np.argsort(vBNI.values)

# call the wps to obtain the horiyon line calculated with SRTM elevation data
HZ = wps.wps_Horizon_SRTM(location)

fig = plt.figure(figsize=(15,8))
plt.scatter(vSAA[idx_sort]*180/np.pi, vSEA[idx_sort]*180/np.pi, s=5, c=vBNI[idx_sort], cmap=newcmp,marker='s',alpha=.5)
plt.title(r'Maximum BNI in (Azimuth, Elevation) $(W.m^{-2})$')
plt.xlabel('Elevqtion [°]')
plt.ylabel('Azimuth [°]')
cb=plt.colorbar()
plt.plot(HZ.AZIMUT,HZ.ELEVATION,'k',linewidth=2)
cb.set_label('BNI $(W.m^{-2})$', rotation=270)
plt.xlim((45,315)) 

#%% QC one component for GHI, DHI and BNI
import matplotlib.pyplot as plt

QCFlag_Test1C_ppl_GHI=(QC_df.GHI <= -4) | (QC_df.GHI > 1.5*QC_df.TOANI*np.sin(QC_df.GAMMA_S0)**1.2 + 100)
QCFlag_Test1C_erl_GHI=(QC_df.GHI <= -2) | (QC_df.GHI > 1.2*QC_df.TOANI*np.sin(QC_df.GAMMA_S0)**1.2 + 50)

QCFlag_Test1C_ppl_DHI=(QC_df.DHI <= -4) | (QC_df.DHI > 0.95*QC_df.TOANI*np.sin(QC_df.GAMMA_S0)**1.2 + 50)
QCFlag_Test1C_erl_DHI=(QC_df.DHI <= -2) | (QC_df.DHI > 0.75*QC_df.TOANI*np.sin(QC_df.GAMMA_S0)**1.2 + 30)

QCFlag_Test1C_ppl_BNI=(QC_df.BNI <= -4) | (QC_df.BNI > QC_df.TOANI)
QCFlag_Test1C_erl_BNI=(QC_df.BNI <= -2) | (QC_df.BNI > 0.95*QC_df.TOANI*np.sin(QC_df.GAMMA_S0)**0.2 + 10)

day = QC_df.index.values.astype('datetime64[D]').astype(QC_df.index.values.dtype)
TOD = (QC_df.index.values - day).astype('timedelta64[s]').astype('double')/60/60

idx_sortTOA = np.argsort(QC_df.TOA.values)

plt.figure(figsize=(15,15))

ax11 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
idxERL=np.where(QCFlag_Test1C_erl_GHI)
plt.plot(day[idxERL],TOD[idxERL],'g.',markersize=0.2,label='ERL flag')
idxPPL=np.where(QCFlag_Test1C_ppl_GHI)
plt.plot(day[idxPPL],TOD[idxPPL],'r.',markersize=2,label='PPL flag')
plt.ylim((0,24))
plt.ylabel('Time of the day [h]')
plt.title("Results of the 1 component QC for GHI")
plt.legend(loc='upper left')

ax12 = plt.subplot2grid((3, 3), (0, 2), colspan=1)
plt.plot(QC_df.TOA,QC_df.GHI, '.',alpha=0.1,markersize=0.5,label='data',color=[0.5,0.5,0.5])
plt.plot(QC_df.TOA[idx_sortTOA],1.5*QC_df.TOANI[idx_sortTOA]*np.sin(QC_df.GAMMA_S0[idx_sortTOA])**1.2 + 100,linewidth=0.5,label='PPL limit')
plt.plot(QC_df.TOA[idx_sortTOA],1.2*QC_df.TOANI[idx_sortTOA]*np.sin(QC_df.GAMMA_S0[idx_sortTOA])**1.2 + 50,linewidth=0.5,label='ERL limit')
plt.plot(QC_df.TOA[QCFlag_Test1C_erl_GHI],QC_df.GHI[QCFlag_Test1C_erl_GHI],'g.',markersize=1,label='ERL flag')
plt.plot(QC_df.TOA[QCFlag_Test1C_ppl_GHI],QC_df.GHI[QCFlag_Test1C_ppl_GHI],'r.',markersize=1,label='PPL flag')
plt.legend(loc='upper left')
plt.xlabel('TOA [W/m2]')
plt.ylabel('GHI [W/m2]')


ax21 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
idxERL=np.where(QCFlag_Test1C_erl_DHI)
plt.plot(day[idxERL],TOD[idxERL],'g.',markersize=0.2,label='ERL flag')
idxPPL=np.where(QCFlag_Test1C_ppl_DHI)
plt.plot(day[idxPPL],TOD[idxPPL],'r.',markersize=2,label='PPL flag')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the 1 component QC for DHI")

ax22 = plt.subplot2grid((3, 3), (1, 2), colspan=1)
plt.plot(QC_df.TOA,QC_df.DHI, '.',alpha=0.1,markersize=0.5,label='data',color=[0.5,0.5,0.5])
plt.plot(QC_df.TOA[idx_sortTOA],0.95*QC_df.TOANI[idx_sortTOA]*np.sin(QC_df.GAMMA_S0[idx_sortTOA])**1.2 + 50,linewidth=0.5,label='PPL')
plt.plot(QC_df.TOA[idx_sortTOA],0.75*QC_df.TOANI[idx_sortTOA]*np.sin(QC_df.GAMMA_S0[idx_sortTOA])**1.2 + 30,linewidth=0.5,label='ERL')
plt.plot(QC_df.TOA[QCFlag_Test1C_erl_DHI],QC_df.DHI[QCFlag_Test1C_erl_DHI],'g.',markersize=1,label='ERL flag')
plt.plot(QC_df.TOA[QCFlag_Test1C_ppl_DHI],QC_df.DHI[QCFlag_Test1C_ppl_DHI],'r.',markersize=1,label='PPL flag')
plt.legend(loc='upper left')
plt.xlabel('TOA [W/m2]')
plt.ylabel('DHI [W/m2]')


ax31 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
idxERL=np.where(QCFlag_Test1C_erl_BNI)
plt.plot(day[idxERL],TOD[idxERL],'g.',markersize=0.2,label='ERL flag')
idxPPL=np.where(QCFlag_Test1C_ppl_BNI)
plt.plot(day[idxPPL],TOD[idxPPL],'r.',markersize=2,label='PPL flag')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the 1 component QC for BNI")

ax32 = plt.subplot2grid((3, 3), (2, 2), colspan=1)
plt.plot(QC_df.TOA,QC_df.BNI, '.',alpha=0.1,markersize=0.5,label='data',color=[0.5,0.5,0.5])
plt.plot(QC_df.TOA[idx_sortTOA],QC_df.TOANI[idx_sortTOA],linewidth=0.5,label='PPL')
plt.plot(QC_df.TOA[idx_sortTOA],0.95*QC_df.TOANI[idx_sortTOA]*np.sin(QC_df.GAMMA_S0[idx_sortTOA])**0.2 + 10,linewidth=0.5,label='ERL')
plt.plot(QC_df.TOA[QCFlag_Test1C_erl_BNI],QC_df.BNI[QCFlag_Test1C_erl_BNI],'g.',markersize=1,label='ERL flag')
plt.plot(QC_df.TOA[QCFlag_Test1C_ppl_BNI],QC_df.BNI[QCFlag_Test1C_ppl_BNI],'r.',markersize=1,label='PPL flag')
plt.legend(loc='upper left')
plt.xlabel('TOA [W/m2]')
plt.ylabel('BNI [W/m2]')
#%% BSRN two component test
import matplotlib.pyplot as plt

KT=QC_df.GHI/QC_df.TOA
KT[QC_df.TOA<1]=0
SZA=QC_df.THETA_Z*180/np.pi

QCFlag_Test2C_bsrn_kd=((QC_df.GHI>50) & (SZA<75) & (KT>1.05)) | ((QC_df.GHI>50) & (SZA>=75) & (KT>1.1))


plt.figure(figsize=(15,5))

ax11 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
plt.plot(day[QCFlag_Test2C_bsrn_kd],TOD[QCFlag_Test2C_bsrn_kd],'r.',markersize=0.8,label='2-C BSRN flag')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the BSRN two-component test")

ax12 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

plt.plot(SZA,KT,'.',markersize=0.5,label='all data',color=[0.75,0.75,0.75])
plt.plot(SZA[QC_df.GHI>50],KT[QC_df.GHI>50],'.',markersize=0.5,label='GHI>50 W/m2',color=[0,0,0.75])
plt.plot(SZA[QCFlag_Test2C_bsrn_kd],KT[QCFlag_Test2C_bsrn_kd],'r.',markersize=1,label="Flagged data")
plt.plot([0,75,75,100],[1.05,1.05,1.1,1.1],color='k',label='upper limit')
    
plt.legend(loc='lower left')
plt.xlabel('SZA [°]')
plt.ylabel('KT [-]')

plt.xlim((10,95))
plt.ylim((0,1.25))

#%% SERI-QC two-component test

KT=QC_df.GHI/QC_df.TOA
KT[QC_df.TOA<1]=0

Kn=QC_df.BNI/QC_df.TOANI
Kn[QC_df.TOANI<1]=0

Kd=QC_df.DHI/QC_df.GHI
Kd[QC_df.GHI<1]=0

QCFlag_Test2C_seri_knkt=(Kn>KT)|(Kn>0.8)|(KT>1)
QCFlag_Test2C_seri_kdkt=((KT<0.6) & (Kd>1.1)) | ((KT>=0.6) & (Kd>0.95)) 


plt.figure(figsize=(15,9))

ax11 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
plt.plot(day[QCFlag_Test2C_seri_knkt],TOD[QCFlag_Test2C_seri_knkt],'r.',markersize=0.8,label='SERI-QC kn-kt test')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the SERI-QC kn-kt test")

ax12 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
plt.plot(KT,Kn,'b.',markersize=0.5,label='all data',alpha=0.1)
plt.plot([0,0.8,1,1],[0,0.8,0.8,0],'k--',label='limit kt-kn test')
plt.plot(KT[QCFlag_Test2C_seri_knkt],Kn[QCFlag_Test2C_seri_knkt],'r.',markersize=0.9,label='flagged data')
plt.legend(loc='upper left')
plt.xlabel('KT [-]')
plt.ylabel('kn [-]')
plt.xlim((0, 1.25))
plt.ylim((0, 1.))     
       
ax21 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
plt.plot(day[QCFlag_Test2C_seri_kdkt],TOD[QCFlag_Test2C_seri_kdkt],'r.',markersize=0.8,label='SERI-QC kd-kt test')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the SERI-QC kd-kt test")

ax22 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
plt.plot(KT,Kd,'b.',markersize=0.5,label='all data',alpha=0.1)
plt.plot([0,0.6,0.6,1,1],[1.1,1.1,0.95,0.95,0],'k--',label='limit kd-kt test')
plt.plot(KT[QCFlag_Test2C_seri_kdkt],Kd[QCFlag_Test2C_seri_kdkt],'r.',markersize=0.9,label='flagged data')
plt.legend(loc='lower left')
plt.xlabel('KT [-]')
plt.ylabel('Kd [-]')

plt.xlim((0, 1.2))
plt.ylim((0, 1.2)) 

#%% BSRN three component tests
GHI=QC_df.GHI
GHI_est=QC_df.DHI+QC_df.BNI*np.cos(QC_df.THETA_Z)
SZA=QC_df.THETA_Z*180/np.pi

QCFlag_Test3C_bsrn_3cmp=((SZA<=75) & (GHI>50) & (np.abs(GHI/GHI_est-1)>0.08)) | ((SZA>75) & (GHI>50) & (np.abs(GHI/GHI_est-1)>0.15))

plt.figure(figsize=(15,9))

ax11 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
plt.plot(day[QCFlag_Test3C_bsrn_3cmp],TOD[QCFlag_Test3C_bsrn_3cmp],'r.',markersize=0.8,label='BSRN 3CMP test')
plt.ylim((0,24))
plt.legend(loc='upper left')
plt.ylabel('Time of the day [h]')
plt.title("Results of the BSRN three-component test")

ax22 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
plt.plot(SZA[GHI>50],GHI[GHI>50]/GHI_est[GHI>50],'b.',markersize=0.5,label='all data with GHI>50 W/m2',alpha=0.1)
plt.plot(SZA[QCFlag_Test3C_bsrn_3cmp],GHI[QCFlag_Test3C_bsrn_3cmp]/GHI_est[QCFlag_Test3C_bsrn_3cmp],'r.',markersize=1,label='Flagged data',alpha=0.1)
plt.plot([10,75,75,90,90, 75,75,10],[1.08,1.08,1.15,1.15,0.85,0.85,0.92,0.92],'k--',label='limit of the BSRN 3C test')
plt.xlabel('SZA [°]')
plt.ylabel('GHI/(DHI+BNI*cos(SZA)) [-]')

plt.legend(loc='lower left')
plt.xlim((10, 90))
plt.ylim((0.5, 1.5)) 

ax22 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
plt.plot(GHI[GHI>50],GHI_est[GHI>50],'b.',markersize=0.5,label='all data with GHI>50 W/m2',alpha=0.1)
plt.plot(GHI[QCFlag_Test3C_bsrn_3cmp],GHI_est[QCFlag_Test3C_bsrn_3cmp],'r.',markersize=1,label='Flagged data',alpha=0.1)
plt.xlabel('GHI [W/m2]')
plt.ylabel('DHI+BNI*cos(SZA) [W/m2]')
plt.legend(loc='lower right')
plt.xlim((0, 1400))
plt.ylim((0, 1400)) 

ax21 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
plt.plot(QC_df.index[QC_df.GHI>50],GHI[QC_df.GHI>50]/GHI_est[QC_df.GHI>50],'b.',markersize=0.5,label='all data with GHI>50 W/m2',alpha=0.1)
plt.plot(QC_df.index[QCFlag_Test3C_bsrn_3cmp],GHI[QCFlag_Test3C_bsrn_3cmp]/GHI_est[QCFlag_Test3C_bsrn_3cmp],'r.',markersize=1,label='flagged data',alpha=0.1)
plt.ylabel('GHI/(DHI+BNI*cos(SZA)) [-]')
plt.legend(loc='lower left')
plt.ylim((0.5, 1.5)) 
