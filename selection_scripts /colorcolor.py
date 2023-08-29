import scipy as sp
import time
import numpy as np 
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import table
from astropy.table import join
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
import pickle
#Function to create color color plots
def colorsthres(c1, c2, c3, prob, cmask, iymin, xlabel0= 'i-z', ylabel0= 'r-i', xlabel1 = 'i-y', 
                             ylabel1 = 'r-i', xlabel2 = 'i-y', ylabel2 = 'i-z',
                             title0 = '',
                             title1 = '',
                             title2 = '',
                             xlim0=(-2, 2), ylim0 = (0, 3), xlim1= (-2, 2), 
                             ylim1=(0, 3), xlim2 = (-2, 2), ylim2 = (0, 3),**c_kwargs):
    
    #pcmask = np.logical_and.reduce(prob,colormaskx)

    fig, (ax0, ax1, ax2) = plt.subplots(figsize=(12,4), ncols=3,  constrained_layout = True)
    
    sd0 = ax0.scatter(c1, c2, c=prob[cmask] , cmap = 'viridis', s = 5, alpha = .05 ,vmin=0, vmax=1, **c_kwargs)
    sd1 = ax1.scatter(c3, c2, c=prob[cmask] , cmap = 'viridis', s = 5, alpha = .05 ,vmin=0, vmax=1, **c_kwargs)
    sd2 = ax2.scatter(c3, c1, c=prob[cmask] , cmap = 'viridis', s = 5, alpha = .05, vmin=0, vmax=1, **c_kwargs)
    sdummy = ax0.scatter(x = 100, y = 100, c=prob[0], cmap = 'viridis', s = 5, alpha = 1 ,vmin=0, vmax=1, **c_kwargs)
    ax0.set_xlim(*xlim0)
    ax0.set_ylim(*ylim0)
    ax0.set_xlabel(xlabel0, fontsize = 15)
    ax0.set_ylabel(ylabel0, fontsize = 15)
    x = np.arange(0.28 , 2, .05)
    ax1.plot(x + 0.08 ,x, c = 'black')
    ax1.axvline(x = iymin , ymax = .35 , c = 'black') 
    ax1.set_xlim(*xlim1)
    ax1.set_ylim(*ylim1)
    ax1.set_xlabel(xlabel1, fontsize = 15)
    ax1.set_ylabel(ylabel1, fontsize = 15)
    ax2.set_xlim(*xlim2)
    ax2.set_ylim(*ylim2)
    ax2.set_xlabel(xlabel2, fontsize = 15)
    ax2.set_ylabel(ylabel2, fontsize = 15)
    ax2.axvline(x = 0.35 , c = 'black') 
    cbar = plt.colorbar(sdummy)
    cbar.set_label('predict_probability')
    
    ax0.set_title(title0)
    ax1.set_title(title1)
    ax2.set_title(title2)
    
    return fig

# Specify the version of the catalog and the folder with the input/output files

# Loading the Farmer version of the COSMOS2020 Cat
catversion1 = 'Farmer'  
dir_in1 ='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'  
dir_out1 = '/Users/yokisalcedo/Desktop/data/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
cat = table.Table.read(dir_in1+'COSMOS2020_{}_jan_processed.fits'.format(catversion1),format='fits',hdu=1).to_pandas()

#All possible non-redundant colors are listed below:
#u: ug ,ur ,ui ,uz , uy
#g: gr, gi, gz, gy
#r: ri, rz, ry
#i: iz, iy
#z: zy

ug = cat['CFHT_u_MAG'] - cat['HSC_g_MAG'] 
ur = cat['CFHT_u_MAG'] - cat['HSC_r_MAG']
ui = cat['CFHT_u_MAG'] - cat['HSC_i_MAG']
uz = cat['CFHT_u_MAG'] - cat['HSC_z_MAG'] 
uy = cat['CFHT_u_MAG'] - cat['HSC_y_MAG']
gr = cat['HSC_g_MAG'] - cat['HSC_r_MAG']
gi = cat['HSC_g_MAG'] - cat['HSC_i_MAG']
gz = cat['HSC_g_MAG'] - cat['HSC_z_MAG']
gy = cat['HSC_g_MAG'] - cat['HSC_y_MAG']
ri = cat['HSC_r_MAG'] - cat['HSC_i_MAG']
rz = cat['HSC_r_MAG'] - cat['HSC_z_MAG']
ry = cat['HSC_r_MAG'] - cat['HSC_y_MAG']
iz = cat['HSC_i_MAG'] - cat['HSC_z_MAG']
iy = cat['HSC_i_MAG'] - cat['HSC_y_MAG']
zy = cat['HSC_z_MAG'] - cat['HSC_y_MAG']
r = cat['HSC_r_MAG']
g = cat['HSC_g_MAG']
i = cat['HSC_i_MAG']
y = cat['HSC_y_MAG']
z = cat['HSC_z_MAG']

cat['ug']= ug
cat['ur']= ur
cat['ui']=ui
cat['uz']=uz
cat['uy']=uy
cat['gr']=gr
cat['gi']=gi
cat['gz']=gz
cat['gy']=gy
cat['ri']=ri
cat['rz']=rz
cat['ry']=ry
cat['iz']=iz
cat['iy']=iy
cat['zy']=zy
cat['HSC_r_MAG'] = r
cat['HSC_g_MAG'] = g
cat['HSC_i_MAG'] = i
cat['HSC_y_MAG'] = y
cat['HSC_z_MAG'] = z

colormaskx = np.logical_and.reduce((np.isfinite(cat['HSC_r_MAG']),
                                    np.isfinite(cat['photoz']),
                                    np.isfinite(cat['HSC_g_MAG']),np.isfinite(cat['HSC_i_MAG']),
                                    np.isfinite(cat['HSC_y_MAG']),np.isfinite(cat['CFHT_u_MAG']),
                                    np.isfinite(cat['HSC_z_MAG']),
                                    (np.logical_or(cat['HSC_g_MAG']< 24.5, cat['HSC_r_MAG']< 24.5))))

#Here we defind our narrow and broad zcut, all RFs and all colors
#(xalln, yn, zalln, galln), (xallb, yb, zallb , gallb )
zcutn = np.logical_and(cat['photoz'] > 1.05, cat['photoz'] < 1.45)
zcutb = np.logical_and(cat['photoz'] > 1.05, cat['photoz'] < 1.55)
xalln = cat.loc[colormaskx,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
xallb = cat.loc[colormaskx,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
zalln = cat['photoz'][colormaskx]
zallb = cat['photoz'][colormaskx]
galln = cat.loc[colormaskx,['HSC_g_MAG']]
gallb = cat.loc[colormaskx,['HSC_g_MAG']]
yn = zcutn[colormaskx]
yb = zcutb[colormaskx]
ralln = cat['HSC_r_MAG'][colormaskx]
rallb = cat['HSC_r_MAG'][colormaskx]
rialln = cat['HSC_r_MAG'][colormaskx] - cat['HSC_i_MAG'][colormaskx]

#Here, we split the data into training and testing sets
xallb_train, xallb_test, yb_train, yb_test, zallb_train, zallb_test, gallb_train, gallb_test, rallb_train, rallb_test = train_test_split(xallb, yb, zallb, gallb, rallb)
clfall_b = RandomForestClassifier()
# fitb = clfall_b.fit(xallb_train, yb_train)   
# pickle.dump(clfall_b, open('clfall_b.pkl', 'wb'))
pickled_clfall_b = pickle.load(open('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/notebooks/clfall_b.pkl', 'rb'))
pickled_clfall_b.predict(xallb_test) 
pickled_clfall_b.predict_proba(xallb_test) 
pickled_clfall_b.predict_proba(xallb)

predallb = pickled_clfall_b.predict(xallb_test) 
proballb = pickled_clfall_b.predict_proba(xallb_test) 
proballb_full = pickled_clfall_b.predict_proba(xallb)
probmaskallb_full = proballb_full[:, 1] >= 0.025

#color cuts for broad zcut
color_cuts = np.logical_and((xallb['iy'] > 0.35),(xallb['ri'] < xallb['iy'] - 0.1)) 

#Here we plot the color-color plot with the color cuts
colorsthres(xallb['iz'][color_cuts], 
xallb['ri'][color_cuts], 
xallb['iy'][color_cuts], 
proballb_full[:, 1], 
color_cuts, 
iymin = 0.35, 
xlim0 = (-0.1,1.0), 
ylim0 = (-0.15, 1.5), 
xlim1 = (-0.5,1.5), 
ylim1 = (-0.5, 1.7), 
xlim2 = (0,1.7), 
ylim2 = (-0.15, 1.0))

#save the figure as a png to the 'paperfigs' folder
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paperfigs/broad_colorcolor_iyri.png', dpi=300)
print('Figure saved')