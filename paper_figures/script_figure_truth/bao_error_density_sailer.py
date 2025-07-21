
import numpy as np
import matplotlib.pyplot as plt
nratio = np.linspace(1,15,100)
errors = np.loadtxt('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/fisher_forecast/errors.txt').reshape((100,2,4))

labels = [r'$0.7 < z < 0.9$',r'$0.9 < z < 1.1$',r'$1.1 < z < 1.3$',r'$1.3 < z < 1.5$']
colors = ['red','orange','green','blue']

fig, ax = plt.subplots(2,1,figsize=(11,10),sharex=True)

ax[0].axvline(x=1,c='k',lw=2,ls='-', label='DESI')
ax[0].axvline(x=4,c='k',lw=2,ls='--', label='target')

for i in range(4):
    ax[0].plot(nratio,100*errors[:,0,i],c=colors[i],lw=4,label=labels[i])
    ax[1].plot(nratio,100*errors[:,1,i],c=colors[i],lw=4)

ax[0].yaxis.set_tick_params(labelsize=20)
ax[0].set_xlim(0.5,10)
ax[0].legend(loc='upper right',frameon=False,framealpha=1, handlelength=0.8, ncol=2, fontsize=22)
ax[0].set_ylabel(r'$\sigma_{\alpha_{\perp}}\, \times$ 100', fontsize=27)
ax[0].set_ylim(0.3,1.4)
ax[0].set_yticks(np.linspace(0.3,1.4,int(12)),minor=True)
ax[0].grid(alpha=0.3,which='both') ; ax[1].grid(alpha=0.3, which='both')

ax[1].set_yticks(np.linspace(0.5,1.75,int(11)),minor=True)
ax[1].set_ylabel(r'$\sigma_{\alpha_{\parallel}}\, \times$ 100', fontsize=27)
ax[1].set_xlabel(r'ELG $\bar{n}(z)$ (normalized to DESI)', fontsize=27)
ax[1].set_ylim(0.50,1.75)
ax[1].set_xticks([2,4,6,8,10,12,14])
ax[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14],minor=True)
ax[1].xaxis.set_tick_params(labelsize=20)
ax[1].yaxis.set_tick_params(labelsize=20)
ax[1].axvline(x=4,c='k',lw=2,ls='--')
ax[1].axvline(x=1,c='k',lw=2,ls='-')
fig.subplots_adjust(hspace=0.15)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/bao_error_density_sailer.png', dpi=300, bbox_inches='tight')