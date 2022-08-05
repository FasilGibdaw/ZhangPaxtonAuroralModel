import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# This is a script for the kp-based auroral model provided in the paper
# https://doi.org/10.1016/j.jastp.2008.03.008
# This is not complete and not in standard, feel free to modify and use it
# -------- last updated on Januray 2022 by Fasil Tesema (fasil.kebede@helsinki.fi)
# --------
# --------
#
MLT = np.arange(0, 24, 0.5)  # magnetic local time
# magnetic latitude: for Southern hemisphere -90:0.5:-30
Mlat = np.arange(40, 90.5, 0.5)
ang = MLT*2*np.pi/24
chi = 90-abs(Mlat)  # co-latitude
kp_m = np.array([0.75, 2.25, 3.75, 5.25, 7, 9])  # kp_model refer the paper
# avoid kp=7 and kp=9 to avoid NaNs in the model calculations

file_dir = os.getcwd()+'/eflux_coeff/'
file_dir2 = os.getcwd()+'/emean_coeff/'


def main():
    plot_kp(5, savefig=True)


def read_coeff(file):
    coeff = np.loadtxt(file, usecols=(1, 2, 3, 4))
    const = coeff[0]
    ind_cos = coeff[1:7]
    ind_sin = coeff[7:]
    return const, ind_cos, ind_sin


def compute_coeff(file):
    const, ind_cos, ind_sin = read_coeff(file)
    co = np.zeros((4, 48))
    for i in range(4):
        CONST = const[i]
        for ij in range(6):
            CONST = CONST+ind_cos[ij, i]*np.cos(ang)+ind_sin[ij, i]*np.sin(ang)
        co[i, :] = CONST
    return co


def flux_coeff(kp):
    if kp <= 1.5 or np.logical_and(kp > 1.5, kp <= 3):
        constL = compute_coeff(file_dir+'K0.txt')
        constU = compute_coeff(file_dir+'K1.txt')
    elif np.logical_and(kp > 3, kp <= 4.5):
        constL = compute_coeff(file_dir+'K1.txt')
        constU = compute_coeff(file_dir+'K2.txt')
    elif np.logical_and(kp > 4.5, kp <= 6):
        constL = compute_coeff(file_dir+'K2.txt')
        constU = compute_coeff(file_dir+'K3.txt')
    elif np.logical_and(kp > 6, kp <= 8):
        constL = compute_coeff(file_dir+'K3.txt')
        constU = compute_coeff(file_dir+'K4.txt')
    elif np.logical_and(kp > 8, kp <= 10):
        constL = compute_coeff(file_dir+'K4.txt')
        constU = compute_coeff(file_dir+'K5.txt')
    return constL, constU


def mean_coeff(kp):
    if kp <= 1.5 or np.logical_and(kp > 1.5, kp <= 3):
        constL = compute_coeff(file_dir2+'K0.txt')
        constU = compute_coeff(file_dir2+'K1.txt')
    elif np.logical_and(kp > 3, kp <= 4.5):
        constL = compute_coeff(file_dir2+'K1.txt')
        constU = compute_coeff(file_dir2+'K2.txt')
    elif np.logical_and(kp > 4.5, kp <= 6):
        constL = compute_coeff(file_dir2+'K2.txt')
        constU = compute_coeff(file_dir2+'K3.txt')
    elif np.logical_and(kp > 6, kp <= 8):
        constL = compute_coeff(file_dir2+'K3.txt')
        constU = compute_coeff(file_dir2+'K4.txt')
    elif np.logical_and(kp > 8, kp <= 10):
        constL = compute_coeff(file_dir2+'K4.txt')
        constU = compute_coeff(file_dir2+'K5.txt')
    return constL, constU


def kpm(kp):
    if kp < 0.75:
        kpm1 = 0.75
        kpm2 = 2.25
    else:
        im1 = np.where(kp_m <= kp)
        im2 = np.where(kp_m >= kp)
        kpm1 = kp_m[im1[0][-1]]
        kpm2 = kp_m[im2[0][0]]
    return kpm1, kpm2


def HemisphericPower(kp):
    kpm1, kpm2 = kpm(kp)
    if kp <= 5:
        HP = 38.66*np.exp(0.1967*kp)-33.99  # -33.99
        HPm1 = 38.66*np.exp(0.1967*kpm1)-33.99
        HPm2 = 38.66*np.exp(0.1967*kpm2)-33.99
    else:
        HP = 4.592*np.exp(0.4731*kp)+20.47  # +20.47
        HPm1 = 4.592*np.exp(0.4731*kpm1)+20.47
        HPm2 = 4.592*np.exp(0.4731*kpm2)+20.47
    F1 = (HPm2-HP)/(HPm2-HPm1)
    F2 = (HP-HPm1)/(HPm2-HPm1)
    return F1, F2


def Eflux(kp):
    kpm1, kpm2 = kpm(kp)
    f1 = (kpm2-kp)/(kpm2-kpm1)
    f2 = (kp-kpm1)/(kpm2-kpm1)
    flux = np.nan*np.zeros((len(MLT), len(Mlat)))
    #Emean = np.nan*np.zeros((len(MLT), len(Mlat)))
    L, U = flux_coeff(kp)
    for i in range(len(MLT)):
        a = L[0][i]
        b = L[1][i]
        c = L[2][i]
        d = L[3][i]
        a2 = U[0][i]
        b2 = U[1][i]
        c2 = U[2][i]
        d2 = L[3][i]
        Eom1 = (a*np.exp((chi-b)/c))/((1+np.exp((chi-b)/d))**2)
        Eom2 = (a2*np.exp((chi-b2)/c2))/((1+np.exp((chi-b2)/d2))**2)
        flux[i:] = f1*Eom1+f2*Eom2
    #Eo = F1*Eom1+F2*Eom2
    return flux


def Emean(kp):
    emean = np.nan*np.zeros((len(MLT), len(Mlat)))
    L, U = mean_coeff(kp)
    F1, F2 = HemisphericPower(kp)
    for i in range(len(MLT)):
        a = L[0][i]
        b = L[1][i]
        c = L[2][i]
        d = L[3][i]
        a2 = U[0][i]
        b2 = U[1][i]
        c2 = U[2][i]
        d2 = L[3][i]
        Eom1 = (a*np.exp((chi-b)/c))/((1+np.exp((chi-b)/d))**2)
        Eom2 = (a2*np.exp((chi-b2)/c2))/((1+np.exp((chi-b2)/d2))**2)
        emean[i:] = F1*Eom1+F2*Eom2
    #Eo = F1*Eom1+F2*Eom2
    return emean


def plot_kp(kp, savefig=False):
    emean = Emean(kp)
    eflux = Eflux(kp)
    Lat = np.arange(40, 90.5, 0.5)  # for Southern hemisphere -90:0.5:-30
    Lon = np.arange(0, 360, 7.5)
    xlat, ylon = np.meshgrid(Lat, Lon)
    ###
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection=ccrs.NorthPolarStereo())
    cs1 = ax1.pcolor(ylon, xlat, emean,
                     transform=ccrs.PlateCarree(), cmap='jet')
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=1, color='white', alpha=0.3, linestyle='--')
    ax1.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    gl.right_labels = gl.left_labels = gl.top_labels = False
    ax1.axis('off')
    plt.tight_layout()
    fig.colorbar(cs1, shrink=0.65, label=r'Mean energy ($KeV$)')
    ax1.set_title('Mean energy, '+'Kp='+str(kp))

    ax2 = fig.add_subplot(122, projection=ccrs.NorthPolarStereo())
    cs2 = ax2.pcolor(ylon, xlat, eflux,
                     transform=ccrs.PlateCarree(), cmap='jet')
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=1, color='white', alpha=0.3, linestyle='--')
    ax2.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    gl.right_labels = gl.left_labels = gl.top_labels = False
    ax2.axis('off')
    plt.tight_layout()
    fig.colorbar(cs2, shrink=0.65, label=r'Flux ($erg/s/cm^{2}$)')
    ax2.set_title('Energy flux, '+'Kp='+str(kp))
    if savefig == True:
        plt.savefig('ZhangPaxtonModel_KP'+str(kp)+'.png')
    plt.show()


if __name__ == '__main__':
    main()
