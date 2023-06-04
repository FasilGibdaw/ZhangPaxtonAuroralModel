import matplotlib.path as mpath
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")
# This is a script for the kp-based auroral model provided in the paper
# https://doi.org/10.1016/j.jastp.2008.03.008
# -------- last updated on Januray 2023 by Fasil Tesema (fasil.kebede@helsinki.fi)
#
MLT = np.arange(0, 24, 0.5)  # magnetic local time
Mlat = np.arange(40, 90.5, 0.5)  # magnetic latitude
ang = MLT*2*np.pi/24
chi = 90-abs(Mlat)  # co-latitude
kp_m = np.array([0.75, 2.25, 3.75, 5.25, 7, 9])  # kp_model refer the paper

file_dir = './eflux_coeff/'
file_dir2 = './emean_coeff/'


def main():
    plot_kp(3, savefig=True)


def read_coeff(file):
    coeff = np.loadtxt(file, usecols=(1, 2, 3, 4))
    const = coeff[0]
    ind_cos = coeff[1:7]
    ind_sin = coeff[7:]
    return const, ind_cos, ind_sin


def compute_coeff(file):
    const, ind_cos, ind_sin = read_coeff(file)
    co = np.zeros((4, len(MLT)))
    for i in range(4):
        CONST = const[i]
        for ij in range(6):
            k = ij +1
            CONST = CONST+ind_cos[ij, i]*np.cos(k*ang)+ind_sin[ij, i]*np.sin(k*ang)
        co[i, :] = CONST
    return co


# define the file paths in a dictionary
file_paths = {
    1.5: ('K0.txt', 'K1.txt'),
    3: ('K0.txt', 'K1.txt'),
    4.5: ('K1.txt', 'K2.txt'),
    6: ('K2.txt', 'K3.txt'),
    8: ('K3.txt', 'K4.txt'),
    10: ('K4.txt', 'K5.txt')
}


def flux_coeff(kp):
    constL, constU = None, None

    # retrieve the file paths based on the value of kp
    for k, v in file_paths.items():
        if kp <= k:
            constL = compute_coeff(file_dir + v[0])
            constU = compute_coeff(file_dir + v[1])
            break

    return constL, constU


def mean_coeff(kp):
    constL, constU = None, None

    # retrieve the file paths based on the value of kp
    for k, v in file_paths.items():
        if kp <= k:
            constL = compute_coeff(file_dir2 + v[0])
            constU = compute_coeff(file_dir2 + v[1])
            break

    return constL, constU


def kpm(kp):
    if kp < 0.75:
        kpm1 = 0.75
        kpm2 = 2.25
    else:
        im1 = np.where(kp_m <= kp)
        im2 = np.where(kp_m > kp)
        kpm1 = kp_m[im1[0][-1]]
        kpm2 = kp_m[im2[0][0]]
    return kpm1, kpm2


def hemispheric_power(kp):
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
    flux = np.full((len(MLT), len(Mlat)), np.nan)
    # flux = np.nan*np.zeros((len(MLT), len(Mlat)))
    flux = []
    L, U = flux_coeff(kp)
    for a, b, c, d, a2, b2, c2, d2 in zip(L[0], L[1], L[2], L[3], U[0], U[1], U[2], U[3]):
        Eom1 = (a*np.exp((chi-b)/c))/((1+np.exp((chi-b)/d))**2)
        Eom2 = (a2*np.exp((chi-b2)/c2))/((1+np.exp((chi-b2)/d2))**2)
        flux.append(f1*Eom1+f2*Eom2)
    # Eo = F1*Eom1+F2*Eom2
    return np.array(flux)


def Emean(kp):
    emean = np.nan*np.zeros((len(MLT), len(Mlat)))
    emean = []
    L, U = mean_coeff(kp)
    F1, F2 = hemispheric_power(kp)
    for a, b, c, d, a2, b2, c2, d2 in zip(L[0], L[1], L[2], L[3], U[0], U[1], U[2], U[3]):
        Eom1 = (a*np.exp((chi-b)/c))/((1+np.exp((chi-b)/d))**2)
        Eom2 = (a2*np.exp((chi-b2)/c2))/((1+np.exp((chi-b2)/d2))**2)
        emean.append(F1*Eom1+F2*Eom2)
    # Eo = F1*Eom1+F2*Eom2
    return np.array(emean)


def find_boundary_indices(array, value):
    top_indices = []
    bottom_indices = []
    _, c = array.shape
    for i in range(c):
        cont = np.where(array[:, i] >= value)[0]
        if cont.size > 0:
            top_ind = cont[0]
            bottom_ind = cont[-1]
        else:
            top_ind = np.nan
            bottom_ind = np.nan
        top_indices.append(top_ind)
        bottom_indices.append(bottom_ind)
    return top_indices, bottom_indices


def plot_kp(kp, savefig=False):
    emean = Emean(kp)
    eflux = Eflux(kp)
    cmap_upper=4
    top_indices, bottom_indices = find_boundary_indices(eflux.T, 0.25)
    Lat = np.arange(40, 90.5, 0.5)  # for Southern hemisphere -90:0.5:-30
    Lon = np.arange(0, 360, 7.5)
    xlat, ylon = np.meshgrid(Lat, Lon)
    ###
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
    fig.subplots_adjust(bottom=0.05, top=0.95,
                        left=0.04, right=0.95, wspace=0.02)
    # Limit the map to -60 degrees latitude and below.
    # ax1.set_extent([-180, 180, 90, 40], ccrs.PlateCarree())
    # ax1.gridlines()

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)

    cs1 = ax1.pcolormesh(ylon, xlat, emean,
                     transform=ccrs.PlateCarree(), cmap='jet',vmin=0,vmax=cmap_upper)

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=1, color='black', alpha=0.3, linestyle='--')
    ax1.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    yticks = list(np.arange(40, 90, 15))
    xx = np.arange(-180, 180, 45)
    gl.xlocator = mticker.FixedLocator(xx)
    loc_x_mlt = [0.485, 0.86, 1.01, 0.86, 0.485, 0.1, -0.05, 0.1]
    loc_y_mlt = [-0.04, 0.11, 0.485, 0.86, 1.02, 0.86, 0.485, 0.1]
    loc_x_lat = [0.5]*6
    loc_y_lat = [0.47, 0.4, 0.3, 0.2, 0.1, 0.]
    mlt_label = [str(elem) for elem in np.arange(0, 24, 3)]
    lat_label = [str(elem) for elem in np.arange(90, 30, -10)]
    for xmlt, ymlt, label_mlt in zip(loc_x_mlt, loc_y_mlt, mlt_label):
        ax1.text(xmlt, ymlt, label_mlt, transform=ax1.transAxes)
    for x_lat, ylat, label_lat in zip(loc_x_lat, loc_y_lat, lat_label):
        ax1.text(x_lat, ylat, label_lat, transform=ax1.transAxes)
    # gl.xlocator = mticker.FixedLocator(xx)
    # ax1.set_xlabel(['0','3','6','9','12','15','18','21'])
    # gl.right_labels = gl.left_labels = gl.top_labels = True

    # ax1.axis('off')
    fig.colorbar(cs1, label=r'Mean energy ($KeV$)')
    ax1.text(0.7, 1, 'Mean energy, '+'Kp='+str(kp), transform=ax1.transAxes)
    ax1.plot(Lon, Lat[bottom_indices], '.k', transform=ccrs.PlateCarree())
    ax1.plot(Lon, Lat[top_indices], '.g', transform=ccrs.PlateCarree())
    ax2 = fig.add_subplot(122, projection=ccrs.NorthPolarStereo())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    cs2 = ax2.pcolormesh(ylon, xlat, eflux,
                     transform=ccrs.PlateCarree(), cmap='jet',vmin=0,vmax=cmap_upper)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=1, color='black', alpha=0.3, linestyle='--')
    ax2.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    # gl.right_labels = gl.left_labels = gl.top_labels = False
    # ax2.axis('off')
    # plt.tight_layout()
    xx = np.arange(-180, 180, 45)
    gl.xlocator = mticker.FixedLocator(xx)
    for xmlt, ymlt, label_mlt in zip(loc_x_mlt, loc_y_mlt, mlt_label):
        ax2.text(xmlt, ymlt, label_mlt, transform=ax2.transAxes)
    for x_lat, ylat, label_lat in zip(loc_x_lat, loc_y_lat, lat_label):
        ax2.text(x_lat, ylat, label_lat, transform=ax2.transAxes)
    fig.colorbar(cs2, label=r'Flux ($erg/s/cm^{2}$)')
    ax2.text(0.7, 1, 'Energy flux, '+'Kp='+str(kp), transform=ax2.transAxes)
    ax2.plot(Lon, Lat[bottom_indices], '.k', transform=ccrs.PlateCarree())
    ax2.plot(Lon, Lat[top_indices], '.g', transform=ccrs.PlateCarree())
    if savefig == True:
        plt.savefig('ZhangPaxtonModel_KP'+str(kp)+'.png', dpi=800)
    plt.show()


if __name__ == '__main__':
    main()
