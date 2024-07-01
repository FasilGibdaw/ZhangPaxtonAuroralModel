
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings


import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Constants and directories
kp_m = np.array([0.75, 2.25, 3.75, 5.25, 7, 9])
file_dir = "./eflux_coeff/"
file_dir2 = "./emean_coeff/"
file_paths = {
    1.5: ("K0.txt", "K1.txt"),
    3: ("K0.txt", "K1.txt"),
    4.5: ("K1.txt", "K2.txt"),
    6: ("K2.txt", "K3.txt"),
    8: ("K3.txt", "K4.txt"),
    10: ("K4.txt", "K5.txt")
}


def prepare_data(mlat, mlt):
    """Prepare common data used in various calculations."""
    ang = mlt * 2 * np.pi / 24
    chi = 90 - np.abs(mlat)
    return ang, chi


def read_coeff(file):
    """Read coefficients from a given file."""
    coeff = np.loadtxt(file, usecols=(1, 2, 3, 4))
    return coeff[0], coeff[1:7], coeff[7:]


def compute_coeff(file, ang):
    """Compute coefficients from a file for different MLTs."""
    const, ind_cos, ind_sin = read_coeff(file)
    return np.array([
        const[i] + sum(
            ind_cos[ij, i] * np.cos((ij + 1) * ang) +
            ind_sin[ij, i] * np.sin((ij + 1) * ang)
            for ij in range(6)
        ) for i in range(4)
    ])


def flux_coeff(kp, ang):
    """Calculate flux coefficients for a given Kp index."""
    for k, v in file_paths.items():
        if kp <= k:
            return (
                compute_coeff(file_dir + v[0], ang),
                compute_coeff(file_dir + v[1], ang)
            )
    return None, None


def mean_coeff(kp, ang):
    """Calculate mean coefficients for a given Kp index."""
    for k, v in file_paths.items():
        if kp <= k:
            return (
                compute_coeff(file_dir2 + v[0], ang),
                compute_coeff(file_dir2 + v[1], ang)
            )
    return None, None


def kpm(kp):
    """Find adjacent Kp_model values."""
    if kp < 0.75:
        return 0.75, 2.25
    im1 = kp_m[kp_m <= kp][-1]
    im2 = kp_m[kp_m > kp][0]
    return im1, im2


def hemispheric_power(kp):
    """Calculate hemispheric power factors."""
    kpm1, kpm2 = kpm(kp)
    if kp <= 5:
        HP = 38.66 * np.exp(0.1967 * kp) - 33.99
        HPm1 = 38.66 * np.exp(0.1967 * kpm1) - 33.99
        HPm2 = 38.66 * np.exp(0.1967 * kpm2) - 33.99
    else:
        HP = 4.592 * np.exp(0.4731 * kp) + 20.47
        HPm1 = 4.592 * np.exp(0.4731 * kpm1) + 20.47
        HPm2 = 4.592 * np.exp(0.4731 * kpm2) + 20.47
    F1 = (HPm2 - HP) / (HPm2 - HPm1)
    F2 = (HP - HPm1) / (HPm2 - HPm1)
    return F1, F2


def Eflux(mlat, mlt, kp):
    """Calculate nonlinear interpolation for energy flux for a given Kp index."""
    ang, chi = prepare_data(mlat, mlt)
    kpm1, kpm2 = kpm(kp)
    f1, f2 = (kpm2 - kp) / (kpm2 - kpm1), (kp - kpm1) / (kpm2 - kpm1)
    L, U = flux_coeff(kp, ang)
    Eom1_L = (L[0] * np.exp((chi - L[1]) / L[2])) / \
        ((1 + np.exp((chi - L[1]) / L[3]))**2)
    Eom1_U = (U[0] * np.exp((chi - U[1]) / U[2])) / \
        ((1 + np.exp((chi - U[1]) / U[3]))**2)
    return f1 * Eom1_L + f2 * Eom1_U


def Emean(mlat, mlt, kp):
    """Calculate nonlinear interpolation for energy mean for a given Kp index."""
    ang, chi = prepare_data(mlat, mlt)
    L, U = mean_coeff(kp, ang)
    F1, F2 = hemispheric_power(kp)
    Eom1_L = (L[0] * np.exp((chi - L[1]) / L[2])) / \
        ((1 + np.exp((chi - L[1]) / L[3]))**2)
    Eom1_U = (U[0] * np.exp((chi - U[1]) / U[2])) / \
        ((1 + np.exp((chi - U[1]) / U[3]))**2)

    return F1 * Eom1_L + F2 * Eom1_U


def find_boundary_indices(array, value):
    """
    Find the indices of the top and bottom boundaries in a 2D array based on a threshold value.
    It scans each column of the 2D array to find where the values first exceed
    or equal the specified threshold (top boundary) and where they last exceed or equal
    the threshold (bottom boundary).

    Args:
        array (numpy.ndarray): The 2D array for which to find boundary indices.
        value (float): The threshold value for determining the boundaries.

    Returns:
        tuple: A tuple containing lists of top and bottom indices for each column.
    """
    top_indices = []
    bottom_indices = []
    _, c = array.shape
    for i in range(c):
        cont = np.where(array[:, i] >= value)[0]
        if cont.size > 0:
            top_ind = cont[0]
            bottom_ind = cont[-1]
        else:
            top_ind = bottom_ind
        top_indices.append(top_ind)
        bottom_indices.append(bottom_ind)
    return top_indices, bottom_indices


def calculate_conductance(mlat, mlt, kp):
    # calculates the conductance (pedersen and hall) for a given Kp index using Robinson 1987 paper
    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA092iA03p02565
    pedersen_conductance = (
        (40 * Emean(mlat, mlt, kp))/(16 + Emean(mlat, mlt, kp)**2))*(Eflux(mlat, mlt, kp)**(1/2))
    hall_conductance = (0.45 * (Emean(mlat, mlt, kp)**0.85)
                        )*pedersen_conductance
    return pedersen_conductance, hall_conductance


def plot_kp(mlat, mlt, kp, savefig=False, cmap_upper=6):
    """
    Plot the energy mean and energy flux maps for a given Kp index. This method generates 
    and displays two subplots: one for the energy mean map and
    one for the energy flux map. The maps are plotted using a North Polar Stereographic
    projection. The auroral boundaries on the maps are indicated using lines based on top and
    bottom boundary indices.

    Args:
        kp (float): The Kp index for which to generate the plots.
        savefig (bool, optional): Whether to save the generated plots as an image. Default is False.
        cmap_upper (int, optional): Upper limit for colormap scaling. Default is 6.

    Returns:
        display and or save figure
        If 'savefig' is True, the plots are saved as image files.
        Else displays figure 
    """
    Lat = mlat  # for Southern hemisphere -90:0.5:-30
    Lon = mlt*15
    mlat, mlt = np.meshgrid(mlat, mlt)
    MLT = mlt.flatten()
    MLAT = mlat.flatten()
    emean = Emean(MLAT, MLT, kp)
    eflux = Eflux(MLAT, MLT, kp)
    # emean, eflux = calculate_conductance(MLAT, MLT, kp)
    emean = emean.reshape(mlat.shape)
    eflux = eflux.reshape(mlat.shape)
    xlat, ylon = mlat, mlt*15

    top_indices, bottom_indices = find_boundary_indices(eflux.T, 0.25)

    ###
    colors = [
        "#000000",
        "#031b03",
        "#08420b",
        "#1a5419",
        "#377f33",
        "#6bb25a",
        "#a3d683",
        "#d4f1a5",
        "#f5ffd5",
    ]  # Example colors
    colors.reverse()
    green_aurora_cmap = mcolors.LinearSegmentedColormap.from_list(
        "GreenAurora", colors
    )
    ###
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
    fig.subplots_adjust(bottom=0.05, top=0.95,
                        left=0.04, right=0.95, wspace=0.02)

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)

    cs1 = ax1.pcolormesh(
        ylon,
        xlat,
        emean,
        transform=ccrs.PlateCarree(),
        cmap=green_aurora_cmap,
        vmin=0,
        vmax=cmap_upper,
    )

    gl = ax1.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color="black",
        alpha=0.3,
        linestyle="--",
    )
    ax1.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    yticks = list(np.arange(40, 90, 15))
    xx = np.arange(-180, 180, 45)
    gl.xlocator = mticker.FixedLocator(xx)
    loc_x_mlt = [0.485, 0.86, 1.01, 0.86, 0.485, 0.1, -0.05, 0.1]
    loc_y_mlt = [-0.04, 0.11, 0.485, 0.86, 1.02, 0.86, 0.485, 0.1]
    loc_x_lat = [0.5] * 6
    loc_y_lat = [0.47, 0.4, 0.3, 0.2, 0.1, 0.0]
    mlt_label = [str(elem) for elem in np.arange(0, 24, 3)]
    lat_label = [str(elem) for elem in np.arange(90, 30, -10)]
    for xmlt, ymlt, label_mlt in zip(loc_x_mlt, loc_y_mlt, mlt_label):
        ax1.text(xmlt, ymlt, label_mlt, transform=ax1.transAxes)
    for x_lat, ylat, label_lat in zip(loc_x_lat, loc_y_lat, lat_label):
        ax1.text(x_lat, ylat, label_lat, transform=ax1.transAxes)
    fig.colorbar(cs1, label=r"Mean energy ($KeV$)")
    ax1.text(0.7, 1, "Mean energy, " + "Kp=" +
             str(kp), transform=ax1.transAxes)
    ax1.plot(Lon, Lat[bottom_indices], "k", transform=ccrs.PlateCarree())
    ax1.plot(Lon, Lat[top_indices], "--r", transform=ccrs.PlateCarree())
    ax2 = fig.add_subplot(122, projection=ccrs.NorthPolarStereo())
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    cs2 = ax2.pcolormesh(
        ylon,
        xlat,
        eflux,
        transform=ccrs.PlateCarree(),
        cmap=green_aurora_cmap,
        vmin=0,
        vmax=cmap_upper-2,
    )
    gl = ax2.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color="black",
        alpha=0.3,
        linestyle="--",
    )
    ax2.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    xx = np.arange(-180, 180, 45)
    gl.xlocator = mticker.FixedLocator(xx)
    for xmlt, ymlt, label_mlt in zip(loc_x_mlt, loc_y_mlt, mlt_label):
        ax2.text(xmlt, ymlt, label_mlt, transform=ax2.transAxes)
    for x_lat, ylat, label_lat in zip(loc_x_lat, loc_y_lat, lat_label):
        ax2.text(x_lat, ylat, label_lat, transform=ax2.transAxes)
    fig.colorbar(cs2, label=r"Flux ($erg/s/cm^{2}$)")
    ax2.text(0.7, 1, "Energy flux, " + "Kp=" +
             str(kp), transform=ax2.transAxes)
    ax2.plot(Lon, Lat[bottom_indices], "k", transform=ccrs.PlateCarree())
    ax2.plot(Lon, Lat[top_indices], "--r", transform=ccrs.PlateCarree())
    if savefig == True:
        plt.savefig("ZhangPaxtonModel_KP" + str(kp) + ".png", dpi=800)
    plt.show()


if __name__ == "__main__":
    # mlt = np.arange(0, 24, 0.15)  # magnetic local time
    # magnetic latitude: for Southern hemisphere -90:0.5:-30
    mlat = np.arange(40, 90.5, 0.15)
    mlt = np.linspace(0, 24, len(mlat))  # magnetic local time
    plot_kp(mlat, mlt, 3, savefig=False, cmap_upper=6)
