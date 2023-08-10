import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings


class AuroraModel:
    """
    The AuroraModel class provides methods for visualizing aurora-related data based on the Zhang-Paxton auroral model.
    See the paper: https://doi.org/10.1016/j.jastp.2008.03.008.
    Class Attributes:
    -----------------
    MLT (numpy.ndarray): 
        An array representing Magnetic Local Time (MLT) values from 0 to 24 in steps of 0.01.
    Mlat (numpy.ndarray): 
        An array representing Magnetic Latitude (Mlat) values from 40 to 90.5 in steps of 0.15.
    ang (numpy.ndarray): 
        Array of angles in radians derived from MLT for calculations.
    chi (numpy.ndarray):
        Array of chi angles (90 - |Mlat|) for calculations.
    kp_m (numpy.ndarray):
        An array representing Kp magnetic activity index values.
    file_dir (str):
        Directory path for flux coefficient data files.
    file_dir2 (str):
        Directory path for mean coefficient data files.
    green_aurora_cmap (matplotlib.colors.LinearSegmentedColormap or None): 
        Colormap for visualizations.
    file_paths (dict): 
        Dictionary mapping Kp values to associated file paths.
    """

    def __init__(self):
        warnings.filterwarnings("ignore")
        self.MLT = np.arange(0, 24, 0.01)
        self.Mlat = np.arange(40, 90.5, 0.15)
        self.ang = self.MLT * 2 * np.pi / 24
        self.chi = 90 - np.abs(self.Mlat)
        self.kp_m = np.array([0.75, 2.25, 3.75, 5.25, 7, 9])
        self.file_dir = "./eflux_coeff/"
        self.file_dir2 = "./emean_coeff/"
        self.green_aurora_cmap = None  # Define your colormap here
        self.file_paths = {
            1.5: ("K0.txt", "K1.txt"),
            3: ("K0.txt", "K1.txt"),
            4.5: ("K1.txt", "K2.txt"),
            6: ("K2.txt", "K3.txt"),
            8: ("K3.txt", "K4.txt"),
            10: ("K4.txt", "K5.txt"),
        }

    def read_coeff(self, file):
        """
        Read the coefficients from a given file.

        Args:
            file (str): Path to the file containing the coefficients.

        Returns:
            tuple: A tuple containing constant, cosine, and sine coefficients.

        The coefficients are organized in the file with constant values in the first row,
        cosine coefficients in the next 6 rows, and sine coefficients in the remaining rows.
        """
        coeff = np.loadtxt(file, usecols=(1, 2, 3, 4))
        const = coeff[0]
        ind_cos = coeff[1:7]
        ind_sin = coeff[7:]
        return const, ind_cos, ind_sin

    def compute_coeff(self, file):
        """
        Compute the Epstein fitting coefficients from a given file.

        Args:
            file (str): The path to the file containing coefficients.

        Returns:
            numpy.ndarray: A 2D array of coefficients for different MLTs.

        This method reads the coefficients from a file and computes the coefficients for different MLTs.
        It first reads the constant, cosine, and sine coefficients using the read_coeff method.
        Then, it iterates through different MLTs and computes the coefficients for each MLT.
        The computed coefficients are stored in a 2D array and returned.
        """
        const, ind_cos, ind_sin = self.read_coeff(file)
        coefficients = np.zeros((4, len(self.MLT)))
        for i in range(4):
            CONST = const[i]
            for ij in range(6):
                k = ij + 1
                CONST = (
                    CONST
                    + ind_cos[ij, i] * np.cos(k * self.ang)
                    + ind_sin[ij, i] * np.sin(k * self.ang)
                )
            coefficients[i, :] = CONST
        return coefficients

    def flux_coeff(self, kp):
        """
        Calculate the flux coefficients for a given Kp index by determining
        the appropriate file paths based on the Kp value and then computing the coefficients
        using the compute_coeff method for the lower and upper ranges.

        Args:
            kp (float): The Kp index.

        Returns:
            tuple: A tuple containing two 2D arrays of lower and upper flux coefficients.
        """
        constL, constU = None, None

        # retrieve the file paths based on the value of kp
        for k, v in self.file_paths.items():
            if kp <= k:
                constL = self.compute_coeff(self.file_dir + v[0])
                constU = self.compute_coeff(self.file_dir + v[1])
                break

        return constL, constU

    def mean_coeff(self, kp):
        """
        Calculate the mean coefficients for a given Kp index by determining
        the appropriate file paths based on the Kp value and then computing the coefficients
        using the compute_coeff method for the lower and upper ranges.

        Args:
            kp (float): The Kp index.

        Returns:
            tuple: A tuple containing two 2D arrays of lower and upper mean coefficients.
        """
        constL, constU = None, None

        # retrieve the file paths based on the value of kp
        for k, v in self.file_paths.items():
            if kp <= k:
                constL = self.compute_coeff(self.file_dir2 + v[0])
                constU = self.compute_coeff(self.file_dir2 + v[1])
                break

        return constL, constU

    def kpm(self, kp):
        """
        Find the two adjacent Kp_model values (Kpm1 and Kpm2) that satisfy the two conditions.
        See Appendix A on the paper.

        Args:
            kp (float): The Kp index.

        Returns:
            tuple: A tuple containing two Kp values for interpolation.
        """
        if kp < 0.75:
            kpm1 = 0.75
            kpm2 = 2.25
        else:
            im1 = np.where(self.kp_m <= kp)
            im2 = np.where(self.kp_m > kp)
            kpm1 = self.kp_m[im1[0][-1]]
            kpm2 = self.kp_m[im2[0][0]]
        return kpm1, kpm2

    def hemispheric_power(self, kp):
        """
        Calculate the hemispheric power factors F1 and F2 for a given Kp index used to combine energy components in the Emean method.
        It calculates the hemispheric power for the given Kp index and adjacent Kp values (kpm1 and kpm2).
        Depending on the value of Kp, different equations are used to calculate hemispheric power.

        Args:
            kp (float): The Kp index for which to calculate the hemispheric power factors.

        Returns:
            tuple: A tuple containing the F1 and F2 hemispheric power factors.

        The calculated F1 and F2 factors are returned as a tuple.
        """
        kpm1, kpm2 = self.kpm(kp)
        if kp <= 5:
            HP = 38.66 * np.exp(0.1967 * kp) - 33.99  # -33.99
            HPm1 = 38.66 * np.exp(0.1967 * kpm1) - 33.99
            HPm2 = 38.66 * np.exp(0.1967 * kpm2) - 33.99
        else:
            HP = 4.592 * np.exp(0.4731 * kp) + 20.47  # +20.47
            HPm1 = 4.592 * np.exp(0.4731 * kpm1) + 20.47
            HPm2 = 4.592 * np.exp(0.4731 * kpm2) + 20.47
        F1 = (HPm2 - HP) / (HPm2 - HPm1)
        F2 = (HP - HPm1) / (HPm2 - HPm1)
        return F1, F2

    def Eflux(self, kp):
        """
        Do nonlinear interpolation for the energy flux and for a given Kp index.

        Args:
            kp (float): The Kp index for which to calculate the energy flux.

        Returns:
            np.ndarray: The energy flux for the given Kp index.

        This method calculates the energy flux using flux_coeff() and hemispheric_power() methods.
        It computes the energy values at each latitude using coefficients and Kp-dependent factors.
        The energy flux values are then combined based on F1 and F2 factors derived from hemispheric_power().
        The resulting energy flux is returned as a NumPy array.
        """
        kpm1, kpm2 = self.kpm(kp)
        f1 = (kpm2 - kp) / (kpm2 - kpm1)
        f2 = (kp - kpm1) / (kpm2 - kpm1)
        flux = np.full((len(self.MLT), len(self.Mlat)), np.nan)
        flux = []
        # Obtain lower and upper coefficients using flux_coeff() method
        L, U = self.flux_coeff(kp)
        # Iterate through coefficients and compute energy mean values
        for a, b, c, d, a2, b2, c2, d2 in zip(
            L[0], L[1], L[2], L[3], U[0], U[1], U[2], U[3]
        ):
            # Calculate energy values for both components
            Eom1 = (a * np.exp((self.chi - b) / c)) / (
                (1 + np.exp((self.chi - b) / d)) ** 2
            )
            Eom2 = (a2 * np.exp((self.chi - b2) / c2)) / (
                (1 + np.exp((self.chi - b2) / d2)) ** 2
            )
            flux.append(f1 * Eom1 + f2 * Eom2)
        return np.array(flux)

    def Emean(self, kp):
        """
        nonlinear interpolation for the energy flux and for a given Kp index using mean_coeff() and hemispheric_power() methods.
        It computes the energy values at each latitude using coefficients and Kp-dependent factors.
        The energy mean values are then combined based on F1 and F2 factors derived from hemispheric_power().

        Args:
            kp (float): The Kp index for which to calculate the mean energy.

        Returns:
            np.ndarray: The mean energy for the given Kp index.
        """
        emean = []
        # Obtain lower and upper coefficients using mean_coeff() method
        L, U = self.mean_coeff(kp)
        # Calculate F1 and F2 factors using hemispheric_power() method
        F1, F2 = self.hemispheric_power(kp)
        for a, b, c, d, a2, b2, c2, d2 in zip(
            L[0], L[1], L[2], L[3], U[0], U[1], U[2], U[3]
        ):
            Eom1 = (a * np.exp((self.chi - b) / c)) / (
                (1 + np.exp((self.chi - b) / d)) ** 2
            )
            Eom2 = (a2 * np.exp((self.chi - b2) / c2)) / (
                (1 + np.exp((self.chi - b2) / d2)) ** 2
            )
            emean.append(F1 * Eom1 + F2 * Eom2)
        # Eo = F1*Eom1+F2*Eom2
        return np.array(emean)

    def find_boundary_indices(self, array, value):
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

    def plot_kp(self, kp, savefig=False, cmap_upper=6):
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
        emean = self.Emean(kp)
        eflux = self.Eflux(kp)
        top_indices, bottom_indices = self.find_boundary_indices(eflux.T, 0.25)
        Lat = np.arange(40, 90.5, 0.15)  # for Southern hemisphere -90:0.5:-30
        Lon = np.arange(0, 360, 0.15)
        xlat, ylon = np.meshgrid(Lat, Lon)
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
            vmax=cmap_upper - 2,
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
    aurora_model = AuroraModel()
    aurora_model.plot_kp(3, savefig=True, cmap_upper=6)
