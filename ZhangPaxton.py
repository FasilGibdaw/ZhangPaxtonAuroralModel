import pandas as pd
import numpy as np

def ZhangPaxton(mlat,mlt, kp):
    """ Calculate electron flux and mean energy using Zhang-Paxton model.
    Based on the paper: 
        Y. Zhang, L.J. Paxton, An empirical Kp-dependent global auroral model based on TIMED/GUVI FUV data, Journal of
        Atmospheric and Solar-Terrestrial Physics, Volume 70, Issues 8–9, 2008, Pages 1231-1242, ISSN 1364-6826, 
        https:doi.org/10.1016/j.jastp.2008.03.008.

    Args:
        mlat (array): magnetic latitude
        mlt (array): magnetic local time
        kp (float): kp index

    Returns:
        eflux (array): electron flux at given mlat, mlt and kp
        emean (array): mean energy at given mlat, mlt and kp
    """
    
    ang = mlt * 2 * np.pi / 24
    chi = 90 - np.abs(mlat)
    eflux_coeff_L, eflux_coeff_U = get_coeff(kp, 'eflux')
    const = eflux_coeff_L.iloc[0].values[2:]
    ind_cos = eflux_coeff_L.iloc[1:7].values[:,2:]
    ind_sin = eflux_coeff_L.iloc[7:].values[:,2:]
    L = [const[i] + sum(ind_cos[ij, i] * np.cos((ij + 1) * ang) + ind_sin[ij, i] * np.sin((ij + 1) * ang) for ij in range(6)) for i in range(4)]
    const = eflux_coeff_U.iloc[0].values[2:]
    ind_cos = eflux_coeff_U.iloc[1:7].values[:,2:]
    ind_sin = eflux_coeff_U.iloc[7:].values[:,2:]
    U = [const[i] + sum(ind_cos[ij, i] * np.cos((ij + 1) * ang) + ind_sin[ij, i] * np.sin((ij + 1) * ang) for ij in range(6)) for i in range(4)]

    # eflux calculation
    kpm1, kpm2 = kpm(kp)
    f1, f2 = (kpm2 - kp) / (kpm2 - kpm1), (kp - kpm1) / (kpm2 - kpm1)
    # L, U = flux_coeff(kp, ang)
    Eom1_L = (L[0] * np.exp((chi - L[1]) / L[2])) / ((1 + np.exp((chi - L[1]) / L[3]))**2)
    Eom1_U = (U[0] * np.exp((chi - U[1]) / U[2])) / ((1 + np.exp((chi - U[1]) / U[3]))**2)
    eflux = f1 * Eom1_L + f2 * Eom1_U
    ## Emean calculation
    emean_coeff_L, emean_coeff_U = get_coeff(kp, 'emean')
    const = emean_coeff_L.iloc[0].values[2:]
    ind_cos = emean_coeff_L.iloc[1:7].values[:,2:]
    ind_sin = emean_coeff_L.iloc[7:].values[:,2:]
    L = [const[i] + sum(ind_cos[ij, i] * np.cos((ij + 1) * ang) + ind_sin[ij, i] * np.sin((ij + 1) * ang) for ij in range(6)) for i in range(4)]
    const = emean_coeff_U.iloc[0].values[2:]
    ind_cos = emean_coeff_U.iloc[1:7].values[:,2:]
    ind_sin = emean_coeff_U.iloc[7:].values[:,2:]
    U = [const[i] + sum(ind_cos[ij, i] * np.cos((ij + 1) * ang) + ind_sin[ij, i] * np.sin((ij + 1) * ang) for ij in range(6)) for i in range(4)]

    F1, F2 = hemispheric_power(8.99)
    Eom1_L = (L[0] * np.exp((chi - L[1]) / L[2])) / \
        ((1 + np.exp((chi - L[1]) / L[3]))**2)
    Eom1_U = (U[0] * np.exp((chi - U[1]) / U[2])) / \
        ((1 + np.exp((chi - U[1]) / U[3]))**2)

    emean = F1 * Eom1_L + F2 * Eom1_U
    return eflux, emean

def kpm(kp):
    """Find adjacent Kp_model values."""
    kp_m = np.array([0.75, 2.25, 3.75, 5.25, 7, 9])
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
def get_coeff(kp, input_file = 'emean'):
    KK = [1.5, 3, 4.5, 6, 8, 10]
    if input_file == 'emean':
        coeff = pd.read_csv('./data/zhang_paxton_emean_coeff.txt', sep='\t')
    elif input_file == 'eflux':
        coeff = pd.read_csv('./data/zhang_paxton_eflux_coeff.txt', sep='\t')
    else:
        print('Epistein coefficients not found')
    for i, k in enumerate(KK):
        if kp <= k and kp >= 1.5:
            coeff_L = coeff[coeff.Kp == 'K'+str(i-1)]
            coeff_U = coeff[coeff.Kp == 'K'+str(i)]
            break
        else:
            coeff_L = coeff[coeff.Kp == 'K0']
            coeff_U = coeff[coeff.Kp == 'K1']
    return coeff_L, coeff_U

def ZhangPaxton_conductance(mlat, mlt, kp):
    # calculates the conductance (pedersen and hall) for a given Kp index using Robinson 1987 paper
    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA092iA03p02565
    eflux, emean = ZhangPaxton(mlat, mlt, kp)
    pedersen_conductance = (
        (40 * emean)/(16 + emean**2))*(eflux**(1/2))
    hall_conductance = (0.45 * (emean**0.85)
                        )*pedersen_conductance
    return hall_conductance, pedersen_conductance
