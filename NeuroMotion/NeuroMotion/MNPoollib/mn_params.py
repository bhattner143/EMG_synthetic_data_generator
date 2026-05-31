import numpy as np


# Range of MU depth and MU angle within each muscle
DEPTH = {
    'ECRB': [0.0130, 0.0220],
    'ECRL': [0.0085, 0.0153],
    'PL': [0.0071, 0.0114],
    'FCU_u': [0.0084, 0.0168],
    'FCU_h': [0.0074, 0.0165],
    'ECU': [0.0092, 0.0168],
    'EDI': [0.0079, 0.0171],
    'FDSI': [0.0169, 0.0231],
    'FCU': [0.0074, 0.0168],
    # Upper-limb muscles (MoBL-ARMS 4.1) — depth estimated from cadaveric anatomy
    'BIClong': [0.0150, 0.0250],
    'BICshort': [0.0120, 0.0210],
    'BRA':      [0.0100, 0.0200],
    'BRD':      [0.0090, 0.0180],
    'TRIlong':  [0.0130, 0.0230],
    'TRIlat':   [0.0110, 0.0210],
    'TRImed':   [0.0100, 0.0190],
    'DELT1':    [0.0160, 0.0280],
    'DELT2':    [0.0160, 0.0280],
    'DELT3':    [0.0120, 0.0220],
    'FCR':      [0.0080, 0.0160],
}

ANGLE = {
    'ECRB': [0.4946, 0.6632],
    'ECRL': [0.4607, 0.5109],
    'PL': [0.0540, 0.0956],
    'FCU_u': [0.7878, 0.8658],
    'FCU_h': [0.9897, 1.0],
    'ECU': [0.7194, 0.7779],
    'EDI': [0.5637, 0.6826],
    'FDSI': [0.1471, 0.2264],
    'FCU': [0.7878, 1.0],
    # Upper-limb muscles — angle estimated from anatomical cross-sections
    'BIClong': [0.3000, 0.5000],
    'BICshort': [0.2500, 0.4500],
    'BRA':      [0.4000, 0.6000],
    'BRD':      [0.3500, 0.5500],
    'TRIlong':  [0.6000, 0.8000],
    'TRIlat':   [0.5500, 0.7500],
    'TRImed':   [0.5000, 0.7000],
    'DELT1':    [0.2000, 0.4000],
    'DELT2':    [0.2500, 0.4500],
    'DELT3':    [0.7000, 0.9000],
    'FCR':      [0.4500, 0.6000],
}

# MS_AREA = F_max / sigma_0, sigma_0 = 0.61 N/mm^2 (MoBL-ARMS 4.1 specific tension)
MS_AREA = {
    'ECRB': 60.100,
    'ECRL': 70.715,
    'PL': 35.335,
    'FCU_u': 78.925,
    'FCU_h': 83.305,
    'ECU': 47.720,
    'EDI': 60.000,
    'FDSI': 25.335,
    'FCU': 162.23,
    # Upper-limb: F_max(N) / 0.61 (N/mm^2) — F_max from MOBL_ARMS_41.osim
    'BIClong': 525.1 / 0.61,   # 860.8 mm^2
    'BICshort': 316.8 / 0.61,  # 519.3 mm^2
    'BRA':      1177.4 / 0.61, # 1930.2 mm^2
    'BRD':      276.0 / 0.61,  # 452.5 mm^2
    'TRIlong':  771.8 / 0.61,  # 1265.2 mm^2
    'TRIlat':   717.5 / 0.61,  # 1176.2 mm^2
    'TRImed':   717.5 / 0.61,  # 1176.2 mm^2
    'DELT1':    1218.9 / 0.61, # 1998.2 mm^2
    'DELT2':    1103.5 / 0.61, # 1809.0 mm^2
    'DELT3':    201.6 / 0.61,  # 330.5 mm^2
    'FCR':      407.9 / 0.61,  # 668.7 mm^2
}  # mm^2

# MU counts from Enoka & Fuglevand (2001) and cadaveric literature
NUM_MUS = {
    'ECRB': 186,
    'ECRL': 204,
    'PL': 164,
    'FCU_u': 205,
    'FCU_h': 217,
    'ECU': 180,
    'EDI': 186,
    'FDSI': 158,
    'FCU': 422,
    # Upper-limb muscles — Enoka & Fuglevand 2001 / Brown et al. 1988
    'BIClong': 750,
    'BICshort': 450,
    'BRA':      800,
    'BRD':      330,
    'TRIlong':  600,
    'TRIlat':   550,
    'TRImed':   400,
    'DELT1':    450,
    'DELT2':    450,
    'DELT3':    200,
    'FCR':      300,
}

# Settings below match iEMG package (git@github.com:smetanadvorak/iemg_simulator.git)
mn_default_settings = {
    'rr': 50,
    'rm': 0.75,
    'rp': 100,
    'pfr1': 40,
    'pfrd': 10,
    'mfr1': 10,
    'mfrd': 5,
    'gain': 30,     # 0.3 per % MVC
    'c_ipi': 0.1,
    'frs1': 50,
    'frsd': 20,
    'fibre_density': 200,
}

# Settings for AC's motoneuron pool model
FCU_total_size = MS_AREA['FCU_u'] + MS_AREA['FCU_h']

PCSA = np.array([252.5, 337.3, 101, 479.8 * MS_AREA['FCU_u']/FCU_total_size, 479.8 * MS_AREA['FCU_h'] / FCU_total_size, 192.9, 39.40 + 109.2 + 94.4 + 48.8 + 72.4, 162.5]) / 50.8  # max iso forces from the MSK model / specific tension of 50.8N/cm2.

# correction coeff from cadeveric results for FDI
nb_fibres_cadaveric = 40500
PCSA_FDI = 60.7 / 50.8  # of the FDI from MSK values
nb_estimated = (PCSA_FDI / (np.pi * (26 * 10**-4)**2 / 4))
correction_factor = nb_fibres_cadaveric / nb_estimated

avg_fibre_diam = 34 * 10**-4  # 34 micrometers, scaled in centimeters, taken from cadaveric literature. Fuglevand uses 46um. 
avg_fibre_CSA = np.pi * avg_fibre_diam**2 / 4  # gives a density of 86 fibres /mm2. Fuglevand uses 20.
nb_fibres_per_muscle = (PCSA / avg_fibre_CSA).astype(int)  # gives between 170k and 600k fibres within the muscles, which is quite conservative. 
nb_fibres_per_muscle = (nb_fibres_per_muscle * correction_factor).astype(int) # Now 40k-140k fibres per muscle, sounds good
NUM_FIBRES_MS = {
    'ECRB': nb_fibres_per_muscle[0],
    'ECRL': nb_fibres_per_muscle[1],
    'PL': nb_fibres_per_muscle[2],
    'FCU_u': nb_fibres_per_muscle[3],
    'FCU_h': nb_fibres_per_muscle[4],
    'ECU': nb_fibres_per_muscle[5],
    'EDI': nb_fibres_per_muscle[6],
    'FDSI': nb_fibres_per_muscle[7],
}