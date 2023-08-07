from cmath import inf
import numpy as np
from os import listdir
from os.path import isdir, join
import json

from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares, minimize

        
def get_available_sites(path):
    fluxsites = [f[4:] for f in listdir(path) if isdir(join(path, f)) and f.startswith("FLX")]
    return fluxsites

def get_igbp_of_site(site):
    IGBP_of_site = {'ES-Abr': 'SAV',
                    'DE-Hai': 'DBF',
                    'FR-FBn': 'MF',
                    'DE-Hzd': 'DBF',
                    'CH-Cha': 'GRA',
                    'AU-Cpr': 'SAV',
                    'AU-DaP': 'GRA',
                    'AU-Dry': 'SAV',
                    'AU-How': 'WSA',
                    'AU-Stp': 'GRA',
                    'BE-Lon': 'CRO',
                    'BE-Vie': 'MF',
                    'CA-Qfo': 'ENF',
                    'DE-Geb': 'CRO',
                    'DE-Gri': 'GRA',
                    'DE-Kli': 'CRO',
                    'DE-Obe': 'ENF',
                    'DE-Tha': 'ENF',
                    'DK-Sor': 'DBF',
                    'FI-Hyy': 'ENF',
                    'FR-LBr': 'ENF',
                    'GF-Guy': 'EBF',
                    'IT-BCi': 'CRO',
                    'IT-Cp2': 'EBF',
                    'IT-Cpz': 'EBF',
                    'IT-MBo': 'GRA',
                    'IT-Noe': 'CSH',
                    'IT-Ro1': 'DBF',
                    'IT-SRo': 'ENF',
                    'NL-Loo': 'ENF',
                    'RU-Fyo': 'ENF',
                    'US-ARM': 'CRO',
                    'US-GLE': 'ENF',
                    'US-MMS': 'DBF',
                    'US-NR1': 'ENF',
                    'US-SRG': 'GRA',
                    'US-SRM': 'WSA',
                    'US-UMB': 'DBF',
                    'US-Whs': 'OSH',
                    'US-Wkg': 'GRA',
                    'ZA-Kru': 'SAV',
    }
    return IGBP_of_site[site]
