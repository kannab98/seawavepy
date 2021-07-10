from seawave.spectrum import spectrum as s
from .. import config
import numpy as np

from typing import Any


const = 11.04
class Dopler(object):
    def __init__(self) -> None:
        super().__init__()

    @property
    def width(self):
        return self.__omega_s__()

    @property
    def shift(self):
        return self.__omega_t__()

    def __call__(self, omega):
        theta0 = np.deg2rad(config['Radar']['Direction'][1])

        deltax = np.deg2rad(config['Radar']['GainWidth'][0])**2
        deltay = np.deg2rad(config['Radar']['GainWidth'][1])**2


        c = config['Constants']['WaveSpeed']

        if config['Radar']['WaveLength'] == "Ku":
            k = 2*np.pi/0.022

        elif config['Radar']['WaveLength'] == "Ka":
            k = 2*np.pi/0.008


        fresnel_coeff = 0.8

        alpha_n0 = self.__alpha_n0__()
        alpha_v0 = self.__alpha_v0__()
        alpha_r0 = self.__alpha_r0__()
        omega_s = self.__omega_s__()
        omega_t = self.__omega_t__()


        sigmayy = s.dblquad(0, 2, 0)
        sigmaxx = s.dblquad(2, 0, 0)
        Kxy = s.dblquad(1, 1, 0)
        S =  fresnel_coeff**2*np.sqrt(np.pi)/(2*k*np.cos(theta0)**4 * np.sqrt(alpha_n0))  \
        * np.exp(- (omega + k * omega_t)**2/(4*k**2*omega_s)) * \
            np.exp(- np.tan(theta0)**2/(2*alpha_n0) * sigmayy) * \
                (
                    np.exp(np.tan(theta0)**2/(2*const*alpha_n0**2)*
                        (
                            (sigmayy**2*deltax)/(alpha_v0) + 
                            (deltay*Kxy)/(alpha_r0 * np.cos(theta0)**2)
                        )
                    )
                ) / (
                    np.sqrt(omega_s) * 
                    np.sqrt(
                        (1 + sigmayy * deltax / (const*alpha_n0)) *
                        (1 + sigmaxx * deltay / (const*alpha_n0*np.cos(theta0)**2))
                    )
                )
        return S


    def __alpha_n0__(self):
        """
        Checked
        """
        Kxy = s.dblquad(1, 1, 0)
        sigmaxx = s.dblquad(2, 0, 0)
        sigmayy = s.dblquad(0, 2, 0)
        return sigmaxx*sigmayy - Kxy**2

    def __alpha_v0__(self):
        """
        Checked
        """
        sigmayy = s.dblquad(0,2,0)
        alpha_n0 = self.__alpha_n0__()
        deltax = np.deg2rad(config['Radar']['GainWidth'][0])**2
        return 1 + (sigmayy * deltax) / (const * alpha_n0)

    def __alpha_r0__(self):
        """
        Checked
        """
        theta0 = np.deg2rad(config['Radar']['Direction'][1])
        sigmaxx = s.dblquad(2, 0, 0)
        alpha_n0 = self.__alpha_n0__()
        deltay = np.deg2rad(config['Radar']['GainWidth'][1])**2
        return 1 + (sigmaxx * deltay) / (const * alpha_n0 * np.cos(theta0)**2)
    
    def __alpha_p__(self):
        """
        Checked
        """
        Kyt = s.dblquad(0, 1, 0, 1)
        Kxy = s.dblquad(1, 1, 0)
        Kxt = s.dblquad(1, 0, 0, 1)
        sigmaxx = s.dblquad(2, 0, 0)
        return 2*Kyt + (2*Kxy*Kxt)/sigmaxx

    def __alpha_u0__(self):
        """
        Checked
        """
        Kxt = s.dblquad(1, 0, 0, 1)
        Kxy = s.dblquad(1, 1, 0)
        sigmaxx = s.dblquad(2, 0, 0)
        alpha_n0 = self.__alpha_n0__()
        alpha_p = self.__alpha_p__()
        return (2*Kxt / sigmaxx) + (alpha_p * Kxy / alpha_n0)

    def __alpha_t__(self):
        """
        Checked
        """
        sigmatt = s.dblquad(0, 0, 0, 2)
        sigmaxx = s.dblquad(2, 0, 0)
        Kxt = s.dblquad(1, 0, 0, 1)
        return sigmatt*sigmaxx - Kxt**2
    

    def __alpha_y0__(self):
        """
        Checked
        """
        Kxy = s.dblquad(1, 1, 0)
        sigmaxx = s.dblquad(2, 0, 0)
        return 2*Kxy/sigmaxx

    def __alpha_00__(self):
        """
        Checked
        """
        alpha = self.__sigma_n0__()
        sigmayy = s.dblquad(0, 2, 0)
        theta0 = np.deg2rad(config['Radar']['Direction'][1])
        return 2*sigmayy/(np.cos(theta0) * alpha)
    
    def __omega_t__(self):
        theta0 = np.deg2rad(config['Radar']['Direction'][1])
        sigmaxx = s.dblquad(2, 0, 0)
        sigmayy = s.dblquad(0, 2, 0)
        alpha_n0 = self.__alpha_n0__()
        alpha_u0 = self.__alpha_u0__()
        alpha_r0 = self.__alpha_r0__()
        alpha_p = self.__alpha_p__()

        deltax = np.deg2rad(config['Radar']['GainWidth'][0])**2
        deltay = np.deg2rad(config['Radar']['GainWidth'][1])**2
        Kxt = s.dblquad(1, 0, 0, 1)
        Kxy = s.dblquad(1, 1, 0)


        return np.sin(theta0) * (
            + alpha_u0 
            - (alpha_n0 * deltax * sigmayy)/ (const*alpha_n0 + sigmayy*deltax) 
            - (alpha_p * sigmaxx * Kxy * deltay) / (const * alpha_n0**2 * alpha_r0 * np.cos(theta0)**2 )
        )

    def __omega_s__(self):
        deltax = np.deg2rad(config['Radar']['GainWidth'][0])**2
        deltay = np.deg2rad(config['Radar']['GainWidth'][1])**2
        sigmaxx = s.dblquad(2, 0, 0)
        alpha_n0 = self.__alpha_n0__()
        alpha_u0 = self.__alpha_u0__()
        alpha_r0 = self.__alpha_r0__()
        alpha_v0 = self.__alpha_v0__()
        alpha_p = self.__alpha_p__()
        alpha_t = self.__alpha_t__()

        theta0 = np.deg2rad(config['Radar']['Direction'][1])

        return (
            + deltay*alpha_p**2*sigmaxx**4/(2*const*alpha_r0*alpha_n0**2) 
            + np.cos(theta0)**2 * (
                + 2*alpha_t/sigmaxx 
                - alpha_p**2 * sigmaxx/(2*alpha_n0) 
                + alpha_u0**2 * deltax/(2*const * alpha_v0)
            )
        )

    



