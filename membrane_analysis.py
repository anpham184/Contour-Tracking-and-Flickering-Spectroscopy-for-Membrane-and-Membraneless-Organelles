import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import pyshtools as pysh

class MembraneAnalysis:
    def __init__(self, file_x, file_y, num_modes=100, lmax=400, lmin=60, pixel_scale=0.078e-6):
        self.file_x = file_x
        self.file_y = file_y
        self.num_modes = num_modes
        self.lmax = lmax
        self.lmin = lmin
        self.pixel_scale = pixel_scale
        self.mean_R = None
        self.spectra = None
        self.R0_values = None
        self.fitted_kappa = None

    def read_coordinates(self):
        x_coords = np.loadtxt(self.file_x, delimiter=' ')
        y_coords = np.loadtxt(self.file_y, delimiter=' ')
        return x_coords, y_coords

    def Center(self, xymempos):
        center = np.mean(xymempos[:, 0]), np.mean(xymempos[:, 1])
        return center

    def Cart2Pol(self, xymempos, Center):
        mempos = xymempos - Center
        radii = np.sqrt(mempos[:, 0] ** 2 + mempos[:, 1] ** 2)
        angles = np.arctan2(mempos[:, 1], mempos[:, 0])
        angles[angles <= 0] += 2 * np.pi
        split_index = np.argmin(angles)
        if split_index != 0:
            angles = np.append(angles[split_index:], angles[:split_index])
            radii = np.append(radii[split_index:], radii[:split_index])
        mempospol = np.column_stack((radii, angles))
        return mempospol

    def estimate_R0(self, mempolpos):
        mempolposlength = len(mempolpos)
        tmp = 0
        for i in range(mempolposlength - 1):
            tmp += (mempolpos[i, 0] + mempolpos[i + 1, 0]) * abs(mempolpos[i + 1, 1] - mempolpos[i, 1])
            if i == mempolposlength - 2:
                tmp += (mempolpos[-1, 0] + mempolpos[0, 0]) * abs(2 * np.pi + mempolpos[0, 1] - mempolpos[-1, 1])
        return (1 / (4 * np.pi)) * tmp

    def fourier_coefficients(self, mempos, r0):
        a = np.zeros((self.num_modes - 1, 1))
        b = np.zeros((self.num_modes - 1, 1))
        c = np.zeros((self.num_modes - 1, 1))
        theta = mempos[:, 1]
        for i in range(2, self.num_modes + 1):
            a_n = (1 / (np.pi * r0)) * np.trapz(mempos[:, 0] * np.cos(i * theta), theta)
            b_n = (1 / (np.pi * r0)) * np.trapz(mempos[:, 0] * np.sin(i * theta), theta)
            c_n = np.sqrt(a_n ** 2 + b_n ** 2)
            a[i - 2, 0] = a_n
            b[i - 2, 0] = b_n
            c[i - 2, 0] = c_n
        return a, b, c

    def get_power_spectrum(self):
        x_coords, y_coords = self.read_coordinates()
        num_frames = x_coords.shape[1]
        xymempos = np.zeros((x_coords.shape[0], 2))
        self.R0_values = np.zeros((num_frames, 1))
        a_n = np.zeros((self.num_modes - 1, num_frames))
        b_n = np.zeros((self.num_modes - 1, num_frames))
        c_n = np.zeros((self.num_modes - 1, num_frames))

        for i in tqdm(range(num_frames)):
            xymempos[:, 0] = x_coords[:, i]
            xymempos[:, 1] = y_coords[:, i]

            Center = self.Center(xymempos)
            polarcoordinates = self.Cart2Pol(xymempos, Center)
            R0 = self.estimate_R0(polarcoordinates)
            self.R0_values[i, 0] = R0
            polarcoordinates[:, 0] -= R0
            a, b, c = self.fourier_coefficients(polarcoordinates, R0)
            a_n[:, i] = a[:, 0]
            b_n[:, i] = b[:, 0]
            c_n[:, i] = c[:, 0]

        u_n = np.var(c_n, axis=1)
        modes_Nr = np.arange(2, self.num_modes + 1, 1).T
        self.spectra = np.column_stack((modes_Nr, u_n))
        self.mean_R = np.mean(self.R0_values) * self.pixel_scale

    def fit_kappa(self, x, kappa):
        kb = 1.380649e-23
        T = 310
        v_q_2 = []
        for q in x:
            v_q = 0
            for l in range(int(q), self.lmax + 1):
                norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
                v_q += (kb * T / kappa) * ((2 * l + 1) / (4 * np.pi)) * norm_leg**2 / ((l - 1) * (l + 2) * (l * (l + 1)))
            v_q_2.append(v_q)
        return v_q_2

    def fit_sigma(self, x, _sigma):
        kb = 1.380649e-23
        T = 310
        kappa = self.fitted_kappa
        v_q_2 = []
        for q in x:
            v_q = 0
            for l in range(int(q), self.lmax + 1):
                norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
                v_q += (kb * T / kappa) * ((2 * l + 1) / (4 * np.pi)) * norm_leg**2 / ((l - 1) * (l + 2) * (l * (l + 1) + _sigma * self.mean_R**2 / kappa))
            v_q_2.append(v_q)
        return v_q_2

    def fit_sigma_kappa(self, x, _sigma, kappa):
        kb = 1.380649e-23
        T = 310
        v_q_2 = []
        for q in x:
            v_q = 0
            for l in range(int(q), self.lmax + 1):
                norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
                v_q += (kb * T / kappa) * ((2 * l + 1) / (4 * np.pi)) * norm_leg**2 / ((l - 1) * (l + 2) * (l * (l + 1) + _sigma * self.mean_R**2 / kappa))
            v_q_2.append(v_q)
        return v_q_2

    def analyze_membrane(self):
        self.get_power_spectrum()

        # Initial kappa value
        self.fitted_kappa = 0.1

        # Fit kappa
        print("Fitting kappa...")
        popt_kappa, pcov_kappa = curve_fit(self.fit_kappa, self.spectra[self.lmin - 2:, 0], self.spectra[self.lmin - 2:, 1], p0=[8e-20])
        self.fitted_kappa = np.float64(popt_kappa[0])
        fitted_kappa_curve = self.fit_kappa(self.spectra[:, 0], popt_kappa)
        print(f"Finished fitting kappa: {self.fitted_kappa}")

        # Fit sigma
        print("Fitting sigma...")
        popt_sigma, pcov_sigma = curve_fit(self.fit_sigma, self.spectra[2:self.lmin - 2, 0], self.spectra[2:self.lmin - 2, 1], p0=[1e-6])
        fitted_sigma_kappa_curve = self.fit_sigma_kappa(self.spectra[:, 0], popt_sigma, popt_kappa)
        print(f"Finished fitting sigma: {popt_sigma}")

        # Print and plot results
        kb = 1.380649e-23
        T = 310
        print(f"Bending stiffness: {self.fitted_kappa / (kb * T)} kbT")
        print(f"Surface tension: {popt_sigma[0] / 1e-6} uN/m")

        plt.figure(figsize=(6, 4))
        plt.plot(self.spectra[:, 0], self.spectra[:, 1], 'o', markerfacecolor='None', markeredgecolor='k', label='Data')
        plt.plot(self.spectra[:, 0], fitted_kappa_curve, 'r-', label='Fitted kappa')
        plt.plot(self.spectra[:, 0], fitted_sigma_kappa_curve, 'b-', label='Fitted_sigma_kappa')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$q$')
        plt.ylabel('$\\langle|u_{q}|^2\\rangle$')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_x = './output/membraneCoordinatesX_nuclear_membrane.txt'
    file_y = './output/membraneCoordinatesY_nuclear_membrane.txt'
    membrane_analysis = MembraneAnalysis(file_x, file_y)
    membrane_analysis.analyze_membrane()
