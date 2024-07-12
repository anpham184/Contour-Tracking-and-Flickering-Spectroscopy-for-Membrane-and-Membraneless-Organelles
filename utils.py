import yaml
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from tifffile import TiffFile
from skimage import io
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.animation as animation

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def path(self):
        return self.config.get('path')

    @property
    def file_name(self):
        return self.config.get('file_name')

    @property
    def save_path(self):
        return self.config.get('save_path')

    @property
    def NrofFrames(self):
        return self.config.get('NrofFrames')

    
    @property
    def sigma(self):
        return self.config.get('sigma')

    @property
    def start_angle(self):
        return self.config.get('start_angle')

    @property
    def end_angle(self):
        return self.config.get('end_angle')

    @property
    def nr_of_angle(self):
        return self.config.get('nr_of_angle')

    @property
    def center(self):
        return self.config.get('center')

class ImageProcessing:
    def __init__(self, config, image_index):
        self.path = config.path
        self.sigma = config.sigma
        self.image_index = image_index
        self.image_stack = self.read_image()
        self.image = self.read_image()
        self.filtered_image = self.apply_gaussian_filter(self.image)
    def read_stack():
        return io.imread(self.path, plugin="tifffile")
    def read_image(self):
        # Open the TIFF file and read the specified image
        with TiffFile(self.path) as tif:
            #print(tif)
            return tif.pages[self.image_index].asarray()

    def apply_gaussian_filter(self, image):
        # Apply a Gaussian filter to the image
        return gaussian_filter(image, sigma=self.sigma)

class ContourTracker_Membrane:
    def __init__(self, image, center, angle_step_size):
        self.image = image
        self.center = center
        self.angle_step_size = angle_step_size

    def line_profile(self, index):
        max_distance = int(np.ceil(np.sqrt(self.center[1]**2 + self.center[0]**2)))
        x1 = self.center[1] + max_distance * np.cos(self.angle_step_size * index)
        y1 = self.center[0] + max_distance * np.sin(self.angle_step_size * index)

        xx = np.clip(x1, 0, self.image.shape[1] - 1).astype(int)
        yy = np.clip(y1, 0, self.image.shape[0] - 1).astype(int)
        length = int(np.hypot(xx - self.center[1], yy - self.center[0]))
        x, y = np.linspace(self.center[1], xx, length), np.linspace(self.center[0], yy, length)

        z = np.asarray(self.image)
        zi = z[y.astype(int), x.astype(int)]
        profile = np.zeros((length, 2))
        profile[:, 0] = np.linspace(0, length, length).T
        profile[:, 1] = zi

        return profile

    def search_coordinates(self, index):
        profile_z = self.line_profile(index)
        max_search = np.argmax(np.max(profile_z, axis=1))
        
        popt, pcov = curve_fit(
            self.half_gaussian,
            profile_z[(max_search-3):, 0],
            profile_z[(max_search-3):, 1],
            p0=[35000, 100, 80, 10],
            maxfev=1000000
        )
        '''
        if index == 1160:
            plt.figure(figsize=(2, 2))
            plt.plot(profile_z[:, 0], profile_z[:, 1], 'o', markersize=2, markeredgewidth=1, markeredgecolor='k', markerfacecolor='None')
            plt.plot(profile_z[:, 0], self.half_gaussian(profile_z[:, 0], *popt), '--r', label='fit')
            plt.xlabel('R (Pixel)')
            plt.ylabel('Intensity (A.U)')
            plt.xlim(62, 85)
            plt.ylim(500, 40000)
            plt.savefig('Gaussian_Fit_Zoom.jpg', dpi=300)
            plt.show()
        '''
        sigma = popt[3]
        mu = popt[2]
        #FWHM = np.absolute(2.35482 * sigma)
        if mu > 0 and sigma < 11 and sigma > 0:
            point_x = self.center[1] + mu * np.cos(self.angle_step_size * index)
            point_y = self.center[0] + mu * np.sin(self.angle_step_size * index)
            
        else:
            point_x = math.nan
            point_y = math.nan

        return point_x, point_y

    @staticmethod
    def half_gaussian(x, p, base, mu, sigma):
        return (p - base) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + base

class ContourTracker_Membraneless:
    def __init__(self, image, center, angle_step_size):
        self.image = image
        self.center = center
        self.angle_step_size = angle_step_size

    def line_profile(self, index):
        max_distance = int(np.ceil(np.sqrt(self.center[1]**2 + self.center[0]**2)))
        x1 = self.center[1] + max_distance * np.cos(self.angle_step_size * index)
        y1 = self.center[0] + max_distance * np.sin(self.angle_step_size * index)

        xx = np.clip(x1, 0, self.image.shape[1] - 1).astype(int)
        yy = np.clip(y1, 0, self.image.shape[0] - 1).astype(int)
        length = int(np.hypot(xx - self.center[1], yy - self.center[0]))
        x, y = np.linspace(self.center[1], xx, length), np.linspace(self.center[0], yy, length)

        z = np.asarray(self.image)
        zi = z[y.astype(int), x.astype(int)]
        profile = np.zeros((length, 2))
        profile[:, 0] = np.linspace(0, length, length).T
        profile[:, 1] = zi

        return profile

    def search_coordinates(self, index):
        profile_z = self.line_profile(index)
        profile_z[:,1] = np.gradient(profile_z[:,1])
        popt, pcov  = curve_fit(self.gaussian, profile_z[:,0], profile_z[:,1], p0=[-1500,80, 10], bounds=([-12000, 60,1],[-10, 120, 15]), maxfev=1000000)
        '''
        if Index==192:
            plt.figure()
            plt.plot(profile_z[:,1] )
            plt.plot(profile_z[:,0], gauss( profile_z[:,0], *popt))
            plt.show()
            print(popt)
        '''
    
        point_x = self.center[1] + popt[1]*np.cos(self.angle_step_size * index)
        point_y = self.center[0] + popt[1]*np.sin(self.angle_step_size * index)
    
        return point_x, point_y

    @staticmethod
    def gaussian(x, A, x0, sigma): 
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

class PostProcessing:
    def __init__(self, filtered_imgs, membraneCoordinatesX, membraneCoordinatesY, save_path, file_name):
        self.filtered_imgs = np.array(filtered_imgs)  # Convert to numpy array
        self.membraneCoordinatesX = membraneCoordinatesX
        self.membraneCoordinatesY = membraneCoordinatesY
        self.save_path = save_path
        self.file_name = file_name
        self.fig, self.ax = plt.subplots()
        


    @staticmethod
    def fill_nan_interpolation(data):
        filled_data = np.copy(data)
        nrows, ncols = data.shape

        for i in range(nrows):
            row = data[i]
            nan_indices = np.isnan(row)
            if np.any(nan_indices):
                non_nan_indices = np.where(~nan_indices)[0]
                f = interpolate.interp1d(non_nan_indices, row[~nan_indices], kind='cubic', fill_value='extrapolate')
                filled_data[i, nan_indices] = f(np.where(nan_indices)[0])
    
        return filled_data

    def save_coordinates(self):
        np.savetxt(f'{self.save_path}/membraneCoordinatesX_{self.file_name}.txt', self.membraneCoordinatesX)
        np.savetxt(f'{self.save_path}/membraneCoordinatesY_{self.file_name}.txt', self.membraneCoordinatesY)
    
    def update(self, frame):
        self.ax.clear()  # Clear the previous frame
        self.ax.imshow(self.filtered_imgs[frame])  # Display the filtered image
        # Plot membrane coordinates for the current frame
        self.ax.plot(self.membraneCoordinatesX[:, frame], self.membraneCoordinatesY[:, frame], color='red', linewidth=2)
    
    def create_animation(self, fps=30, bitrate=1800, dpi=600):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.filtered_imgs.shape[0], interval=100, repeat=True)
        self.fig.set_size_inches(4, 4)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnPham'), bitrate=bitrate)
        ani.save(f'{self.save_path}/animation_{self.file_name}.mp4', writer=writer, dpi=dpi)
        plt.show()