# A Python library for processing and analysing images from the
# Aberystwyth University PanCam Emulator, AUPE.
#
# Roger Stabbins
# Natural History Museum, London
# 9/5/2025

from pathlib import Path
from typing import Dict, List, Literal, Tuple
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import scipy
import PIL.Image

LEVEL_DICT = {
    'raw': 'raw image no stretch',
    'bps': 'brightest pixel stretch',
    '99p': '99th percentile stretch',
    'wps': 'white patch stretch',
    'ctc': 'calibration target colour',
    'ctr': 'calibration target reflectance'
}

# WP_RED = 245
# WP_GREEN = 245
# WP_BLUE = 243

WP_RED = 243
WP_GREEN = 244
WP_BLUE = 243

BIT_DEPTH = 8

# TODO put this as an AUPE class, and load this info from csv file
# be sure to include a date stamp and version number for this AUPE information

# define a class for parsing a directory of AUPE images. Accepts the SOL and scene,
# and finds the directory, then gives a list of images in the directory, and 
# sorts the list into HRC, WAC RGB and 
# WAC MS image sets.
class AupeIO:
    def __init__(self, sol: str, scene: str, trial: str=None):
        self.sol = sol
        self.scene = scene
        self.trial = trial

        if trial is not None:
            self.scene_dir = Path('..', '..', 'data', sol, scene, trial)
        else:
            self.scene_dir = Path('..', '..', 'data', sol, scene)
        # build the processed data directory

        self.hrc_out_dir = None
        self.wac_rgb_out_dir = None
        self.wac_ms_out_dir = None

        self.build_scene_dir()
        
        self.sort_images()

        png_files = list(self.scene_dir.glob("*.png"))
        # sort by name
        png_files.sort()

        self.hrc_files = []
        self.lwac_rgb_files = []
        self.rwac_rgb_files = []
        self.wac_ms_files = []
        self.sort_images()
        

    def build_scene_dir(self):
        """Build the scene directory from the sol and scene
        """
        # check if the directory exists
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Directory {self.scene_dir} does not exist")
        # build the output directory
        if self.trial is not None:
            self.out_dir = Path('..', '..', 'processed', self.sol, self.scene, self.trial)
        else:
            self.out_dir = Path('..', '..', 'processed', self.sol, self.scene, 'Trial1')
            self.trial = 'Trial1'

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # create the subdirectories for HRC, WAC RGB and WAC MS outputs
        hrc_dir = Path(self.out_dir, 'HRC')
        hrc_dir.mkdir(parents=True, exist_ok=True)
        wac_rgb_dir = Path(self.out_dir, 'WAC_RGB')
        wac_rgb_dir.mkdir(parents=True, exist_ok=True)
        wac_ms_dir = Path(self.out_dir, 'WAC_MS')
        wac_ms_dir.mkdir(parents=True, exist_ok=True)

        # set the output directories
        self.hrc_out_dir = hrc_dir
        self.wac_rgb_out_dir = wac_rgb_dir
        self.wac_ms_out_dir = wac_ms_dir

    def sort_images(self):
        """Load the images from the directory and sort them into HRC, WAC RGB and WAC MS
        """
        # get all the images in the directory
        png_files = list(self.scene_dir.glob("*.png"))
        # sort by name
        png_files.sort()  

        # if the filename inlcudes the string 'HRC' put in list hrc_imgs
        hrc_files = []
        # if the filename includes any of 'WAC_2', 'WAC_3', 'WAC_4' put in list wac_rgb_imgs
        lwac_rgb_files = []
        rwac_rgb_files = []
        # else, put the filename in list wac_ms_imgs
        wac_ms_files = []

        for png_file in png_files:
            if 'HRC' in png_file.name:
                hrc_files.append(png_file)
            elif 'LWAC2_' in png_file.name or 'LWAC3_' in png_file.name or 'LWAC4_' in png_file.name:
                lwac_rgb_files.append(png_file)
            elif 'RWAC2_' in png_file.name or 'RWAC3_' in png_file.name or 'RWAC4_' in png_file.name:
                rwac_rgb_files.append(png_file)
            else:
                wac_ms_files.append(png_file)      
          
        # stash the lists
        self.hrc_files = hrc_files
        self.lwac_rgb_files = lwac_rgb_files 
        self.rwac_rgb_files = rwac_rgb_files
        self.wac_rgb_files = lwac_rgb_files + rwac_rgb_files
        self.wac_ms_files = wac_ms_files # TODO sort this into LWAC and RWAC
    
    def file_dict(self, filepath: Path, img_type: Literal['HRC', 'WAC_RGB', 'WAC_MS']) -> Dict:
        """For the given filepath, return a dictionary giving:
        - full file path
        - file name
        - sol
        - scene
        - trial
        - output directory

        :param filepath: file path to the image
        :type filepath: Path
        :return: File information needed to process the image
        :rtype: Dict
        """        
        file_dict = {}
        file_dict['filepath'] = filepath
        file_dict['trial'] = self.trial
        file_dict['scene'] = self.scene
        file_dict['sol'] = self.sol
        if img_type == 'HRC':
            file_dict['img_type'] = 'HRC'
            file_dict['out_dir'] = self.hrc_out_dir
        elif img_type == 'WAC_RGB':
            file_dict['img_type'] = 'WAC_RGB'
            file_dict['out_dir'] = self.wac_rgb_out_dir
        elif img_type == 'WAC_MS':
            file_dict['img_type'] = 'WAC_MS'
            file_dict['out_dir'] = self.wac_ms_out_dir
        
        return file_dict
    
    def wac_rgb_file_dict(self, wac_rgb_files: List) -> List:
        """For the given list of WAC RGB files, return a list of dictionaries
        giving the file information for each file.

        :param wac_rgb_files: list of WAC RGB files
        :type wac_rgb_files: List
        :return: list of dictionaries with file information
        :rtype: List
        """        
        file_dicts = []
        for wac_rgb_file in wac_rgb_files:
            file_dict = self.file_dict(wac_rgb_file, 'WAC_RGB')
            file_dicts.append(file_dict)
        
        # sort the list of dictionaries by the file name descending
        file_dicts.sort(key=lambda x: x['filepath'].name, reverse=False)
        
        return file_dicts
    
    def hrc_file_dict(self, hrc_file: Path) -> Dict:
        """For the given HRC file, return a dictionary
        giving the file information for the file, duplicated into each
        of the red, green and blue slots.

        :param hrc_file: HRC file
        :type hrc_file: Path
        :return: dictionary with file information
        :rtype: Dict
        """        
        file_dicts = [
            self.file_dict(hrc_file, 'HRC'),
            self.file_dict(hrc_file, 'HRC'),
            self.file_dict(hrc_file, 'HRC')
        ]
        return file_dicts

class AupeInfo:
    def __init__(self, filepath: Path):
        """Holds AUPE information not included in the 
        image metadata, namely mapping filter ids to
        cwl and fwhm. 
        
        Note, these values change between different versions of AUPE 
        and previous datasets, hence access via csv file. We should
        be able to log multiple AUPE instances, and load the appropriate
        one for the given dataset.

        In future, allow to import
        full transmission spectra, like sptk.

        :param filepath: file holding aupe information
        :type filepath: Path
        """
        # read the filepath csv file into the object
        self.cwls = None
        self.fwhms = None
        self.load_aupe_info(filepath)

        # cam number -> camera does not typically change between AUPE versions.
        self.cam_dict = {
                2: 'HRC',
                0: 'LWAC',
                1: 'RWAC'}
        
        # self.load_flat_fields() # TODO
        # self.load_bias_frames() # TODO
    
    def load_aupe_info(self, filepath):
        """Load the AUPE information from the csv file
        """
        # read the filepath csv file into the object
        aupe_info = pd.read_csv(filepath, index_col=0)
        self.cwls = aupe_info['cwl'].to_dict()
        self.fwhms = aupe_info['fwhm'].to_dict()
        # TODO read version number/date/project from info file

class CalibrationTarget:
    def __init__(self):
        self.patch_names = []
        self.patch_reflectance = None  # np.ndarray
        self.patch_srgb = None  # np.ndarray
        self.patch_rois = None  # np.ndarray
        self.observed_values = {}  # dict of metrics per patch

    def load_data(self):
        filepath = Path('..', 'data', 'colorchecker_srgb_d50.csv')
        # read the csv file into a pandas dataframe
        cal_targ_df = pd.read_csv(filepath, index_col=0)

    def load_rois(self):
        pass

    def draw_rois(self, image):
        pass

    def measure_patches(self):
        pass

    def save_rois(self):
        pass

class Img:
    def __init__(self, 
                 file_dict: Dict, 
                 aupe_info: AupeInfo):
        # from filepath
        self.filepath = file_dict['filepath']
        self.filename = self.filepath.name

        self.scene = file_dict['scene']
        self.sol = file_dict['sol']
        self.trial = file_dict['trial']
        
        self.out_dir = file_dict['out_dir']

        self.channel = self.filename.split('_')[3]
        # from metadata
        self.camera = None
        self.pan = None
        self.tilt = None
        self.exposure = None
        self.capture_timestamp = None
        self.stretch = {
            'raw': {
                'factor': 1.0/((2**BIT_DEPTH) - 1),
                'roi': [None, None, None, None]
            },
            'bps': {
                'factor': None,
                'roi': [None, None, None, None]
            },
            'wps': {
                'factor': None,
                'roi': [None, None, None, None]
            },
            '99p': {
                'factor': None,
                'roi': [None, None, None, None]
            }
        }
        self.units = None
        # from image
        self.width = None
        self.height = None
        self.dtype = None
        self.image = None  # np.ndarray        
        cam_num = self.load_image()
        self.camera = aupe_info.cam_dict[cam_num]
        # set cwl and fwhm
        self.cwl = aupe_info.cwls[self.channel]
        self.fwhm = aupe_info.fwhms[self.channel]

    def load_image(self):
        """Load the image and metadata from the aupe image file exif data
        """
        # read the metadata from the image file using the PIL exif reader
        img = PIL.Image.open(self.filepath)
        metadata = img.info
        # set the attributes from the metadata
        self.timestamp = metadata['AU_timestampUTC']        
        self.exposure = float(metadata['AU_exposureTime'])
        self.pan = float(metadata['AU_pan'])
        self.tilt = float(metadata['AU_tilt'])
        # self.filternum = int(metadata['AU_filterNum']) # ignore this use filename
        # read the image data
        self.image = np.array(img)
        # derive metadata from the image data
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.dtype = self.image.dtype
        self.units = 'DN'
        return int(metadata['AU_camNum'])

    def extract_stretch_coefficient(self, 
                method: Literal['raw', 'bps', 'wps', '99p']='raw',
                wp_roi: Tuple[int, int, int, int]=None) -> float:
        """Get the stretch coefficient for the given image according to
        the selected method.
        'bps': "brightest pixel stretch" - stretches such that the brightest pixel
        has a value of 1.0.
        'wps': "white patch stretch" - stretches such that the white patch of the
        MacBeth colorchecker has a value of 1.0.

        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99p']
        """        
        if method == 'bps':
            # get the brightest pixel value and location
            bp_val = np.nanmax(self.image)            
            bp_loc = np.unravel_index(np.argmax(self.image, axis=None), self.image.shape)
            # set the stretch coefficient to 1.0 / bp_val
            self.stretch['bps'] = {}
            self.stretch['bps']['factor'] = 1.0 / bp_val
            self.stretch['bps']['roi'] = (bp_loc[0], bp_loc[1], 1, 1)
        elif method == '99p':
            # get the brightest pixel value and location
            pct_val = np.percentile(self.image, 99)            
            # set the stretch coefficient to 1.0 / bp_val
            self.stretch['99p'] = {}
            self.stretch['99p']['factor'] = 1.0 / pct_val
            # ROI for percentile stretch is not valid.
        elif method == 'wps':
            if wp_roi is None:
                # zoom in on the colorchecker target
                print('First draw an ROI around the calibration target')
                title = 'Select Calibration Target ROI'
                ct_roi = cv2.selectROI(title, self.image)
                # switch order of roi to (y, x, h, w)
                ct_roi = (ct_roi[1], ct_roi[0], ct_roi[3], ct_roi[2])           
                cv2.destroyWindow(title)             
                # get the white patch value
                print('Now draw an ROI around the white patch on the calibration target')
                title = 'Select White Patch ROI'
                ct_img = self.image[ct_roi[0]:ct_roi[0]+ct_roi[2], ct_roi[1]:ct_roi[1]+ct_roi[3]]
                wp_roi = cv2.selectROI(title, ct_img)
                # switch order of roi to (y, x, h, w)
                wp_roi = (wp_roi[1], wp_roi[0], wp_roi[3], wp_roi[2])           
                cv2.destroyWindow(title) 
                wp_val = np.mean(ct_img[wp_roi[0]:wp_roi[0]+wp_roi[2], wp_roi[1]:wp_roi[1]+wp_roi[3]])
                # set the stretch coefficient to 1.0 / max_val
                print(f"White patch value: {wp_val} {self.units}")
                # put the wp_roi back into the original image
                wp_loc = (ct_roi[0] + wp_roi[0], ct_roi[1] + wp_roi[1], wp_roi[2], wp_roi[3])   
            else:
                # get the white patch value
                wp_val = np.mean(self.image[wp_roi[0]:wp_roi[0]+wp_roi[2], wp_roi[1]:wp_roi[1]+wp_roi[3]])
                # set the stretch coefficient to 1.0 / max_val
                # put the wp_roi back into the original image
                wp_loc = (wp_roi[0], wp_roi[1], wp_roi[2], wp_roi[3])
            self.stretch['wps'] = {}         
            self.stretch['wps']['factor'] = 1.0 / wp_val
            self.stretch['wps']['roi'] = wp_loc
        elif method == 'raw':
            print('No stretch method chosen')
        else:
            raise ValueError(f"Unknown stretch method: {method}")
    
        return self.stretch[method]['factor']

    def reset_stretch_coefficient(self,
                method: Literal['all', 'raw', 'bps', 'wps', '99p']='all'):
        """Reset the stretch coefficient for the given image according to
        the selected method.
        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99p']
        """
        if method == 'all':             
            self.stretch = {
                'raw': {
                    'factor': 1.0,
                    'roi': [None, None, None, None]
                },
                'bps': {
                    'factor': None,
                    'roi': [None, None, None, None]
                },
                'wps': {
                    'factor': None,
                    'roi': [None, None, None, None]
                },
                '99p': {
                    'factor': None,
                    'roi': [None, None, None, None]
                }
            }
        elif method == 'raw':
            self.stretch['raw'] = {
                'factor': 1.0/((2**BIT_DEPTH) - 1),
                'roi': [None, None, None, None]
            }
        elif method == 'bps':
            self.stretch['bps'] = {
                'factor': None,
                'roi': [None, None, None, None]
            }
        elif method == 'wps':
            self.stretch['wps'] = {
                'factor': None,
                'roi': [None, None, None, None]
            }
        elif method == '99p':
            self.stretch['99p'] = {
                'factor': None,
                'roi': [None, None, None, None]
            }
        else:
            raise ValueError(f"Unknown stretch method: {method}")
        

    def exposure_correction(self):
        """Correct for the exposure of the image, by converting
        to units of DN/s
        """        
        self.image = np.divide(self.image, self.exposure)
        self.units = 'DN/s'
        self.dtype = self.image.dtype
        # update strretch coefficients
        self.stretch['raw']['factor'] = self.stretch['raw']['factor'] * self.exposure

        if self.stretch['bps']['factor'] is not None:
            self.stretch['bps']['factor'] = self.stretch['bps']['factor'] * self.exposure
        if self.stretch['wps']['factor'] is not None:
            self.stretch['wps']['factor'] = self.stretch['wps']['factor'] * self.exposure
        if self.stretch['99p']['factor'] is not None:
            self.stretch['99p']['factor'] = self.stretch['99p']['factor'] * self.exposure

 
    def flat_field(self):
        pass

    def bias_correction(self):
        pass

    def show_image(self, stretch_method: Literal['raw', 'bps', 'wps', '99p']='raw'):
        """Display the image using matplotlib,
        and optionally show the histogram of the image data.
        """  

        title = f"{self.sol} {self.scene} {self.trial} {self.channel} {self.cwl}±{int(self.fwhm/2)} nm ({stretch_method})"
        
        # note the design here is that we always keep the original image in the 
        # self.image attribute, and just adjust the stretch ocefficient for 
        # display
        if self.stretch[stretch_method]['factor'] != 1.0:
            self.extract_stretch_coefficient(stretch_method)

        if stretch_method != 'raw':
            print(f"Stretching image using {LEVEL_DICT[stretch_method]}")
            disp_img = np.clip(self.image * self.stretch[stretch_method]['factor'], 0.0, 1.0)        
        else:
            disp_img = self.image / 255

        fig, ax = plt.subplots(1,2, figsize=(8, 4))
        disp = ax[0].imshow(disp_img, vmin=0.0, vmax=1.0, cmap='viridis')
        # add colorbar
        plt.colorbar(disp,fraction=0.046, pad=0.10, orientation='horizontal')
        
        ax[1].hist(disp_img.ravel()/self.stretch[stretch_method]['factor'], bins=256, color='gray', alpha=0.5)
        
        ax[1].set_xlabel(f"Pixel Value {self.units} ({stretch_method})")
        ax[1].tick_params(labelleft=False, left=False)
        # add title and labels
        fig.suptitle(title)
        plt.show()  

        # show the image at full resolution
        plt.imshow(disp_img, interpolation='none')
        plt.axis('off')
        # set the title
        plt.title(title)

    def export_image(self, stretch_method: Literal['raw', 'bps', 'wps', '99p']='bps'):
        """Export the image to a file, using the stretch method

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        """   
        
        title = f"{self.sol}_{self.scene}_{self.trial}_{self.channel}_{self.cwl}_{int(self.fwhm)}_nm_{stretch_method}.png"

        if stretch_method == 'raw':
            if self.units == 'DN/s':
                # convert to uint8
                disp_img = (self.image * self.exposure).astype(np.uint8)
            else:
                disp_img = (self.image).astype(np.uint8)
        else:
            print(f"Exporting image using {LEVEL_DICT[stretch_method]}")
            if self.stretch[stretch_method]['factor'] is None:
                self.extract_stretch_coefficient(stretch_method)
            disp_img = np.clip(self.image * self.stretch[stretch_method]['factor'], 0.0, 1.0)        
            # convert to uint8
            disp_img = (disp_img * 255).astype(np.uint8)

        # TODO - format metadata
        metadata = {
            'AU_sol': self.sol,
            'AU_scene': self.scene,
            'AU_trial': self.trial,
            'AU_camera': self.camera,
            'AU_channel': self.channel,
            'AU_cwl': str(self.cwl),
            'AU_fwhm': str(self.fwhm),
            'AU_pan': str(self.pan),
            'AU_tilt': str(self.tilt),
            'AU_exposureTime': str(self.exposure),
            'AU_timestampUTC': self.timestamp,
            'AU_stretch_method': stretch_method            
        }
        # use opencv to write the image to file
        # check and make a single frame output directory
        out_dir = self.out_dir / 'single_frame'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / title
        plt.imsave(
            str(out_file.absolute()), 
            disp_img, 
            vmin=0, vmax=255, cmap='gray', 
            format='png', 
            metadata=metadata)
    
        return out_file

class RGB:
    def __init__(self,
                 rgb_path_dict: Tuple[Dict, Dict, Dict],
                 aupe_info: AupeInfo):
        self.red = Img(rgb_path_dict[2], aupe_info)
        self.green = Img(rgb_path_dict[1], aupe_info)
        self.blue = Img(rgb_path_dict[0], aupe_info)
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)
        self.ccm = np.empty((3,3))
        self.balance_vector = {
            'raw': np.ones(3),
            'bps': np.zeros(3),
            'wps': np.zeros(3),
            '99p': np.zeros(3)
        }        
        self.camera = self.red.camera
        self.trial = self.red.trial
        self.scene = self.red.scene
        self.sol = self.red.sol
        self.pan = self.red.pan
        self.tilt = self.red.tilt
        self.timestamp = self.red.timestamp
        self.units = self.red.units
        self.dtype = self.red.dtype
        self.out_dir = self.red.out_dir
        self.stretch = ''
        self.calibration_target = None

    def display_histogram(self):
        pass

    def display_channels(self):
        pass

    def display_channel_histograms(self):
        pass

    def exposure_correct(self):
        """Exposure correct each channel
        """        
        self.red.exposure_correction()
        self.green.exposure_correction()
        self.blue.exposure_correction()
        # update the rgb image
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)
        # update the balance vector
        for method in self.balance_vector.keys():
            if method != 'raw':
                self.balance_vector[method] = self.balance_vector[method] * self.red.exposure

    def flat_field(self):
        pass

    def bias_subtract(self):
        pass

    def apply_ccm(self):
        self.stretch = 'ctc'

    def apply_balance_vector(self, method: Literal['raw', 'bps', 'wps', '99p']='raw'):
        if method != 'raw':
            print(f"Stretching image using {LEVEL_DICT[method]}")
            if (self.balance_vector[method] == np.zeros(3)).all():
                self.extract_balance_vector(method)
            r_disp_img = np.clip(self.red.image * self.balance_vector[method][0], 0.0, 1.0)        
            g_disp_img = np.clip(self.green.image * self.balance_vector[method][1], 0.0, 1.0)
            b_disp_img = np.clip(self.blue.image * self.balance_vector[method][2], 0.0, 1.0)
        else:
            r_disp_img = self.red.image
            g_disp_img = self.green.image
            b_disp_img = self.blue.image
                
        stretch_img = np.stack([r_disp_img, g_disp_img, b_disp_img], axis=2)
        return stretch_img

    def load_calibration_target(self):
        """Load in the calibration target data, including ROIs, if these have been saved.
        """

    def draw_calibration_target(self, cal_targ: CalibrationTarget):
        """Draw the calibration target on the image
        """
        # draw the calibration target on the image
        # TODO - add a method to save the rois
        disp_img = self.apply_balance_vector('99p')

        # patches

        # for patch in patches:


    def extract_ccm(self):
        pass

    def extract_balance_vector(self, method: Literal['raw', 'bps', 'wps', '99p']='raw'):
        """Extract the stretch coefficient for each channel of the RGB image.

        :param method: stretch method, defaults to 'raw'
        :type method: Literal['raw', 'bps', 'wps', '99p']
        """        

        if method == 'wps':
            # draw the roi on the colour image stack
            # zoom in on the colorchecker target
            title = 'Select Calibration Target Approx. ROI'
            print('First draw an ROI around the calibration target')
            ct_roi = cv2.selectROI(title, cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR))
            # switch order of roi to (y, x, h, w)
            ct_roi = (ct_roi[1], ct_roi[0], ct_roi[3], ct_roi[2])           
            cv2.destroyWindow(title)             
            # get the white patch value
            title = 'Select White Patch ROI'
            print('Now draw an ROI around the white patch on the calibration target')
            ct_img = self.rgb_image[ct_roi[0]:ct_roi[0]+ct_roi[2], ct_roi[1]:ct_roi[1]+ct_roi[3]]
            wp_roi = cv2.selectROI(title, cv2.cvtColor(ct_img, cv2.COLOR_RGB2BGR))
            # switch order of roi to (y, x, h, w)
            wp_roi = (wp_roi[1], wp_roi[0], wp_roi[3], wp_roi[2])           
            cv2.destroyWindow(title)
            # add the ct roi to the wp_roi
            wp_roi = (ct_roi[0] + wp_roi[0], ct_roi[1] + wp_roi[1], wp_roi[2], wp_roi[3])
            # get the white patch value
            # red stretch
            r_stretch = self.red.extract_stretch_coefficient('wps', wp_roi)        
            # green stretch
            g_stretch = self.green.extract_stretch_coefficient('wps', wp_roi)
            # blue stretch
            b_stretch = self.blue.extract_stretch_coefficient('wps', wp_roi)  
            # apply macbeth colourchecker white patch weightings
            r_stretch = r_stretch * WP_RED / 255
            g_stretch = g_stretch * WP_GREEN / 255
            b_stretch = b_stretch * WP_BLUE / 255          
        else:
            # red stretch
            r_stretch = self.red.extract_stretch_coefficient(method)        
            # green stretch
            g_stretch = self.green.extract_stretch_coefficient(method)
            # blue stretch
            b_stretch = self.blue.extract_stretch_coefficient(method)

        self.balance_vector[method] = np.array([r_stretch, g_stretch, b_stretch])

    def reset_balance_vector(self, 
                    method: Literal['all', 'raw', 'bps', 'wps', '99p']='all'):
        """Reset the balance vector to all ones
        """
        if method == 'all':
            self.balance_vector = {
                'raw': np.ones(3),
                'bps': np.zeros(3),
                'wps': np.zeros(3),
                '99p': np.zeros(3)
            }
        elif method == 'raw':
            self.balance_vector[method] = np.ones(3)
        else:
            self.balance_vector[method] = np.zeros(3)            

    def show_image(self, 
                   colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm']='raw'):
        """Display the RGB image using matplotlib,
        and optionally show the histogram of the image data.
        """
        title = f"{self.sol} {self.scene} {self.trial} {self.camera} RGB ({LEVEL_DICT[colour_correction]})"

        # apply balance vector or colour correction matrix
        if colour_correction != 'ccm':
            # apply balance vector
            disp_img = self.apply_balance_vector(colour_correction)
        else:
            # apply colour correction matrix
            pass        
        
        fig, ax = plt.subplots(1,2, figsize=(8, 4))  
        disp = ax[0].imshow(disp_img, vmin=0.0, vmax=1.0)

        # for each channel, show a histogram - retain original image units
        ax[1].hist(disp_img[:,:,0].ravel()/self.balance_vector[colour_correction][0], bins=256, color='red', alpha=0.5)
        ax[1].hist(disp_img[:,:,1].ravel()/self.balance_vector[colour_correction][1], bins=256, color='green', alpha=0.5)
        ax[1].hist(disp_img[:,:,2].ravel()/self.balance_vector[colour_correction][2], bins=256, color='blue', alpha=0.5)
        ax[1].tick_params(labelleft=False, left=False)
        # label x axis with units
        ax[1].set_xlabel(f"Pixel Value {self.units} ({colour_correction})")
        # add title and labels
        fig.suptitle(title)
        plt.show()

        # show the image at full resolution
        plt.imshow(disp_img, interpolation='none')
        plt.axis('off')
        # set the title
        plt.title(title)

    def export_image(self, colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm']='bps'):
        """Export the image to a file, using the stretch method

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        """   
        
        title = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_RGB_{colour_correction}.png"

        if colour_correction == 'raw':
            if self.units == 'DN/s':
                # convert to uint8
                disp_img = (self.rgb_image * self.red.exposure).astype(np.uint8) # note this doesn't actually work...
            else:
                disp_img = (self.rgb_image).astype(np.uint8)
        else:
            print(f"Exporting image using {LEVEL_DICT[colour_correction]}")
            if (self.balance_vector[colour_correction] == np.zeros(3)).all():
                self.extract_balance_vector(colour_correction)
            disp_img = np.clip(self.apply_balance_vector(colour_correction), 0.0, 1.0)        
            # convert to uint8
            disp_img = (disp_img * 255).astype(np.uint8)

        # TODO - format metadata
        metadata = {
            'AU_sol': self.sol,
            'AU_scene': self.scene,
            'AU_trial': self.trial,
            'AU_camera': self.camera,
            'AU_pan': str(self.pan),
            'AU_tilt': str(self.tilt),
            'AU_timestampUTC': self.timestamp,
            'AU_colour_correction': colour_correction            
        }
        # use opencv to write the image to file
        # check and make a single frame output directory
        out_file = self.out_dir / title
        plt.imsave(
            str(out_file.absolute()), 
            disp_img, 
            vmin=0, vmax=255, cmap='gray', 
            format='png', 
            metadata=metadata)
    
        return out_file


class HRC(RGB):
    """HRC class inherits the methods of the RGB class, but handles HRC by
    loading the same un-debayered image into each channel.
    Then, during image debayering, it rewrites the red, green and blue image
    data. The allocation of the same HRC image to each channel
    is handled by the AupeIO class.
    """    
    def __init__(self, rgb_path_dict: Tuple[Dict, Dict, Dict],
                 aupe_info: AupeInfo):
        super().__init__(rgb_path_dict, aupe_info)
        self.debayered = False

    def display_raw(self):
        pass

    def debayer(self):
        # debayer the image
        raw_img = self.red.image
        col_img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)
        self.red.image = col_img[:,:,2]
        self.green.image = col_img[:,:,1]
        self.blue.image = col_img[:,:,0]
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)
        self.reset_balance_vector('all')
        self.debayered = True

class WAC_MS:
    def __init__(self):
        self.band_images = {}
        self.band_channels = []
        self.band_exposures = []
        self.band_stack = None  # np.ndarray (w x h x n)
        self.false_rgb = RGB()
        self.refl_coeffs = None  # np.ndarray (n,)
        self.refl_offset = None  # np.ndarray (n,)
        self.camera = None
        self.scene = None
        self.sol = None
        self.pan = None
        self.tilt = None
        self.timestamp = None
        self.units = None
        self.stretch = None
        self.calibration_target = None

    def set_false_color(self, bands, stretches):
        pass

    def display_false_color(self):
        pass

    def export_false_color(self):
        pass

    def load_calibration_target(self):
        pass

    def exposure_correct(self):
        pass

    def flat_field(self):
        pass

    def bias_subtract(self):
        pass

    def extract_refl_coeffs(self):
        pass

    def apply_refl_correction(self):
        pass

    def coeffs_to_illuminant_spd(self):
        pass
