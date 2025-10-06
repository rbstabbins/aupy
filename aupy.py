# A Python library for processing and analysing images from the
# Aberystwyth University PanCam Emulator, AUPE.
#
# Roger Stabbins
# Natural History Museum, London
# 9/5/2025

from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union, Optional
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from roipoly import RoiPoly
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from spectral import envi
from PIL import Image
from PIL import ImageDraw, ImageFont

import colour
from colour.characterisation import CCS_COLOURCHECKERS
from colour_checker_detection import detect_colour_checkers_inference as detect_target
from colour_checker_detection.detection.common import sample_colour_checker, as_int32_array

STRETCH_DICT = {
    'raw': 'raw image no stretch',
    'bps': 'brightest pixel stretch', # for individual channels/bands/frames
    'bpb': 'brightest pixel balanced', # each channel/band/frame is bps stretched independently
    'bpu': 'brightest pixel unbalanced', # channel proportions are maintained whilst stretching brightest pixel across all channels
    '99s': '99th percentile stretch', # for individual channels/bands/frames
    '99b': '99th percentile balanced', # each channel/band/frame is 99pct stretched independently
    '99u': '99th percentile unbalanced', # channel proportions are maintained whilst stretching 99pct across all channels
    'wps': 'white patch stretch', # for individual channels/bands/frames
    'wpb': 'white patch balanced', # each channel/band/frame is wps stretched independently
    'wpu': 'white patch unbalanced', # channel proportions are maintained whilst stretching wps across all channels
    'ccm': 'colour correction matrix', # each output R,G,B channel is a weighted mixture of the input R + G + B channels, encoded in a 3x3 matrix
    'ctr': 'calibration target reflectance'
}

# D50 illuminant (direct sunlight 5000K + skylight) RGB values of white patch
# WP_RED = 245
# WP_GREEN = 245
# WP_BLUE = 243

# C illuminant (shade) RGB values of white patch
WP_RED = 243
WP_GREEN = 244
WP_BLUE = 243

BIT_DEPTH = 8

def gamma_curve(x: NDArray, gamma: float) -> NDArray:
    """Gamma curve function, to use for Gamma fitting routine.

    :param x: abscissa values
    :type x: NDArray
    :param a: Initial estimate of amplitude
    :type a: float
    :param gamma: initial estimate of gamma
    :type gamma: float
    :return: ordinate values of the Gamma curve function
    :rtype: NDArray
    """
    return (x ** gamma)

def gauss(x: NDArray, a: float, x0: float, sigma: float) -> NDArray:
    """Gaussian function, to use for Gauss fitting routine.

    :param x: abscissa values
    :type x: NDArray
    :param a: Initial estimate of amplitude
    :type a: float
    :param x0: initial estimate for centre (mean)
    :type x0: float
    :param sigma: initial estimate of width (standard deviation)
    :type sigma: float
    :return: ordinate values of the Gaussian function
    :rtype: NDArray
    """        
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

class AupeInfo:
    """A class to hold the AUPE information for a given dataset, that is not
    included in the image metadata. This includes the filter positions, 
    filter ids, cwl and fwhm.

    The purpose of this is to handle changes of these values between different 
    versions of AUPE and previous datasets, hence access via csv file. 
    
    We should be able to log multiple AUPE instances, and load the appropriate
    one for the given dataset.
    """    
    def __init__(self, filepath: Path):
        """
        :param filepath: file holding aupe information
        :type filepath: Path
        """
        # read header lines of version and date
        header = pd.read_csv(filepath, nrows=2, usecols=[0,1], index_col=0)
        self.aupe_info_version = header.loc['version'].values[0]
        self.aupe_info_date = header.loc['date'].values[0]
        # read the data
        self.filepath = filepath
        self.data_dir = filepath.parent
        aupe_info = pd.read_csv(filepath, index_col=0, header=3)
        self.filter_pos = aupe_info.index.to_list()
        self.filter_id = aupe_info['filter_id'].to_dict()
        self.cwl = aupe_info['cwl'].to_dict()
        self.fwhm = aupe_info['fwhm'].to_dict()

        # cam number -> camera (does not typically change between AUPE versions)
        self.cam_dict = {
                2: 'HRC',
                0: 'LWAC',
                1: 'RWAC'}
        
        # self.load_flat_fields() # TODO
        # self.load_bias_frames() # TODO
        
    def inverse_filter_id(self) -> Dict[str, str]:
        """Invert the filter id dictionary to get the filter id from the filter
        position.

        :return: Inverted filter id dictionary
        :rtype: Dict[str, str]
        """
        inv_filter_id = {v: k for k, v in self.filter_id.items()}
        return inv_filter_id
    
    def inverse_cwl(self) -> Dict[str, int]:
        """Invert the cwl dictionary to get the cwl from the filter position.

        :return: Inverted cwl dictionary
        :rtype: Dict[str, str]
        """
        inv_cwl = {v: k for k, v in self.cwl.items()}
        return inv_cwl
    
    def inverse_fwhm(self) -> Dict[str, int]:
        """Invert the fwhm dictionary to get the fwhm from the filter position.

        :return: Inverted fwhm dictionary
        :rtype: Dict[str, str]
        """
        inv_fwhm = {v: k for k, v in self.fwhm.items()}
        return inv_fwhm
    
    def filter_ids2pos(self, 
                    filter_ids: List[str]) -> List[str]:
        """Convert a list of filter ids to a list of filter positions

        :param filter_ids: List of filter ids to convert
        :type filter_ids: List[str]
        :return: List of filter positions corresponding to the filter ids
        :rtype: List[str]
        """
        filter_pos_lut = self.inverse_filter_id()
        filter_pos = [filter_pos_lut[filter_id] for filter_id in filter_ids]
        return filter_pos

    def set_filter_ids(self, 
                    camera: Literal['HRC', 'LWAC', 'RWAC', 'LRWAC'],
                    frame_type: Literal['RGB', 'MSC']) -> List[str]:
        """Set the filter ids to load for the given camera and frame type.

        :param camera: Camera to load the image from
        :type camera: Literal['HRC', 'LWAC', 'RWAC', 'LRWAC']
        :param frame_type: Frame type to load the image into
        :type frame_type: Literal['RGB', 'MSC']
        :return: List of filter ids to load for the given camera and frame type
        :rtype: List[str]
        """        
        filter_ids = []

        if camera == 'HRC': # initialise with raw HRC frame - ID HR0
            if frame_type == 'RGB':
                filter_ids = ['HR0', 'HR0', 'HR0']
            elif frame_type == 'Single':
                filter_ids = ['HR0']
            elif frame_type == 'MSC':
                filter_ids = ['HR0', 'HR0', 'HR0']
            else:
                raise ValueError(f"Unknown frame type {frame_type} for HRC")
            
        elif camera == 'LWAC':
            if frame_type == 'RGB':
                filter_ids = ['L1R', 'L2G', 'L3B']
            elif frame_type == 'MSC':
                filter_ids = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']

        elif camera == 'WAC':
            if frame_type == 'RGB':
                filter_ids = ['R1R', 'R2G', 'R3B']
            elif frame_type == 'MSC':
                filter_ids = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12']

        # 'virtual' camera - holds WAC->warped-2-HRC images
        elif camera == 'WACHRC':
            if frame_type == 'MSC':
                filter_ids = [
                            'H1R', 'H2G', 'H3B',
                            'G01', 'G02', 'G03', 'G04', 'G05', 'G06',
                              'G07', 'G08', 'G09', 'G10', 'G11', 'G12']
                
        else:
            raise ValueError(f"Unknown camera {camera}")
            # TODO - add support for NavCams
        
        return filter_ids
    
class AupeIO:
    '''Class for preparing image filepaths for a given AUPE camera, frame type,
    sol, scene, and trial (optional). 
    
    Subset of filter ids can be specified.
    
    The function assumes that the images are stored in 'data' directory, 
    adjacent to the script (or notebook) running the programme. 
    
    It assumes that information describing the instance of aupe is stored in a 
    'data' subdirectory next to the aupy.py module.    
    '''
    def __init__(self, 
                 camera: Literal['HRC', 'WAC'],
                 frame_type: Literal['Single', 'RGB', 'MSC'],
                 sol: str,
                 scene: str, 
                 trial: str='',
                 filter_ids: Optional[List[str]]=None,
                 campaign_dir: Path=Path('..','data'),
                 aupe_info_path: Path=Path('.','data','aupe_info.csv')) -> None:
        """
        :param camera: Camera to load the image from
        :type camera: Literal['HRC', 'LWAC', 'RWAC']
        :param frame_type: Frame type to load the image into
        :type frame_type: Literal['Single', 'RGB', 'MSC']
        :param sol: Sol to load the image from
        :type sol: str
        :param scene: Scene to load the image from
        :type scene: str
        :param trial: Trial to load the image from
        :type trial: str
        :param filter_ids: List of filter ids to load the image from, 
            defaults to ['']
        :type filter_ids: List[str]
        :param campaign_dir: Directory holding the campaign data, 
            defaults to Path('..','data')
        :type campaign_dir: Path
        :param aupe_info_path: Path to the aupe_info.csv file, 
            defaults to Path('.','data','aupe_info.csv')
        :type aupe_info_path: Path
        """
        self.camera = camera
        self.frame_type = frame_type
        self.sol = sol
        self.scene = scene
        self.campaign_dir = campaign_dir

        # handle the case where trial is not in input directory
        if trial != '':
            self.scene_dir = Path(campaign_dir, sol, scene, trial)
        else:
            self.scene_dir = Path(campaign_dir, sol, scene)
            trial = 'Trial1'
        self.trial = trial

        if not self.campaign_dir.exists():
            raise FileNotFoundError(f"{self.campaign_dir} does not exist")
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"{self.scene_dir} does not exist")

        self.out_dir = Path(self.campaign_dir,
                                '..', 
                                'processed',
                                self.sol,
                                self.scene,
                                self.trial,
                                self.camera,
                                self.frame_type)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.aupe_info = AupeInfo(aupe_info_path)

        # set the list of filters to load for given camera and frame type
        if filter_ids is not None and filter_ids[0] != '':
            # check if the given filter ids are valid
            for filter_id in filter_ids:
                if filter_id not in self.aupe_info.filter_id.values():
                    raise ValueError(f"Filter {filter_id} not recognised")
        elif frame_type == 'Single':
            # if frame type is single, then a filter id is required
            raise ValueError("Filter id required for single frame type")
        else:
            # otherwise, get the filters for the given camera and frame type
            filter_ids = self.aupe_info.set_filter_ids(camera, frame_type)
        
        self.filter_ids = filter_ids #e.g. 'G01', 'G02', 'L1R', etc
        self.filter_pos = self.aupe_info.filter_ids2pos(filter_ids) #e.g.'LWAC1'
                
        # find the image filepaths to load into the frame
        self.input_files = []
        png_files = list(self.scene_dir.glob("*.png"))
        for filter_pos in self.filter_pos: # note order preserved
            # get the files that match the filter pos code
            files = [path for path in png_files if filter_pos+'_' in path.name]
            # add the files to the input files list
            self.input_files += files

    def load_frame(self) -> Union[None, 'Img', 'HRC', 'WAC_RGB', 'RGB', 'MSC']:
        """Load the frame from the input files, and return the frame object.

        :return: Frame object for the given camera and frame type
        :rtype: Union[Img, HRC, WAC_RGB, MSC]
        """        
        # if there are no files, skip
        if len(self.input_files) == 0:
            # raise a file not found error
            raise FileNotFoundError(
                        "No files found. Please check the sol/scene/trial.")

        input_file_dicts = self.file_dicts()
        if self.frame_type == 'Single':
            if len(input_file_dicts) > 1:
                raise ValueError(f"Multiple files found for single frame type: {input_file_dicts}")
            frame = Img(input_file_dicts[0], self.aupe_info)
            return frame
        
        elif self.frame_type == 'RGB':
            if self.camera == 'HRC':
                frame = HRC.from_filedicts((input_file_dicts[0],
                                            input_file_dicts[1],
                                            input_file_dicts[2]), 
                                            self.aupe_info)
                frame.debayer()
                return frame
            elif self.camera == 'WAC':
                frame = WAC_RGB.from_filedicts((input_file_dicts[0],
                                                input_file_dicts[1],
                                                input_file_dicts[2],), 
                                                self.aupe_info)
                return frame
        
        elif self.frame_type == 'MSC':
            if self.camera == 'WAC' or self.camera == 'WACHRC':
                # create a WAC MSC frame                
                frame = MSC.from_filedicts(input_file_dicts, self.aupe_info)
                return frame
            elif self.camera == 'HRC':
                frame = HRC.from_filedicts((input_file_dicts[0],
                                            input_file_dicts[1],
                                            input_file_dicts[2]), 
                                            self.aupe_info)
                frame = frame.debayer(to_msc=True)                
            return frame
        
        else:
            raise ValueError(f"Unknown frame type {self.frame_type}")
    
    def file_dicts(self) -> List[Dict[str, Union[str, Path]]]:
        """For the given filepath, return a list (to preserve order) of 
        dictionaries giving:
        - full file path
        - filter id
        - sol
        - scene
        - trial
        - output directory

        :return: File information needed to process the image
        :rtype: List[Dict[str, Union[str, Path]]]
        """        
        input_file_dicts = []
        for i, input_file in enumerate(self.input_files):
            file_dict = {}
            file_dict['filepath'] = input_file
            file_dict['filter_id'] = self.filter_ids[i]
            file_dict['sol'] = self.sol
            file_dict['scene'] = self.scene
            file_dict['trial'] = self.trial
            file_dict['out_dir'] = self.out_dir
            input_file_dicts.append(file_dict)
        
        return input_file_dicts

class CalibrationTarget:
    """Calibration Target Class for hosting reference and observed calibration
    target patch values, and the colour correction matrix, as well
    as methods for finding and analysing the calibration target within a frame.

    Can specify the illuminant and colour checker to use for the calibration
    target. The default is ICC D50 and ColorChecker24 - After November 2014.
    """    
    def __init__(self,
                illuminant: Literal[
                     'A', 'B', 'C', 
                     'D50', 'D55', 'D65', 'D75', 
                     'ICC D50']='ICC D50',
                colour_checker: Literal[
                    'ColorChecker 1976',
                    "ColorChecker 2005",
                    "BabelColor Average",
                    "ColorChecker24 - Before November 2014",
                    "ColorChecker24 - After November 2014",
                    "ColorCheckerSG - Before November 2014",
                    "ColorCheckerSG - After November 2014",
                    "TE226 V2"]='ColorChecker24 - After November 2014',
                bad_patches: Optional[Dict[str, List[str]]]={},
                ) -> None:
        """Initialise the calibration target class

        :param illuminant: Illuminant to use for the calibration target,
            defaults to 'ICC D50'
        :type illuminant: Literal[
            'A', 'B', 'C',
            'D50', 'D55', 'D65', 'D75',
            'ICC D50']
        :param colour_checker: Colour checker to use for the calibration target,
            defaults to 'ColorChecker24 - After November 2014'
        :type colour_checker: Literal[
            'ColorChecker24 - After November 2014',
            'ColorChecker 1976',
            "ColorChecker 2005",
            "BabelColor Average",
            "ColorChecker24 - Before November 2014",
            "ColorChecker24 - After November 2014",
            "ColorCheckerSG - Before November 2014",
            "ColorCheckerSG - After November 2014",
            "TE226 V2"]
        """
        # reference values
        self.illuminant = illuminant
        self.colour_checker = colour_checker
        self.patch_ref_xyY = self.load_patch_colours('xyY')
        self.patch_ref_XYZ = self.load_patch_colours('XYZ')
        self.patch_ref_sRGB = self.load_patch_colours('sRGB')
        self.patch_names = list(CCS_COLOURCHECKERS[colour_checker].data.keys())
        self.rows = CCS_COLOURCHECKERS[colour_checker].rows
        self.cols = CCS_COLOURCHECKERS[colour_checker].columns
        self.patch_ref_refl = self.load_patch_spectra()
        # observed values
        self.target_outline = np.zeros((4,2))
        self.ccm = np.zeros((3,3)) # Colour Correction Matrix
        self.gamma = 1.0
        self.patch_obs_drgb = None # do we need this?
        self.patch_obs_srgb = None  # do we need this?
        self.patch_obs_refl = None  # np.ndarray
        self.bad_patches = bad_patches
        
    def load_patch_colours(self, 
                space: Literal['xyY', 'XYZ', 'sRGB']
                ) -> NDArray:
        """Load reference colours for the calibration target patches
        via the colour science python library.
        
        :param space: Colour space to use for the reference values
        :type space: Literal['xyY', 'XYZ', 'sRGB']
        :return: Reference colours for the calibration target patches in the
        specified colour space.
        :rtype: NDArray
        """
        # load reference values from the colour science library
        ref_ct = CCS_COLOURCHECKERS[self.colour_checker]

        # get xyY values
        ref_ct_xyY = np.array(list(ref_ct.data.values()))
        if space == 'xyY':
            return ref_ct_xyY
        
        # get XYZ values
        ref_ct_XYZ = colour.xyY_to_XYZ(ref_ct_xyY)
        if space == 'XYZ':
            return ref_ct_XYZ

        # update the illuminant
        illuminant_ccs = colour.CCS_ILLUMINANTS[
            "CIE 1931 2 Degree Standard Observer"][self.illuminant]

        # get sRGB values (without Gamma encoding)
        ref_ct_RGB = colour.XYZ_to_sRGB(ref_ct_XYZ, illuminant_ccs, 
                                                apply_cctf_encoding=False)
        if space == 'sRGB':
            return ref_ct_RGB
        
        else:
            raise ValueError(f"Unknown space {space} for colourchecker data")

    def load_patch_spectra(self, show: bool=False)-> Dict[str, NDArray]:
        """Load the high resolution spectral reflectance data for the 
        calibration target patches.

        Note this assumes that the patch columns match the order of the 
        colour checker patches in the colour science library.

        Note that the reflectance values are given in percentage (0 - 100), so 
        we convert them to reflectance values (0 - 1) by dividing by 100.

        :return: Dictionary containing the wavelengths and reflectance values
        :rtype: Dict[str, NDArray]
        """        
        filepath = Path('.', 'data', 'colorchecker_spectra.csv')
        # read the csv file into a pandas dataframe
        data = pd.read_csv(filepath, index_col=0, header=1)
        # convert the dataframe to a dictionary
        patch_ref_refl = {}
        patch_ref_refl['wavelengths'] = data.index.to_numpy()
        # convert to reflectance values, and clamp to 0-1 range
        patch_ref_refl['reflectance'] = np.clip(data.to_numpy() / 100, 0,1)

        # if show, plot the reflectance spectra
        if show:
            plt.figure(figsize=(10, 5))
            cols = np.clip(self.patch_ref_sRGB, 0,1)
            for i, patch in enumerate(self.patch_names):
                plt.plot(patch_ref_refl['wavelengths'], 
                         patch_ref_refl['reflectance'][:, i],
                         color=cols[i], 
                         label=patch)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title('Calibration Target Patch Reflectance Spectra')
            plt.grid()
            plt.show()
    
        return patch_ref_refl

    def sample_patch_spectra(self, frame: 'MSC') -> NDArray:
        """Sample the reference patch reflectance spectra, given by the 
        reference csv file, with the transmission profiles of the bands of the 
        given frame (must be an MSC frame).
        
        :param frame: The frame to get the transmission profiles from to 
            sample the reference patch reflectance values with.
        :type frame: MSC
        :return: Patch reflectance values for each band of the frame
        :rtype: NDArray
        """
        if not isinstance(frame, MSC):
            raise ValueError(f"Frame must be an MSC frame, got {type(frame)}")

        # interpolate the reference patch reflectance spectra to the frame band
        # transmission wavelengths
        patch_refl_interp = interp1d(
                                    self.patch_ref_refl['wavelengths'],
                                    self.patch_ref_refl['reflectance'],
                                    axis=0,
                                    bounds_error=False,
                                    fill_value='extrapolate')
        patch_refl = patch_refl_interp(frame.response_wavelengths)

        # sample the reference patch reflectance values for each band of the
        # frame
        # i.e. R[cwl] = sum_wvl(R[wvl] * T_cwl[wvl]) / sum_lambda(T_cwl[wvl])
        patch_refl_vals = np.divide(
                        np.matmul(frame.response_functions.T, patch_refl).T,
                                    np.sum(frame.response_functions, axis=0))

        return patch_refl_vals

    def find_target_outline(self, image, show: bool=False) -> bool:
        """Automatically find the Calibration Target 
        using the colour checker inference detection algorithm

        :param image: The image containing the calibration target
        :type image: NDArray
        :return: True if the target was found, False otherwise
        :rtype: bool
        """        
        print("Searching for colour checker...")
        result = detect_target(image, additional_data=True)
        
        # check if the run was successful
        if result == ():
            print("No colour checker found")
            print("Searching for colour checker in cropped image...")
            # if the first search fails, we try to find it again in a sub-frame
            crop = 150
            cropped_image = image[crop:-crop, crop:-crop]
            result = detect_target(cropped_image, additional_data=True)
            if result == ():
                # # if this also fails, we return false
                print("No calibration target found")
                return False
            else:
                print(f"Found {len(result)} colour checkers")
                # get the first one, as we only expect one
                colour_checker_data = result[0]
                # this gives the quadrilateral containing the calibration target
                # we need to offset the by the image reduction of crop pixels
                colour_checker_data.quadrilateral[:, 0] += crop
                colour_checker_data.quadrilateral[:, 1] += crop
                self.target_outline = colour_checker_data.quadrilateral
        else:   
            print(f"Found {len(result)} colour checkers")
            # get the first one, as we only expect one
            colour_checker_data = result[0]
            # this gives the quadrilateral containing the calibration target
            self.target_outline = colour_checker_data.quadrilateral

        if show:
            fig, ax = self.show_target_outline(image)

        return True

    def draw_target_outline(self, image: NDArray, show: bool=False) -> bool:
        """Draw the target outline manually on the RGB image using
        the roipoly library.

        :param image: The image containing the calibration target
        :type image: NDArray
        :return: True if the target was found, False otherwise
        :rtype: bool
        """   

        # Predraw the approximate area using OpenCV ROI select,
        # so that we can zoom in on the target before drawing the more precise
        # polyroi outline of the target.
        prompt = "Select Approx. Bounding box of calibration target"
        ct_box = cv2.selectROI(prompt, np.flip(image, 2))
        cv2.destroyWindow(prompt)
        # switch order of roi to (y, x, h, w)
        ct_box = (ct_box[1], ct_box[0], ct_box[3], ct_box[2])           
        # crop image to box
        ct_img = image[ct_box[0]:ct_box[0]+ct_box[2], 
                       ct_box[1]:ct_box[1]+ct_box[3]]

        if len(ct_img) == 0:
            print("No ROI selected")
            plt.close()
            return False
    
        default_backend = mpl.get_backend()
        mpl.use('QtAgg')  # need this backend for RoiPoly to work 
        fig = plt.figure(figsize=(10,10), dpi=80)

        plt.imshow(ct_img, origin='upper')
        plt.title(f'Mark precise corners of Colour Checker')

        my_roi = RoiPoly(fig=fig) # draw new ROI in red color
        plt.close()
        mpl.use(default_backend)  # reset backend

        # Get the coords for the ROIs
        # offset the coords by the ROI location
        quad_roi_x = [x + ct_box[1] for x in my_roi.x]
        quad_roi_y = [y + ct_box[0] for y in my_roi.y]
        points = np.array([quad_roi_x, quad_roi_y]).T[0:4]

        if len(points) != 4:
            print("Invalid number of points for calibration target outline.")
            return False
        else:            
            self.target_outline = points
            if show:
                fig, ax = self.show_target_outline(image)
            return True

    def show_target_outline(self, image: NDArray):
        """Show the target quadrilateral on the image

        :param image: The image containing the calibration target
        :type image: NDArray
        """
        annotated_image = image.copy()

        # check the format of the image, and convert uint8 if neccesary
        if annotated_image.dtype != np.uint8:
            annotated_image = (annotated_image * 255).astype(np.uint8)

        # Ensure points are in the correct shape and type for cv2.polylines        
        pts = np.array(self.target_outline, dtype=np.int32).reshape((-1, 1, 2))
        # draw the quadrilateral on the image
        annotated_image = cv2.polylines(annotated_image, 
                                        [pts], 
                                        isClosed=True, 
                                        color=(255, 0, 0), 
                                        thickness=2)
        # show the image
        # make figure
        plt.style.use('default')
        fig, ax = plt.subplots(1,1, figsize=(4, 4))
        ax.imshow(annotated_image)
        plt.show()
        # set the title
        ax.set_title('Calibration Target Outline')

        return fig, ax

    def show_target(self, image: NDArray) -> None:
        """Show the target warped to the target outline, as analysed
        by the colour checker patch value evaluation.

        :param image: The image containing the calibration target
        :type image: NDArray
        """
        if self.target_outline is None or self.target_outline.size == 0:
            raise ValueError("Calibration target outline not set. " \
                                "Please run find_calibration_target() first.")
        
        width, height, rectangle, samples = self.get_target_dimensions()

        patch_data = sample_colour_checker(
                            image, 
                            self.target_outline, 
                            rectangle, 
                            samples,
                            working_width=width,
                            working_height=height,
                            reference_values=None)
        
        # draw where the patches are on the sampled calibration target          
        colour.plotting.plot_image(
                colour.cctf_encoding(
                    np.clip(patch_data.colour_checker, 0, 1)))

    def get_target_dimensions(self) -> Tuple[int, int, NDArray, int]:
        """Get the target rectangle dimensions as a rectangle of the form
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], and the number of pixels
        sampled per patch.

        :return: Target Rectangle and number of pixels sampled per patch
        :rtype: Tuple[NDArray, int]
        """        
        # check the target has been drawn
        if self.target_outline is None or self.target_outline.size == 0:
            raise ValueError("Calibration target outline not set. " \
                                "Please run find_calibration_target() first.")
        
        # get the approximate width and height of the calibration target        
        q = self.target_outline
        width = np.abs(q[0][0] - q[3][0]).astype(np.int32)
        height = np.abs(q[0][1] - q[1][1]).astype(np.int32)

        # let the minimum width be 5x number of columns, and height be 5x number of rows
        width = max(width, 5 * self.cols)
        height = max(height, 5 * self.rows)

        samples = int(np.floor(np.sqrt(0.25*(width * height)//24)))

        # ensure samples is at least 3
        if samples < 3:
            samples = 3

        print(f"Target Width: {width} Height: {height}, Samples: {samples**2}")
        rectangle = as_int32_array([
                            [0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]])
        
        return width, height, rectangle, samples

    def get_bad_patches(self, 
                        frame: Union['Img', 'RGB', 'HRC', 'WAC_RGB', 'MSC']
                        ) -> None:
        """Find the bad patches in each channel of the given frame.
        A baad patch means an overexposed patch.
        Set bad_patches attribute to a boolean array of shape
        (n_patches, n_channels) where True indicates a bad patch.

        :param frame: The frame to get the bad patches for
        :type frame: Union[Img, HRC, WAC_RGB]
        """
        # check the frame outline has been set
        if self.target_outline is None or self.target_outline.size == 0:
            raise ValueError("Calibration target outline not set. " \
                                "Please run find_calibration_target() first.")
        
        bad_dn = 10 # set the max allowed patch average to X DN below the max value

        # if an image
        if isinstance(frame, Img):
            # get the values of the image
            observed_vals = self.get_observed_colours(frame.image)
            # get the upper limit of the patch values
            if frame.units == 'DN':
                patch_max = 2**BIT_DEPTH - bad_dn
            elif frame.units == 'DN/s':
                patch_max = (2**BIT_DEPTH - bad_dn) / frame.exposure
            elif frame.units == 'Reflectance':
                patch_max = 1.0
            else:
                raise ValueError(f"TODO: Define max value for {frame.units}")
            # tile the patch_max to match the number of patches
            patch_max = np.tile([patch_max], (len(self.patch_names),1))
            # check if any of the observed values are above the patch max
            bad_idx = np.around(observed_vals,2) >= np.around(patch_max,2)
            self.bad_patches = bad_idx

        # if a RGB or HRC or WAC_RGB frame
        elif isinstance(frame, (RGB, WAC_RGB, HRC)):
            # get the values of the image
            observed_vals = self.get_observed_colours(frame.rgb_image)
            # get the upper limit of the patch values
            if frame.units == 'DN':
                patch_max = np.array([2**BIT_DEPTH - bad_dn] * 3)
            elif frame.units == 'DN/s':
                red_patch_max = (2**BIT_DEPTH - bad_dn) / frame.red.exposure
                green_patch_max = (2**BIT_DEPTH - bad_dn) / frame.green.exposure
                blue_patch_max = (2**BIT_DEPTH - bad_dn) / frame.blue.exposure
                patch_max = np.array([red_patch_max,
                                        green_patch_max,
                                        blue_patch_max])   
            elif frame.units == 'Reflectance':
                patch_max = np.array([1.0] * 3)
            else:
                raise ValueError(f"TODO: Define max value for {frame.units}")
            # tile the patch_max to match the number of patches
            patch_max = np.tile([patch_max], (len(self.patch_names),1))        
            # for each channel, check if the observed values are above the patch max
            bad_idx = np.around(observed_vals,2) >= np.around(patch_max,2)
            self.bad_patches = bad_idx

        # if a multispectral cube frame
        elif isinstance(frame, MSC):
            # get the observed colours from the reflectance values
            observed_vals, observed_stds = self.get_observed_spectra(frame, method='mean')
            # get the upper limit of the patch values            
            if frame.units == 'DN':
                patch_max = np.array([2**BIT_DEPTH - bad_dn] * len(frame.imgs))
            elif frame.units == 'DN/s':
                patch_max = np.array([(2**BIT_DEPTH - bad_dn) / img.exposure for img in frame.imgs])
            elif frame.units == 'Reflectance':
                patch_max = np.array([1.0] * frame.n_bands)
            else:
                raise ValueError(f"TODO: Define max value for {frame.units}")
            # tile the patch_max to match the number of patches
            patch_max = np.tile([patch_max], (len(self.patch_names),1))
            # check if any of the observed values are above the patch max
            bad_idx = np.around(observed_vals,2) >= np.around(patch_max,2)
            self.bad_patches = bad_idx

    def get_observed_colours(self, image: NDArray, show: bool=False) -> NDArray:
        """Extract the patch colour values from the given image.

        :param image: The image containing the calibration target
        :type image: NDArray
        :param show: Show the sampled patches on the image, defaults to False
        :type show: bool, optional
        :return: An array giving the values of each patch in the image for
            each channel of the frame
        :rtype: NDArray
        """                
        width, height, rectangle, samples = self.get_target_dimensions()

        # we use the colour detection library to sample the patches.
        # Note that we set the reference values to None, as we don't want
        # the algorithm to check the orientation of the patches, as the
        # frame we are using might not be an approximate of the colour checker
        # colours. The orientation should have been determined in the
        # find_calibration_target method.
        patch_data = sample_colour_checker(
                            image, 
                            self.target_outline, 
                            rectangle, 
                            samples,
                            working_width=width,
                            working_height=height,
                            reference_values=None)
        
        if show:             
            # draw where the patches are on the sampled calibration target
            masks_i = np.zeros(patch_data.colour_checker.shape)
            for i, mask in enumerate(patch_data.swatch_masks):
                masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1            
            colour.plotting.plot_image(
                colour.cctf_encoding(
                    np.clip(patch_data.colour_checker + masks_i * 0.25, 0, 1)))

        return patch_data.swatch_colours

    def compute_ccm(self, 
                observed_cols: NDArray,
                reference_cols: NDArray) -> NDArray:
        """Compute the colour correction matrix and gamma curve for the 
        calibration target, from the given observed and reference values.

        :param observed_cols: Array of observed colours for each patch
        :type observed_cols: NDArray
        :param reference_cols: Array of reference colours for each patch
        :type reference_cols: NDArray
        :return: 3x3 Colour correction matrix
        :rtype: NDArray
        """        
        # check the observed and reference values have the same shape
        if observed_cols.shape != reference_cols.shape:
            raise ValueError(f"Observed values shape {observed_cols.shape} " \
                             f"does not match reference values shape " \
                             f"{reference_cols.shape}")
        if observed_cols.shape[1] != 3:
            raise ValueError(f"Observed values must have 3 channels, " \
                             f"got {observed_cols.shape[1]} channels")
        if reference_cols.shape[1] != 3:
            raise ValueError(f"Reference values must have 3 channels, " \
                             f"got {reference_cols.shape[1]} channels")
    
        # drop any row with nan values
        mask = np.all(~np.isnan(observed_cols), axis=1) & np.all(~np.isnan(reference_cols), axis=1)  
        observed_cols = observed_cols[mask]
        reference_cols = reference_cols[mask]

        ccm = colour.matrix_colour_correction(observed_cols, reference_cols)

        # check the diagnonals of the ccm are all >0
        if not np.all(np.diag(ccm) > 0):
            print("Colour Correction Matrix (CCM) has non-positive " \
                             "diagonal elements. Replacing with identity matrix.")
            ccm = np.eye(3)

        # apply the ccm to the reference values to get the corrected values
        cor_vals = colour.apply_matrix_colour_correction(observed_cols, ccm)
        # find the gamma curve                    
        fit, covar = curve_fit(gamma_curve, 
                                cor_vals.flatten(), 
                                reference_cols.flatten(),
                                p0=[1.0])
        # check the fit is valid
        if not np.isfinite(fit[0]) or fit[0] <= 0:
            print(f"Invalid gamma fit: {fit[0]}")
            fit[0] = 1.0
        self.ccm = ccm
        self.gamma = fit[0]

        return ccm, fit[0]

    def calibrate_colour(self, 
                frame: Union['RGB', 'WAC_RGB', 'HRC'],                 
                show: bool=False) -> None:
        """Calibrate the Colour Correction Matrix (CCM) and Gamma 
        for the given frame.   

        :param frame: The RGB frame containing the calibration target
        :type frame: Union['RGB', 'WAC_RGB', 'HRC']
        :param show: Show the calibration steps, defaults to False
        :type show: bool, optional
        :return: 3x3 colour correction matrix for the scene
        :rtype: NDArray
        """    
        # check that the outline of the target has been set
        if self.target_outline is None or self.target_outline.size == 0:
            raise ValueError("Calibration target outline not set." \
                "Please run find_target_outline(frame) or draw_target_outline(frame) first.")    
        
        if frame.units != 'Reflectance':
            drgb_image = frame.get_image('bpu') # get vals from raw image            
            obs_ct_dRGB_vals = self.get_observed_colours(drgb_image, show=show)
            # get the reference values
            ref_ct_sRGB_vals = self.patch_ref_sRGB
            # compute the colour correction matrix

            # apply bad patches
            if len(self.bad_patches) > 0:
                bad_patch_mask = self.bad_patches
                # if any line has a True value, it is a bad patch, so we need to discar all 3 channels
                bad_patch_mask = np.tile(np.any(bad_patch_mask, axis=1), (bad_patch_mask.shape[1], 1)).T 
            else:
                # if no bad patches, use all patches
                bad_patch_mask = np.zeros((len(self.patch_names), 3), dtype=bool)

            obs_masked = obs_ct_dRGB_vals.copy()
            ref_masked = ref_ct_sRGB_vals.copy()

            obs_masked[bad_patch_mask] = np.nan
            ref_masked[bad_patch_mask] = np.nan

            ccm, gamma = self.compute_ccm(obs_masked, ref_masked)
            frame.ccm = ccm
            frame.gamma = gamma
        else:
            # get the observed values from the reflectance values
            refl_image = frame.rgb_image
            obs_ct_refl_vals = self.get_observed_colours(refl_image, show=show)
            obs_ct_dRGB_vals = obs_ct_refl_vals

            # get the reference values in sRGB
            ref_ct_RGB_vals = self.patch_ref_sRGB

            # apply bad patches
            if len(self.bad_patches) > 0:
                bad_patch_mask = self.bad_patches
            else:
                # if no bad patches, use all patches
                bad_patch_mask = np.zeros((len(self.patch_names), 3), dtype=bool)

            obs_masked = obs_ct_dRGB_vals.copy()
            ref_masked = ref_ct_RGB_vals.copy()

            obs_masked[bad_patch_mask] = np.nan
            ref_masked[bad_patch_mask] = np.nan

            ccm, gamma = self.compute_ccm(obs_masked, ref_masked)
            
            # set the ccm and gamma on the frame
            frame.ccm = ccm
            frame.gamma=gamma # leave gamma as linear, as we expect the reflectance units to be linear.

        if show:
            # apply the ccm to the observed calibration target and compare
            # to the reference values
            # get reference values for given illuminant
            ref_colour_checker = CCS_COLOURCHECKERS[self.colour_checker]     
            illuminant_ccs = colour.CCS_ILLUMINANTS[
                        "CIE 1931 2 Degree Standard Observer"][self.illuminant]
            ref_colour_checker = colour.characterisation.ColourChecker(
                    'Reference Patch Colours', 
                    ref_colour_checker.data, 
                    illuminant=illuminant_ccs,
                    rows=ref_colour_checker.rows, 
                    columns=ref_colour_checker.columns)
            
            # convert to observed sRGB to xyY and build colourchecker
            cor_ct_sRGB_vals = np.dot(ccm, obs_ct_dRGB_vals.T).T

            # cor_ct_sRGB_vals = obs_ct_sRGB_vals
            # apply the gamma correction to the observed values
            cor_ct_sRGB_vals = gamma_curve(cor_ct_sRGB_vals, self.gamma)

            cor_ct_xyY_vals = colour.XYZ_to_xyY(
                    colour.RGB_to_XYZ(cor_ct_sRGB_vals, 'sRGB', illuminant_ccs))
            
            cor_colour_checker = colour.characterisation.ColourChecker(
                    'Recovered Patch Colours', 
                    dict(zip(ref_colour_checker.data.keys(), cor_ct_xyY_vals)), 
                    illuminant=illuminant_ccs,
                    rows=ref_colour_checker.rows, 
                    columns=ref_colour_checker.columns)
            
            colour.plotting.plot_multi_colour_checkers([ref_colour_checker, cor_colour_checker])   
            
            # draw a plot of the rgb values against one another to see the trend            
            plt.style.use('default') # set style to default
            fig, ax = plt.subplots(1, 1)            
            cols = np.clip(self.patch_ref_sRGB, 0,1)
            for patch in range(len(self.patch_names)):
                ax.scatter(
                        cor_ct_sRGB_vals[patch,0], 
                        self.patch_ref_sRGB[patch,0], 
                        color=cols[patch,:].flatten(),
                        edgecolor='red'
                        )
                ax.scatter(
                        cor_ct_sRGB_vals[patch,1], 
                        self.patch_ref_sRGB[patch,1], 
                        color=cols[patch,:].flatten(),
                        edgecolor='green'
                        )
                ax.scatter(
                        cor_ct_sRGB_vals[patch,2], 
                        self.patch_ref_sRGB[patch,2], 
                        color=cols[patch,:].flatten(),
                        edgecolor='blue')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('Corrected sRGB')
            ax.set_ylabel('Reference sRGB')
            # set square axis
            ax.set_aspect('equal', adjustable='box')
            # set title
            ax.set_title('Corrected vs Reference sRGB')
            # set x and y limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # save the plot
            fig.savefig(Path(frame.out_dir, 'calibration_target_sRGB_fit.png'))

    def get_observed_spectra(self,
                                frame,
                                method: Literal['mean', 'gauss-fit'],
                                show: bool=False) -> Tuple[NDArray, NDArray]:
        """Extract the patch values from each spectral band of an MSC stack.

        :param stack: The image stack containing the calibration target
        :type stack: NDArray
        :return: An array giving the values of each patch in the image for
        each channel of the frame
        :rtype: NDArray
        """
        width, height, rectangle, samples = self.get_target_dimensions()
        
        obs_vals = []
        obs_ave = []
        obs_std = []
        for band in frame.imgs:
            cwl = band.cwl

            # we use the colour detection library to sample the patches.
            # Note that we set the reference values to None, as we don't want
            # the algorithm to check the orientation of the patches, as the
            # frame we are using might not be an approximate of the colour
            # checker colours. The orientation should have been determined in
            # the find_calibration_target method.
            
            patch_data = sample_colour_checker(band.image,
                                            self.target_outline,
                                            rectangle,
                                            samples,
                                            working_width=width,
                                            working_height=height,
                                            reference_values=None)
            
            obs_vals.append(patch_data.swatch_colours)
            col_check_img = patch_data.colour_checker
            swatch_ave = []
            swatch_std = []

            # draw where the patches are on the sampled calibration target
            if show:
                # Using the additional data to plot the colour checker and masks
                masks_i = np.zeros(col_check_img.shape)
                for i, mask in enumerate(patch_data.swatch_masks):
                    masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
                
                # normalise the colour checker image
                col_check_disp_img = col_check_img/col_check_img.max()

                # plot the colour checker image with the masks
                colour.plotting.plot_image(
                    colour.cctf_encoding(
                        np.clip(col_check_disp_img + masks_i * 0.25, 0, 1)),
                                    title=f"{band.filter_id} {cwl} nm Sampling")

            # create a figure to plot the pixel values and fits of each patch
            if show:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(self.rows, self.cols,
                                                figsize=(self.cols, self.rows))            
            # compute the observed value for each patch
            for p, mask in enumerate(patch_data.swatch_masks):
                # get the pixel values covered by the patch mask
                patch = col_check_img[mask[0]:mask[1], mask[2]:mask[3], ...]

                # get the mean and standard deviation of the patch
                ave = np.mean(patch, axis=(0,1))
                std = np.std(patch, axis=(0,1))

                # TODO if the standard deviation is 0, implies an overexposed 
                # patch, so remove it from the analysis

                # 2 methods for computing the observed values
                #   A. mean of the patch
                #   B. Gaussian fit to the patch histogram 
                #        - good for discarding marks/scratches/dust on the patch
                if method == 'mean':
                    # use mean and standard deviation of th patch values
                    swatch_ave.append(ave)
                    swatch_std.append(std)

                elif method == 'gauss-fit':
                    # get the histogram of the patch pixel values
                    histo_range = np.max([4*std, ave/10]) 
                    ydata, xdata = np.histogram(patch,
                                                bins=patch.size//2,
                                                range=(ave-histo_range, ave+histo_range))
                    
                    # fit Gaussian to the histogram
                    try:
                        params, *_ = curve_fit(gauss,
                                          xdata[:-1],
                                          ydata,
                                          [10, ave, std])
                        fit_amp, fit_ave, fit_std = params
                    except RuntimeError:
                        # if the fit fails, use the mean and std of the patch
                        fit_amp = 0.0
                        fit_ave = ave
                        fit_std = std

                    swatch_ave.append(fit_ave)
                    swatch_std.append(fit_std)

                    # show the histogram and fit for each patch
                    if show:                        
                        fit_y = gauss(xdata[:-1],fit_amp,fit_ave,fit_std)
                        r = p // self.cols # current column                    
                        c = p % self.cols # current row
                        col = colour.cctf_encoding(
                                        np.clip(self.patch_ref_sRGB[p], 0,1))
                        alp = (fit_ave / col_check_img.max())**2
                        ax[r][c].plot(xdata[:-1], ydata, 'o', c=col,alpha=alp)
                        ax[r][c].plot(xdata[:-1], fit_y, '-', c=col,alpha=alp)
                        ax[r][c].set_axis_off()
                        ax[r][c].axvline(fit_ave, ls='--')
                        ax[r][c].axvline(ave, ls='-.')
                        ax[r][c].set_title(f"{self.patch_names[p]}", c=col, 
                                                            fontsize='x-small')

            if show:
                fig.suptitle(f"{band.filter_id} {cwl} nm Gauss-Fits")
                fig.tight_layout()                

            obs_ave.append(np.array(swatch_ave))
            obs_std.append(np.array(swatch_std))

        # convert the lists to ndarrays
        obs_ave = np.array(obs_ave).T
        obs_std = np.array(obs_std).T

        return obs_ave, obs_std

    def calibrate_reflectance(self,
                            frame: 'MSC',
                            method: Literal['mean', 'gauss-fit'] = 'gauss-fit',  
                            bad_patch_list: Optional[List[str]]=None,                          
                            show: bool=False) -> None:
        """Calibrate the given frame to the values of the colour checker patches 
        in each band of the frame, setting the reflectance correction 
        coefficients for each band.

        :param frame: The MSC frame to calibrate the reflectance values for
        :type frame: MSC
        :param show: plot intermediary steps, defaults to False
        :type show: bool, optional
        """        

        # check frame units
        if frame.units == 'Reflectance':
            print("Frame already in reflectance, skipping calibration")
            return

        # get the observed spectra
        # TODO propagate uncertainties
        obs_ave, obs_std = self.get_observed_spectra(frame, method, show=False)
        # get the reference spectra
        ref_refl = self.sample_patch_spectra(frame)

        colors = colour.cctf_encoding(np.clip(self.patch_ref_sRGB, 0,1))

        # get bad patch mask
        if len(self.bad_patches) > 0:
            # if no bad patches are given, use all patches
            bad_patch_mask = self.bad_patches
        else:
            # if bad patches are given, create a mask for them
            bad_patch_mask = np.zeros((len(self.patch_names), frame.n_bands), dtype=bool)

        if bad_patch_list is not None:
            # modulate th abd patch mask with the list of bad patches
            for bad_patch in bad_patch_list:
                if bad_patch in self.patch_names:
                    idx = self.patch_names.index(bad_patch)
                    # set the bad patch mask for this patch to True
                    bad_patch_mask[idx, :] = True
                else:
                    print(f"Patch {bad_patch} not found in patch names, skipping")
                    
        # prepare plot of the observed vs reference values fit for each band
        if show:
            ncols=3
            nrows=frame.n_bands//ncols
            plt.style.use('default')
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,sharey=True,
                                            figsize=(2*3, 2*frame.n_bands//3))
        
        # compute the reflectance coefficients for each band of the frame
        refl_coeffs = []
        refl_offsets = []
        for b, band in enumerate(frame.imgs):
            band_bad_patches = bad_patch_mask[:, b].flatten() # get the bad patches for this band
            x = obs_ave[~band_bad_patches,b] # here we are using the bad band locations to filter the observed values
            y = ref_refl[~band_bad_patches,b]            
            result = linregress(x, y)
            refl_coeffs.append(result.slope)
            refl_offsets.append(result.intercept)
            # TODO propagate uncertainties of the regression line

            # draw observed vs reference values fit for this band
            if show:                
                c = b % ncols # corrent column
                r = b // ncols # current row
                if nrows == 1:
                    this_ax = ax[c]
                else:
                    this_ax = ax[r][c]
                col = colors[~band_bad_patches]
                this_ax.scatter(x, y, c=col)
                this_ax.plot(x, result.intercept + result.slope*x, 'r')
                this_ax.set_xlabel(f'Power {frame.units}')
                this_ax.set_ylabel(f'Reflectance')
                this_ax.set_title(f'{band.filter_id} {band.cwl} nm')

        if show:
            fig.suptitle(f'Reflectance Calibration for {frame.camera} {frame.sol} {frame.scene} {frame.trial}', fontsize='x-small')
            fig.tight_layout()
            # save the plot
            filter_id_str = '_'.join(frame.filter_ids)
            fig.savefig(Path(frame.out_dir,
                            f'reflectance_calibration_band_fits_{frame.sol}_{frame.scene}_{frame.trial}_{frame.camera}_{filter_id_str}.png'))

        # set the reflectance coefficients for each band of the frame
        for b, band in enumerate(frame.imgs):
            band.refl_coeff = refl_coeffs[b]
            band.refl_offset = refl_offsets[b]
            print(f"{band.filter_id} {band.cwl} nm: coeff={refl_coeffs[b]:0.3}1/DN/s, offset={refl_offsets[b]:0.3}")
        # set the reflectance coefficients and offsets on the frame
        frame.refl_coeffs = np.array(refl_coeffs)
        frame.refl_offsets = np.array(refl_offsets)
            
        # plot the spectral reflectance values for each band against the reference spectra
        if show:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(self.rows, self.cols, sharex=True, 
                                sharey=True, figsize=(2*self.cols, 2*self.rows))
            for p, patch_name in enumerate(self.patch_names):                
                # get the observed values for this patch
                cwls = frame.cwls # wavelengths of the frame                
                obs_vals = obs_ave[p, :]
                # scale the observed values to reflectance
                fit_spec = obs_vals*refl_coeffs + refl_offsets
                ref_spec = ref_refl[p, :]
                if bad_patch_list is not None:
                    if patch_name in bad_patch_list:
                        col = 'k'
                    else:
                        col = colors[p]
                else:
                    col = colors[p]
                plt.style.use('dark_background')
                # draw the observed vs reference values fit for this patch
                r = p // self.cols # current row
                c = p % self.cols # current column                
                if self.rows == 1:
                    this_ax = ax[c]
                else:
                    this_ax = ax[r][c]
                this_ax.plot(cwls,ref_spec,c=col,ls='--',label='Ref.',marker='x')
                this_ax.plot(cwls,fit_spec,c=col,ls='-',label='Fit',marker='o')
                # TODO add propogation of uncertainties to errorbars
                this_ax.set_ylim(0, 1)
                this_ax.set_title(f"{patch_name.title()}", c=col,
                                                            fontsize='x-small')
                this_ax.legend(fontsize='x-small')
                # set title of legend
            fig.supxlabel('Wavelength (nm)')
            fig.supylabel('Reflectance')
            fig.suptitle(f'Reflectance Calibration Patch Fits for {frame.camera} {frame.sol} {frame.scene} {frame.trial}')
            fig.tight_layout()
        # save the plot
            filter_id_str = '_'.join(frame.filter_ids)
            fig.savefig(Path(frame.out_dir,
                            f'reflectance_calibration_patch_fits_{frame.sol}_{frame.scene}_{frame.trial}_{frame.camera}_{filter_id_str}.png'))

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
        self.tag = None  # tag for the image, reserved for warping to describe feature that rectification is optimised for
        
        self.out_dir = file_dict['out_dir']

        parts = self.filepath.stem.split('_')
        self.channel = parts[2].split('-')[-1]

        # from metadata
        # read the metadata from the image file using the PIL exif reader
        img = Image.open(self.filepath)
        metadata = img.info

        self.pan = 0.0 # no metadata given
        self.tilt = 0.0 # no metadata given
        
        if parts[4][-2] == 'm':
            self.exposure = float(parts[4].split('ms')[0])/1000
        else:
            self.exposure = float(parts[4].split('s')[0])

        self.timestamp = ('_').join([parts[0], parts[1]])
        self.camera = 'WAC' # Not in the metadta, not always in the filename. Fix to WAC for now, update later to handle HRC. aupe_info.cam_dict[int(metadata['AU_camNum'])]

        # from image
        # read the image data
        self.image = np.array(img)
        # derive metadata from the image data
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.dtype = self.image.dtype
        self.units = 'DN'
        
        # set cwl and fwhm
        self.cwl = aupe_info.cwl[self.channel]
        self.fwhm = aupe_info.fwhm[self.channel]
        self.filter_id = aupe_info.filter_id[self.channel]

        # load filter transmission data
        self.filter_response = self.load_filter_response()

        # initialise the stretch parameters
        self.stretch = {
            'raw': {
                'factor': None,
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
            '99s': {
                'factor': None,
                'roi': [None, None, None, None]
            }
        }

        self.refl_coeff = None  # reflectance correction coefficient
        self.refl_offset = None  # reflectance correction offset
            
    def load_filter_response(self) -> Dict:
        """Method to load the filter transmission data for the filter id 
        of the Img

        :return: Filter transmission data
        :rtype: NDArray
        """        
        repsonse_path = Path('.', 'data', 'response_files')
        # get the camera
        # get the filter ID
        response_file = Path(repsonse_path, self.camera,
                                f"{self.filter_id}.csv")
        if not response_file.exists():
            # e.g. if loading HRC, then filter responses need to be loaded after debayering
            print(f"Filter response file {response_file} does not exist")
            return None
        # read the csv file into a pandas dataframe
        filter_df = pd.read_csv(response_file, index_col=0)
        # convert the dataframe to a dictionary of the index and values
        filter_response = {}
        filter_response['wavelengths'] = filter_df.index.values.astype(np.float32)
        filter_response['response'] = filter_df.values.astype(np.float32)

        return filter_response
    
    def plot_filter_response(self, ax: plt.Axes=None) -> plt.Axes:
        """Plot the filter response function for the Img"""
        wavelengths = self.filter_response['wavelengths']
        response = self.filter_response['response']
        
        if ax is None:
            plt.style.use('default')
            fig, ax = plt.subplots()
        ax.plot(wavelengths, response, label=self.filter_id)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Response')
        ax.set_title(f'Filter Response Function for {self.camera} {self.filter_id}')
        ax.legend()

        return ax

    def exposure_correct(self):
        """Correct for the exposure of the image, by converting
        to units of DN/s
        """        
        # check if the image is already in DN/s
        if self.units == 'DN/s':
            print("Image has already been exposure corrected")
        else:
            self.image = np.divide(self.image, self.exposure)
            self.units = 'DN/s'
            self.dtype = self.image.dtype
            # update stretch coefficients
            if self.stretch['raw']['factor'] is not None:
                self.stretch['raw']['factor'] = self.stretch['raw']['factor'] * self.exposure
            if self.stretch['bps']['factor'] is not None:
                self.stretch['bps']['factor'] = self.stretch['bps']['factor'] * self.exposure
            if self.stretch['wps']['factor'] is not None:
                self.stretch['wps']['factor'] = self.stretch['wps']['factor'] * self.exposure
            if self.stretch['99s']['factor'] is not None:
                self.stretch['99s']['factor'] = self.stretch['99s']['factor'] * self.exposure
 
    def flat_field(self):
        pass

    def bias_correction(self):
        pass

    def extract_stretch_coefficient(self, 
                method: Literal['raw', 'bps', 'wps', '99s']='raw',
                wp_roi: Tuple[int, int, int, int]=None) -> float:
        """Get the stretch coefficient for the given image according to
        the selected method.
        'raw': no stretch, just divide by the max value for the bit-depth
        'bps': "brightest pixel stretch" - stretches such that the brightest pixel
        has a value of 1.0.
        '99s': "99th percentile stretch" - stretches such that the 99th percentile
        of the image is 1.0.
        'wps': "white patch stretch" - stretches such that the white patch of the
        MacBeth colorchecker has a value of 1.0.

        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99s']
        """      
        if method == 'raw':
            # set the stretch coefficient to 1.0 / max bit-depth value
            self.stretch['raw'] = {}
            if self.units == 'DN':
                self.stretch['raw']['factor'] = 1.0 / ((2**BIT_DEPTH) - 1)
            elif self.units == 'DN/s':
                self.stretch['raw']['factor'] = 1.0 / ((2**BIT_DEPTH) - 1) * self.exposure
            elif self.units == 'Reflectance':
                self.stretch['raw']['factor'] = 1.0
            else:
                raise ValueError(f"Unknown units {self.units} for raw image")
            self.stretch['raw']['roi'] = (0, 0, self.width, self.height)
        elif method == 'bps':
            # get the brightest pixel value and location
            bp_val = np.nanmax(self.image)            
            bp_loc = np.unravel_index(np.argmax(self.image, axis=None), self.image.shape)
            # set the stretch coefficient to 1.0 / bp_val
            self.stretch['bps'] = {}
            self.stretch['bps']['factor'] = 1.0 / bp_val
            self.stretch['bps']['roi'] = (bp_loc[0], bp_loc[1], 1, 1)
        elif method == '99s':
            # get the 99th percentile pixel value and location
            pct_val = np.percentile(self.image, 99)            
            # set the stretch coefficient to 1.0 / bp_val
            self.stretch['99s'] = {}
            self.stretch['99s']['factor'] = 1.0 / pct_val
            self.stretch['99s']['roi'] = (0, 0, self.width, self.height)
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
        else:
            raise ValueError(f"Unknown stretch method: {method}")
    
        return self.stretch[method]['factor']

    def reset_stretch_coefficient(self,
                method: Literal['all', 'raw', 'bps', 'wps', '99s']='all'):
        """Reset the stretch coefficient for the given image according to
        the selected method.
        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99s']
        """
        if method == 'all':             
            self.stretch = {
                'raw': {
                    'factor':  None,
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
                '99s': {
                    'factor': None,
                    'roi': [None, None, None, None]
                }
            }
        elif method == 'raw':
            self.stretch['raw'] = {
                'factor': None,
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
        elif method == '99s':
            self.stretch['99s'] = {
                'factor': None,
                'roi': [None, None, None, None]
            }
        else:
            raise ValueError(f"Unknown stretch method: {method}")

    def apply_stretch(self, stretch_method: Literal['raw', 'bps', 'wps', '99s']='raw'):
        """Apply the stretch coefficient to the image, and return the stretched image.
        Stretched iamge is always in the range of 0.0 to 1.0.

        :param stretch_method: method for finding the stretch coefficient
        :type stretch_method: Literal['raw', 'bps', 'wps', '99s']
        :return: stretched image
        :rtype: np.ndarray
        """
        print(f"Stretching image using {STRETCH_DICT[stretch_method]}")
        if self.stretch[stretch_method]['factor'] is None:
            self.extract_stretch_coefficient(stretch_method)
        print(f'Applying stretch factor of {self.stretch[stretch_method]["factor"]}')
        disp_img = np.clip(self.image * self.stretch[stretch_method]['factor'], 0.0, 1.0)        
        
        return disp_img
    
    def apply_reflectance_calibration(self):
        """Apply the reflectance calibration to the image, if the reflectance
        correction coefficients are set.
        """
        # check if the exposure correction has been applied
        if self.units == 'Reflectance':
            print('Reflectance calibration has already been applied.')
            return
        elif self.units != 'DN/s':
            raise ValueError("Exposure correction has not been applied. "
                             "Please apply the exposure correction before applying the reflectance calibration.")

        if self.refl_coeff is not None and self.refl_offset is not None:
            # apply the reflectance correction
            self.image = (self.image * self.refl_coeff) + self.refl_offset
            # update the units
            self.units = 'Reflectance'
            # update the stretch coefficients
            for method in self.stretch:
                if self.stretch[method]['factor'] is not None:
                    self.stretch[method]['factor'] *= self.refl_coeff
        else:
            raise ValueError("Reflectance correction coefficients are not set. "
                             "Please set them before applying the reflectance calibration.")

    def get_image(self, 
                stretch_method: Literal[None, 'raw', 'bps', 'wps', '99s']=None,
                dtype: Literal[np.uint8, np.uint16, np.float32, np.float64]=np.uint8
                ) -> np.ndarray:
        """Get a copy of the image data, optionally applying the stretch method
        :param stretch_method: method for finding the stretch coefficient
        :type stretch_method: Literal['raw', 'bps', 'wps', '99s'], optional
        :return: image data
        :rtype: np.ndarray
        """
        if stretch_method is None:
            image = self.image.copy()
        else:
            image = self.apply_stretch(stretch_method)
        return image

    def show_image(self, 
                   stretch_method: Literal['raw', 'bps', 'wps', '99s']='raw'
                ) -> Tuple[plt.Figure, plt.Axes]:
        """Display the image using matplotlib,
        and optionally show the histogram of the image data.
        """  

        title = f"{self.sol} {self.scene} {self.trial} {self.channel} {self.filter_id} {self.cwl}±{int(self.fwhm/2)} nm ({stretch_method})"

        disp_img = self.get_image(stretch_method) # image is always in range of 0 - 1
        
        plt.style.use('default')
        fig, ax = plt.subplots(1,2, figsize=(8, 4))
        disp = ax[0].imshow(disp_img, vmin=0.0, vmax=1.0, cmap='viridis')
        # add colorbar
        plt.colorbar(disp,fraction=0.046, pad=0.10, orientation='horizontal')
        
        ax[1].hist(disp_img.ravel(), bins=256, color='gray', alpha=0.5)
        
        ax[1].set_xlabel(f"Pixel Value {self.units} ({stretch_method})")
        ax[1].set_ylabel("Frequency")
        ax[1].tick_params(labelleft=False, left=False)
        # add title and labels
        fig.suptitle(title)
        plt.show()  

        # # show the image at full resolution
        # plt.imshow(disp_img, interpolation='none')
        # plt.axis('off')
        # # set the title
        # plt.title(title)

        return fig, ax

    def export_image(self, stretch_method: Literal['raw', 'bps', 'wps', '99s']='bps'):
        """Export the image to a file, using the stretch method, in uint8 format.

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        """   
        
        if self.tag is not None:
            title = f"{self.sol}_{self.scene}_{self.trial}_{self.channel}_{self.tag}_{int(self.fwhm)}_nm_{stretch_method}_{self.tag}.png"
        else:
            title = f"{self.sol}_{self.scene}_{self.trial}_{self.channel}_{self.cwl}_{int(self.fwhm)}_nm_{stretch_method}.png"

        disp_img = self.get_image(stretch_method)

        # convert to uint8 - image should always be in range of 0 - 1
        out_img = (disp_img * 255).astype(np.uint8)
        
        # TODO - format metadata and check it works
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
        out_dir = self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / title
        plt.imsave(
            str(out_file.absolute()), 
            out_img, 
            vmin=0, vmax=255, cmap='gray', 
            format='png', 
            metadata=metadata)
    
        return out_file        

class RGB:
    def __init__(self,
                 rgb_imgs: Tuple[Img, Img, Img]):
        self.red = rgb_imgs[0]
        self.green = rgb_imgs[1]
        self.blue = rgb_imgs[2]
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)
        self.exposures = np.array([self.red.exposure, self.green.exposure, self.blue.exposure])
        self.ccm = np.empty((3,3))
        self.gamma = 1.0
        self.balance_vector = {
            'raw': np.zeros(3),
            'bpu': np.zeros(3),
            'bpb': np.zeros(3),
            '99b': np.zeros(3),
            '99u': np.zeros(3),
            'wps': np.zeros(3),
        }        
        self.camera = self.red.camera
        self.tag = None  # tag for the image, reserved for warping to describe feature that rectification is optimised for
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
    
    @classmethod
    def from_filedicts(cls, 
                    file_dicts: Tuple[Dict, Dict, Dict],
                    aupe_info: AupeInfo) -> 'RGB':
        """Create an RGB object from a tuple of file dictionaries.
        :param file_dicts: Tuple of file dictionaries for red, green, and blue channels
        :type file_dicts: Tuple[Dict, Dict, Dict]
        :param aupe_info: AupeInfo object containing camera and filter information
        :type aupe_info: AupeInfo
        :return: RGB object
        :rtype: RGB
        """
        # create Img objects for each channel
        red_img = Img(file_dicts[0], aupe_info)
        green_img = Img(file_dicts[1], aupe_info)
        blue_img = Img(file_dicts[2], aupe_info)

        # create RGB object
        rgb = cls((red_img, green_img, blue_img))
        return rgb
    
    def exposure_correct(self):
        """Exposure correct each channel
        """      
        if self.units == 'DN/s':
            print('Frame has already been exposure corrected')
        else:
            self.red.exposure_correct()
            self.green.exposure_correct()
            self.blue.exposure_correct()
            self.units = self.red.units
            self.dtype = self.red.dtype
            # update the rgb image  
            self.rgb_image = np.stack([
                                    self.red.image, 
                                    self.green.image, 
                                    self.blue.image], axis=2)

            # update the balance vector
            for method in self.balance_vector.keys():
                self.balance_vector[method] = self.balance_vector[method] * self.red.exposure

    def flat_field(self):
        pass

    def bias_subtract(self):
        pass

    def load_ccm(self,
                 path: str=None,
                 camera: str=None,
                 sol: str=None,
                 scene: str=None,
                 trial: str=None):
        """Load the colour correction matrix from a csv file
        """ 
        if camera is None:
            camera = self.camera
        if sol is None:
            sol = self.sol
        if scene is None:
            scene = self.scene
        if trial is None:
            trial = self.trial

        if path is None:
            # get the ccm directory
            proc_dir = self.out_dir.parent.parent.parent.parent.parent
            ccm_dir = Path(proc_dir, sol, scene, trial, camera, 'RGB')
            # check the ccm dir exists
            if not ccm_dir.exists():
                raise FileNotFoundError(f"CCM directory {ccm_dir} does not exist")
            # load the ccm from a csv file
            filename = f"{sol}_{scene}_{trial}_{camera}_ccm.csv"
            path = Path(ccm_dir, filename)
            # check the file exists
            if not path.exists():
                raise FileNotFoundError(f"CCM file {filename} does not exist in {ccm_dir}")

        ccm_df = pd.read_csv(path, header=None)
        # read the gamma value from this
        self.ccm = ccm_df.to_numpy()
        print(f"CCM loaded from {path}")

    def export_ccm(self):
        """Export the colour correction matrix to a csv file
        """        
        # save the ccm to a csv file
        ccm_df = pd.DataFrame(self.ccm)
        # add the camera, sol, scene, trial to the filename
        # TODO figure if there is any other metadata we can apply - e.g. in shade, in sun, indoors etc.
        filename = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_ccm.csv"
        # add gamma to a line in the csv

        ccm_df.to_csv(Path(self.out_dir, filename), index=False, header=False)
        print(f"CCM saved to {self.out_dir}/{filename}")

    def apply_ccm(self):
        """Apply the colour correction matrix to the RGB image
        """
        # if image is in relfectance units, get image,
        if self.units == 'Reflectance':
            drgb_image = self.rgb_image.copy()
        else:
            drgb_image  = self.get_image('bpu')
        # check if the ccm is set
        if self.ccm is None:
            print("No colour correction matrix set")
            # search for the latest calibration target in the 
        srgb_image = colour.apply_matrix_colour_correction(drgb_image, self.ccm)
        # apply the gamma correction
        srgb_image = np.clip(srgb_image, 0.0, None) # clamp negative vals to 0
        # srgb_image = gamma_curve(srgb_image, self.gamma) # don't do gamma anymore - data is linear. Anyway, should be doing this correction before matrix application.
        srgb_image = np.clip(srgb_image, 0.0, 1.0)

        srgb_image = colour.cctf_encoding(srgb_image)

        return srgb_image

    def extract_balance_vector(self, 
                               method: Literal[
                                            'raw', 
                                            'bpb', 
                                            'bpu', 
                                            'wps',     
                                            '99b',
                                            '99u']='raw'):
        
        """Extract the stretch coefficient for each channel of the RGB image.

        :param method: stretch method, defaults to 'raw'
        :type method: Literal['raw', 'bpb', 'bpu', 'wpb', 'wpu', '99b', '99u'], optional
        """        

        if method == 'raw':
            # no stretch, channels not balanced
            # red stretch
            r_stretch = self.red.extract_stretch_coefficient(method)        
            # green stretch
            g_stretch = self.green.extract_stretch_coefficient(method)
            # blue stretch
            b_stretch = self.blue.extract_stretch_coefficient(method)
        elif method == 'bpb':
            # independent brightest pixel stretch, balancing channels
            # red stretch
            r_stretch = self.red.extract_stretch_coefficient('bps')        
            # green stretch
            g_stretch = self.green.extract_stretch_coefficient('bps')
            # blue stretch
            b_stretch = self.blue.extract_stretch_coefficient('bps')
        elif method == 'bpu':
            # all channel brightest pixel stretch, unbalanced
            # get the brightest pixel value and location
            bp_val = np.nanmax(self.rgb_image)            
            bp_loc = np.unravel_index(np.argmax(self.rgb_image, axis=None), self.rgb_image.shape)
            # set the stretch coefficient to 1.0 / bp_val
            r_stretch = 1.0 / bp_val
            g_stretch = 1.0 / bp_val
            b_stretch = 1.0 / bp_val
        elif method == '99b':
            # independent 99th percentile stretch, balancing channels
            # red stretch
            r_stretch = self.red.extract_stretch_coefficient('99s')        
            # green stretch
            g_stretch = self.green.extract_stretch_coefficient('99s')
            # blue stretch
            b_stretch = self.blue.extract_stretch_coefficient('99s')
        elif method == '99u':
            # all channel 99th percentile stretch, unbalanced
            # get the 99th percentile pixel value and location
            pct_val = np.percentile(self.rgb_image, 99)            
            # set the stretch coefficient to 1.0 / bp_val
            r_stretch = 1.0 / pct_val
            g_stretch = 1.0 / pct_val
            b_stretch = 1.0 / pct_val
        elif method == 'wps':
            # independent white patch stretch, balancing channels
            # draw the roi on the colour image stack
            # zoom in on the colorchecker target
            title = 'Select Calibration Target Approx. ROI'
            print('First draw an ROI around the calibration target')
            # make a copy of the rgb image scaled to 0-255 for display
            disp_img = (self.rgb_image * 255).astype(np.uint8)
            ct_roi = cv2.selectROI(title, cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
            # switch order of roi to (y, x, h, w)
            ct_roi = (ct_roi[1], ct_roi[0], ct_roi[3], ct_roi[2])           
            cv2.destroyWindow(title)             
            # get the white patch value
            title = 'Select White Patch ROI'
            print('Now draw an ROI around the white patch on the calibration target')
            ct_img = disp_img[ct_roi[0]:ct_roi[0]+ct_roi[2], ct_roi[1]:ct_roi[1]+ct_roi[3]]
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
            raise ValueError(f"Unknown stretch method: {method}")

        self.balance_vector[method] = np.array([r_stretch, g_stretch, b_stretch])

    def reset_balance_vector(self, 
                method: Literal['all', 'raw', 'bpb', 'bpu', 
                                'wps', '99b', '99u']='all') -> None:
        """Reset the balance vector to all zeroes

        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bpb', 'bpu', 'wps', '99b', '99u', 'all'], optional
        """
        if method == 'all':
            self.balance_vector = {
                'raw': np.zeros(3),
                'bpb': np.zeros(3),
                'bpu': np.zeros(3),
                'wps': np.zeros(3),
                '99b': np.zeros(3),
                '99u': np.zeros(3)
            }
            self.red.reset_stretch_coefficient('all')
            self.green.reset_stretch_coefficient('all')
            self.blue.reset_stretch_coefficient('all')
        elif method == 'raw':
            self.balance_vector[method] = np.ones(3)
            self.red.reset_stretch_coefficient('raw')
            self.green.reset_stretch_coefficient('raw')
            self.blue.reset_stretch_coefficient('raw')
        else:            
            self.balance_vector[method] = np.zeros(3)            
            self.red.reset_stretch_coefficient(method)
            self.green.reset_stretch_coefficient(method)
            self.blue.reset_stretch_coefficient(method)

    def apply_balance_vector(self, colour_correction: Literal['raw', 'bpb', 'bps', 'wps', '99p']='raw'):

        print(f"Stretching image using {STRETCH_DICT[colour_correction]}")
        if (self.balance_vector[colour_correction] == np.zeros(3)).all():
            self.extract_balance_vector(colour_correction)
        r_disp_img = np.clip(self.red.image * self.balance_vector[colour_correction][0], 0.0, 1.0)        
        g_disp_img = np.clip(self.green.image * self.balance_vector[colour_correction][1], 0.0, 1.0)
        b_disp_img = np.clip(self.blue.image * self.balance_vector[colour_correction][2], 0.0, 1.0)
                
        stretch_img = np.stack([r_disp_img, g_disp_img, b_disp_img], axis=2)
        return stretch_img

    def get_image(self,
                  colour_correction: Literal['raw', 'bpb', 'bpu', 'wps', '99b', '99u', 'ccm']='raw'
                  ) -> np.ndarray:
        """Get a copy of the image data, optionally applying the stretch method
        :param colour_correction: method for finding the stretch coefficient
        :type colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm'], optional
        :return: image data
        :rtype: np.ndarray
        """
        if colour_correction == 'ccm':
            # apply the colour correction matrix
            image = self.apply_ccm()
        else:
            if (self.balance_vector[colour_correction] == np.zeros(3)).all():
                self.extract_balance_vector(colour_correction)
            # apply the balance vector
            image = self.apply_balance_vector(colour_correction)
        
        return image

    def show_image(self, 
                   colour_correction: Literal['raw', 'bpb', 'bpu', 'wps', '99b','99u', 'ccm']='raw'):
        """Display the RGB image using matplotlib,
        and optionally show the histogram of the image data.
        """
        title = f"{self.sol} {self.scene} {self.trial} {self.camera} R({self.red.cwl} nm) G({self.green.cwl} nm) B({self.blue.cwl} nm) ({colour_correction})"

        disp_img = self.get_image(colour_correction)
        
        # # apply encoding
        # disp_img = colour.cctf_encoding(disp_img)

        plt.style.use('default')
        fig, ax = plt.subplots(1,2, figsize=(8, 4))  
        
        disp = ax[0].imshow(disp_img, vmin=0, vmax=1.0) # what constraints should be applied here???

        # for each channel, show a histogram - retain original image units
        ax[1].hist(disp_img[:,:,0].ravel(), bins=256, histtype='stepfilled', color='red', alpha=0.5)
        ax[1].hist(disp_img[:,:,1].ravel(), bins=256, histtype='stepfilled', color='green', alpha=0.5)
        ax[1].hist(disp_img[:,:,2].ravel(), bins=256, histtype='stepfilled', color='blue', alpha=0.5)
        ax[1].tick_params(labelleft=False, left=False)
        # label x axis with units
        ax[1].set_xlabel(f"Pixel Value {self.units} ({colour_correction})")
        # add title and labels
        fig.suptitle(title)
        plt.show()

        return fig, ax

    def export_image(self, 
                     colour_correction: Literal['raw', 'bpb', 'bpu', 'wps', '99b','99u', 'ccm']='raw',
                     show: bool=False) -> Path:
        """Export the image to an 8-bit RGB image file, using the stretch method

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        :param tag: Optional tag to add to the filename - e.g. the image feature
            used to perform rectification with, defaults to None
        :type tag: str, optional
        :param show: Whether to show the image after exporting, defaults to False
        :type show: bool, optional
        :return: Path to the exported image file
        :rtype: Path
        """   
        
        if self.tag is not None:
            # add the tag to the title
            title = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_RGB_{self.red.filter_id}{self.green.filter_id}{self.blue.filter_id}_{colour_correction}_{self.tag}.png"
        else:
            title = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_RGB_{self.red.filter_id}{self.green.filter_id}{self.blue.filter_id}_{colour_correction}.png"

        print(f"Exporting image using {STRETCH_DICT[colour_correction]}")
        if colour_correction == 'ccm':
            # apply the colour correction matrix
            disp_img = self.get_image('ccm')
            # convert to uint8
            disp_img = (disp_img * 255).astype(np.uint8)
        else:
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
            vmin=0, vmax=255, 
            format='png', 
            metadata=metadata)
        
        if colour_correction == 'ccm':
            self.export_ccm()

        if show:        
            # show the image at full resolution
            plt.style.use('default')
            fig, ax = plt.subplots(1,1, figsize=(8, 4))  
            ax.imshow(disp_img, vmin=0, vmax=255, interpolation='none')            
            # set the title
            fig.suptitle(title, fontsize='x-small')
    
        return out_file

class HRC(RGB):
    """HRC class inherits the methods of the RGB class, but handles HRC by
    loading the same un-debayered image into each channel.
    Then, during image debayering, it rewrites the red, green and blue image
    data. The allocation of the same HRC image to each channel
    is handled by the AupeIO class.
    """    
    def __init__(self,  rgb_imgs: Tuple[Img, Img, Img], aupe_info: AupeInfo):
        super().__init__(rgb_imgs)        
        self.debayered = False
        self.aupe_info = aupe_info
        
        # relabel the channels of the hrc r,g,b images, and update other
        # attributes
        self.red.channel = 'HRCR'
        self.red.cwl = aupe_info.cwl['HRCR']
        self.red.fwhm = aupe_info.fwhm['HRCR']  
        self.red.filter_id = aupe_info.filter_id['HRCR']
        self.red.filter_response = self.red.load_filter_response()
        self.green.channel = 'HRCG'
        self.green.cwl = aupe_info.cwl['HRCG']
        self.green.fwhm = aupe_info.fwhm['HRCG']
        self.green.filter_id = aupe_info.filter_id['HRCG']
        self.green.filter_response = self.green.load_filter_response()
        self.blue.channel = 'HRCB'
        self.blue.cwl = aupe_info.cwl['HRCB']
        self.blue.fwhm = aupe_info.fwhm['HRCB']
        self.blue.filter_id = aupe_info.filter_id['HRCB']
        self.blue.filter_response = self.blue.load_filter_response()

    @classmethod
    def from_filedicts(cls, 
                    file_dicts: Tuple[Dict, Dict, Dict],
                    aupe_info: AupeInfo) -> 'RGB':
        """Create an RGB object from a tuple of file dictionaries.
        :param file_dicts: Tuple of file dictionaries for red, green, and blue channels
        :type file_dicts: Tuple[Dict, Dict, Dict]
        :param aupe_info: AupeInfo object containing camera and filter information
        :type aupe_info: AupeInfo
        :return: RGB object
        :rtype: RGB
        """
        # create Img objects for each channel
        red_img = Img(file_dicts[0], aupe_info)
        green_img = Img(file_dicts[1], aupe_info)
        blue_img = Img(file_dicts[2], aupe_info)

        # create RGB object
        rgb = cls((red_img, green_img, blue_img), aupe_info)
        return rgb

    def debayer(self, 
                method: Literal[
                    'simple', 
                    'edge-aware', 
                    'variable-number-of-gradients']='edge-aware',
                to_msc: bool=False) -> Union[None, 'MSC']:
        """Perform debayering on the HRC frame, using the OpenCV debayer
        method.        
        """        
        if self.debayered:
            print('Frame is already debayered')
        else:
            raw_img = self.red.image

            method_dict = {
                'simple': cv2.COLOR_BayerBG2BGR,
                'edge-aware': cv2.COLOR_BayerBG2BGR_EA,
                'variable-number-of-gradients': cv2.COLOR_BayerBG2BGR_VNG
            }

            col_img = cv2.cvtColor(raw_img, method_dict[method])

            self.red.image = col_img[:,:,2]
            self.green.image = col_img[:,:,1]
            self.blue.image = col_img[:,:,0]
            self.rgb_image = np.stack([
                                        self.red.image, 
                                        self.green.image, 
                                        self.blue.image], 
                                        axis=2)
            self.reset_balance_vector('all')
            self.debayered = True
        
            if to_msc:
                # make a blank MSC, and put the debayered images in it
                msc_frame = MSC(
                    imgs=[self.red, self.green, self.blue],
                    aupe_info=self.aupe_info)
                print('HRC frame debayered and converted to MSC')
                # set the false colour to the hrc frame
                msc_frame.false_rgb = self
                return msc_frame
            else:
                return None
            

    # def hrc2wac_ccm(self, 
    #                  wac_frame: RGB, 
    #                  method: Literal['auto', 'manual']='auto') -> NDArray:
    #     """Find the CCM that translates the HRC dRGB colour values to the LWAC
    #     d_eRGB colour values.
    #     """        
    #     # get the calibration target patch values from the HRC image
    #     hrc_cal_targ = CalibrationTarget()
    #     hrc_drgb = self.get_image('99b') # get vals from raw image
    #     if method == 'auto':            
    #         hrc_cal_targ.find_target_outline(hrc_drgb)
    #     elif method == 'manual':
    #         hrc_cal_targ.draw_target_outline(hrc_drgb)
    #     else:
    #         raise ValueError(f"Unknown method: {method}")             
    #     drgb_image = self.get_image('bpu') # get vals from raw image
    #     hrc_ct_drgb = hrc_cal_targ.get_observed_colours(drgb_image)

    #     # get the hrc drgb -> srgb ccm
    #     hrc_ct_ref_srgb = hrc_cal_targ.patch_ref_sRGB
    #     hrc_ccm = hrc_cal_targ.compute_ccm(hrc_ct_drgb, hrc_ct_ref_srgb)

    #     # get the calibration target patch values from the RWAC image
    #     wac_cal_targ = CalibrationTarget()
    #     wac_dRGB = wac_frame.get_image('99b') # get vals from raw image
    #     if method == 'auto':            
    #         wac_cal_targ.find_target_outline(wac_dRGB)
    #     elif method == 'manual':
    #         wac_cal_targ.draw_target_outline(wac_dRGB)
    #     else:
    #         raise ValueError(f"Unknown method: {method}")
    #     # get the observed values from the calibration target
    #     wac_dRGB = wac_frame.get_image('bpu')
    #     wac_ct_drgb = wac_cal_targ.get_observed_colours(wac_dRGB)

    #     # compute the CCM from the HRC to RWAC patch values
    #     hrc2wac_ccm = wac_cal_targ.compute_ccm(hrc_ct_drgb, wac_ct_drgb)

    #     # get the wac drgb -> srgb ccm
    #     wac_ct_ref_srgb = wac_cal_targ.patch_ref_sRGB
    #     wac_ccm = wac_cal_targ.compute_ccm(wac_ct_drgb, wac_ct_ref_srgb)        

    #     # hrc ccm is then product of the hrc2wac_ccm and the wac ccm
    #     # # check error of the ccm
    #     # error = np.abs(hrc_ccm - hrc2wac_ccm @ wac_ccm)
    #     # print(error)

    #     # save the ccm to a csv file
    #     # get the output directory
    #     out_dir = Path('.', 'data', 'ccms', self.camera)
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     # save the ccm to a csv file
    #     ccm_df = pd.DataFrame(hrc2wac_ccm)
    #     # add the camera, sol, scene, trial to the filename
    #     filename = f"ccm_HRC2{wac_frame.camera}.csv"
    #     ccm_df.to_csv(Path(out_dir, filename), index=False, header=False)
    #     print(f"CCM saved to {out_dir}/{filename}")

    #     return hrc2wac_ccm
    
    # def load_hrc2wac2srgb_ccm(self, wac_frame: RGB):
    #     """Load the HRC to WAC to sRGB CCM from a csv file
    #     """
    #     # get the ccm directory
    #     ccm_dir = Path('.', 'data', 'ccms', self.camera)
    #     # check the ccm dir exists
    #     if not ccm_dir.exists():
    #         raise FileNotFoundError(f"CCM directory {ccm_dir} does not exist")
    #     # load the ccm from a csv file
    #     filename = f"ccm_HRC2{wac_frame.camera}.csv"
    #     # check the file exists
    #     if not Path(ccm_dir, filename).exists():
    #         raise FileNotFoundError(f"CCM file {filename} does not exist in {ccm_dir}")
    #     ccm_df = pd.read_csv(Path(ccm_dir, filename), header=None)
    #     hrc2wac_ccm = ccm_df.to_numpy()
    #     print(f"CCM loaded from {ccm_dir}/{filename}")

    #     # load the wac to srgb ccm
    #     wac_ccm = wac_frame.ccm

    #     # set the hrc ccm
    #     self.ccm = hrc2wac_ccm @ wac_ccm
        
class WAC_RGB(RGB):
    def __init__(self, rgb_imgs: Tuple[Img, Img, Img]):
        super().__init__(rgb_imgs)

class MSC:
    """A Class for hosting and processing MultiSpectral Cubes. Key functions
    include MSC loading and reflectance calibration.
    """    
    def __init__(self, 
                 imgs: List[Img],
                 aupe_info: AupeInfo):
        self.n_bands = len(imgs)
        self.aupe_info = aupe_info
        self.imgs = imgs # TODO change to preserve dict order
        self.filter_ids = [img.filter_id for img in imgs]
        self.cwls = np.array([img.cwl for img in self.imgs])
        self.fwhms = np.array([img.fwhm for img in self.imgs])

        self.response_functions = np.array([img.filter_response['response'] for img in self.imgs]).squeeze().T
        self.response_wavelengths = self.imgs[0].filter_response['wavelengths']

        # set the plot colour for each band, using a standard categorical colour map 
        if self.n_bands <= 10:       
            self.cwl_cols = mpl.color_sequences['tab10'][:self.n_bands]
        elif self.n_bands <= 20:
            self.cwl_cols = mpl.color_sequences['tab20'][:self.n_bands]
        else:
            raise ValueError(f"Too many bands ({self.n_bands}) for colour map")
        
        self.exposures = np.array([img.exposure for img in self.imgs])
        self.stack = np.stack([img.image for img in self.imgs], axis=2)
        self.units = 'DN'
        self.dtype = self.imgs[0].image.dtype
        self.camera = self.imgs[0].camera
        self.tag = None # optinoal tag for indicating the image feature used to perform rectification with
        self.trial = self.imgs[0].trial
        self.scene = self.imgs[0].scene
        self.sol = self.imgs[0].sol
        self.timestamp = self.imgs[0].timestamp
        self.pan = self.imgs[0].pan
        self.tilt = self.imgs[0].tilt
        self.out_dir = self.imgs[0].out_dir
        self.refl_coeffs = np.zeros(self.n_bands)
        self.refl_offsets = np.zeros(self.n_bands)
        self.false_rgb = None
        self.calibration_target = None

    @classmethod
    def from_filedicts(cls, 
                    file_dicts: List[Dict],
                    aupe_info: AupeInfo) -> 'RGB':
        """Create an RGB object from a tuple of file dictionaries.
        :param file_dicts: Tuple of file dictionaries for red, green, and blue channels
        :type file_dicts: Tuple[Dict, Dict, Dict]
        :param aupe_info: AupeInfo object containing camera and filter information
        :type aupe_info: AupeInfo
        :return: RGB object
        :rtype: RGB
        """
        # create Img objects for each channel
        imgs = [Img(path_dict, aupe_info) for path_dict in file_dicts] 

        # create MSC object
        msc = cls(imgs, aupe_info)
        return msc

    def plot_exposures(self):
        """Plot the exposures as a function of wavelength
        """    
        plt.style.use('default')
        fig, ax = plt.subplots(1,1, figsize=(8, 4))    

        ax.plot(
            self.cwls,
            self.exposures,
            marker='o',
            linestyle='-',            
            label='Exposure Time (s)'
        )
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Exposure Time (s)')
        ax.set_title(f"{self.sol} {self.scene} {self.trial} {self.camera}")

        ax = self.underplot_bands(self.exposures.max(), ax)

        return fig, ax

    def underplot_bands(self, y_val, ax: plt.Axes) -> plt.Axes:
            
        ax.bar(    
            x=self.cwls,
            height=self.exposures.max(),         
            width = self.fwhms,
            label = self.filter_ids,
            color= self.cwl_cols,
            alpha=0.5      
        )
    
        # add a legend
        ax.legend(loc='lower right', fontsize=8, title='Filter ID')
        
        return ax

    def plot_filter_responses(self):
        """Plot the response functions as a function of wavelength
        """    
        plt.style.use('default')
        fig, ax = plt.subplots(1,1, figsize=(8, 4))    

        for i, band in enumerate(self.filter_ids):
            ax.plot(
                self.response_wavelengths,
                self.response_functions[:, i],
                label=band,
                color=self.cwl_cols[i]
            )
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Response Function')
        ax.set_title(f"{self.sol} {self.scene} {self.trial} {self.camera}")
        ax.legend(loc='upper right', fontsize=8, title='Filter ID')

        return fig, ax

    def flat_field(self):
        pass

    def bias_subtract(self):
        pass

    def exposure_correct(self):
        """Apply exposure correction to each band of the image stack
        and update the units and dtype of the image stack.
        """  
        if self.units == 'DN/s':
            print('Frame has already been exposure corrected')
        else:
            for band in self.imgs:
                band.exposure_correct()
            self.units = 'DN/s'
            self.dtype = np.float32
            # update the image stack
            self.stack = np.stack([band.image for band in self.imgs], axis=2)    
    
    # special HRC methods
    def reflectance_from_camera(self, frame: 'MSC'):
        """Find the transform matrices that convert the reflectance coefficients
        from the given camera frame to this camera frame. Export these
        matrices to the data directory.

        :param frame: MSC frame containing the reflectance coefficients
        :type frame: MSC
        """        
        # check that both frames have reflectance coefficients
        if (self.refl_coeffs == np.zeros(3)).all() and (frame.refl_coeffs == np.zeros(3)).all():
            raise ValueError("Both frames must have reflectance coefficients and offsets set")
        
        # check that both frames have the same number of bands
        if self.n_bands != frame.n_bands:
            raise ValueError(f"Both frames must have the same number of bands ({self.n_bands} != {frame.n_bands})")
                
        m_r = colour.matrix_colour_correction(frame.refl_coeffs, self.refl_coeffs)
        o_r = colour.matrix_colour_correction(frame.refl_offsets, self.refl_offsets)
        # save the matrices to a csv file
     
        # get the input data directory
        self.aupe_info.data_dir
        # save the matrices as csv files
        filepath = self.aupe_info.data_dir / f"{frame.camera}_2_{self.camera}_reflectance_coeffs.csv"        
        np.savetxt(filepath, m_r, delimiter=",")
        print(f"{frame.camera}-2-{self.camera} Reflectance transfer coefficients saved to {filepath}")
        filepath = self.aupe_info.data_dir / f"{frame.camera}_2_{self.camera}_reflectance_offset.csv"
        np.savetxt(filepath, o_r, delimiter=",")
        print(f"{frame.camera}-2-{self.camera} Reflectance transfer offsets saved to {filepath}")

    
    def load_reflectance_coefficients_from_transfer(self,
                            frame: 'MSC')-> None:
        """Load the reflectance transfer coefficients from a csv file,
        and apply this to the reflectance coefficients and offsets of the given
        frame.

        :param frame: MSC frame of the source camera type
        :type frame: MSC
        """
        # get the file
        self.aupe_info.data_dir
        
        # read the reflectance coefficients transfer matrix
        filepath = self.aupe_info.data_dir / f"{frame.camera}_2_{self.camera}_reflectance_coeffs.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Reflectance transfer coefficients file {filepath} does not exist")
        m_r = np.genfromtxt(filepath, delimiter=',')

        # read the reflectance offsets transfer matrix
        filepath = self.aupe_info.data_dir / f"{frame.camera}_2_{self.camera}_reflectance_offset.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Reflectance transfer offsets file {filepath} does not exist")
        o_r = np.genfromtxt(filepath, delimiter=',')

        # apply the transfer coefficients to the reflectance coefficients and offsets to the frame
        refl_coeffs = m_r @ frame.refl_coeffs
        refl_offsets = o_r @ frame.refl_offsets
        # update the reflectance coefficients and offsets
        self.update_reflectance_coefficients(
            refl_coeffs=refl_coeffs.tolist(),
            refl_offsets=refl_offsets.tolist()
        )     

        # we also copy over the ccm to the false rgb image   
        if self.false_rgb is not None:
            # preferentially load the ccm of this object
            ccm_path = self.aupe_info.data_dir / f"{self.camera}_reflectance2srgb_ccm.csv"
            if ccm_path.exists():
                self.false_rgb.load_ccm(ccm_path)
            else:
                # if the ccm file does not exist, use the destination frame's ccm
                self.false_rgb.ccm = frame.false_rgb.ccm
            # update the false rgb image
            self.false_rgb.red.refl_coeff = self.refl_coeffs[0]
            self.false_rgb.green.refl_coeff = self.refl_coeffs[1]
            self.false_rgb.blue.refl_coeff = self.refl_coeffs[2]
            self.false_rgb.red.refl_offset = self.refl_offsets[0]
            self.false_rgb.green.refl_offset = self.refl_offsets[1]
            self.false_rgb.blue.refl_offset = self.refl_offsets[2]

    def update_reflectance_coefficients(self,
        refl_coeffs: List[float],
        refl_offsets: List[float]) -> None:
        """Update the reflectance coefficients and offsets for each band in the image stack.
        :param refl_coeffs: list of reflectance coefficients for each band
        :type refl_coeffs: List[float]
        :param refl_offsets: list of reflectance offsets for each band
        :type refl_offsets: List[float]
        :param filter_ids: list of filter IDs for each band
        :type filter_ids: List[str]
        """
        self.refl_coeffs = np.array(refl_coeffs)
        self.refl_offsets = np.array(refl_offsets)
        # update the reflectance coefficients in each band
        for i, band in enumerate(self.imgs):
            band.refl_coeff = self.refl_coeffs[i]
            band.refl_offset = self.refl_offsets[i]
        

    def apply_reflectance_calibration(self):
        """Apply reflectance calibration to the image stack.
        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99s']
        """
        if self.units == 'Reflectance':
            print('Frame has already been reflectance calibrated')
            return
        elif self.units != 'DN/s':
            raise ValueError("Image stack must be exposure corrected before reflectance calibration")        

        for band in self.imgs:
            band.apply_reflectance_calibration()

        # update the stack
        self.stack = np.stack([band.image for band in self.imgs], axis=2)
        # update the units and dtype
        self.units = 'Reflectance'
        self.dtype = np.float32

        # if false_rgb is set, update it
        if self.false_rgb is not None:  
            # check the stack has been updated
            self.false_rgb.rgb_image = np.stack([
                                            self.false_rgb.red.image, 
                                            self.false_rgb.green.image,
                                            self.false_rgb.blue.image], axis=2)
            self.false_rgb.units = 'Reflectance'
            self.false_rgb.dtype = np.float32

            # updating stretch coefficients
            for method, balance_vector in self.false_rgb.balance_vector.items():
                # only update the coefficients if they have been set
                if (balance_vector != np.zeros(3)).all():
                    self.false_rgb.extract_balance_vector(method)
    
    def export_reflectance_coefficients(self):
        """Export the reflectance coefficients to a csv file.
        :return: path to the exported csv file
        :rtype: Path
        """
        coeffs_df = pd.DataFrame({
            'filter_id': self.filter_ids,
            'reflectance_coefficient': self.refl_coeffs,
            'reflectance_offset': self.refl_offsets
        })
        # add the camera, sol, scene, trial to the filename
        filter_id_str = ''.join(self.filter_ids)
        filename = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_{filter_id_str}_refl_coeffs.csv"
        coeffs_df.to_csv(self.out_dir / filename, index=False)
        print(f"Reflectance coefficients saved to {self.out_dir}/{filename}")
        
        return self.out_dir / filename
                
    def set_false_color(self, bands: Dict):
        """Set the false color RGB image from the specified bands.
        :param bands: tuple of band filter IDs to use for the false color image
        :type bands: Tuple[str, str, str]
        """
        # map filter ids to imgs
        band_dict = dict(zip(self.filter_ids, self.imgs))
        false_colour_rgb = WAC_RGB(
                                    (band_dict[bands['R']], 
                                    band_dict[bands['G']],
                                    band_dict[bands['B']]))
        self.false_rgb = false_colour_rgb     
        # set the output directory to RGB
        self.false_rgb.out_dir = self.out_dir / 'RGB'           
        self.false_rgb.out_dir.mkdir(parents=True, exist_ok=True)

    def export_2_envi(self):
        """Export the Reflectance calibrated image stack to an ENVI file format.
        :return: path to the exported ENVI file
        :rtype: Path
        """   

        # save the reflectance coefficients to a csv file
        self.export_reflectance_coefficients()
        filter_id_str = ''.join(self.filter_ids)
        if self.tag is not None:
            envi_path = Path(self.out_dir, f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_{filter_id_str}_refl_{self.tag}.hdr")
        else:
            envi_path = Path(self.out_dir, f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_{filter_id_str}_refl.hdr")

        envi.save_image(
            str(envi_path.resolve()),
            self.stack,
            dtype=self.dtype,
            force=True,
            ext='.img',
            metadata={
                'acquisition time': f"{self.timestamp}", # might need reformatting to ENVI compatible format
                'band names': self.filter_ids,
                'bands': self.n_bands,
                'data type': 4,
                'fwhm': self.fwhms.tolist(),
                'wavelength': self.cwls.tolist(),
                'wavelength units': 'nm'
            }
        )
    
    def export_2_gif(self):
        """Export the image stack to a gif file, showing the filter IDs and
        centre wavelengths.
        """
        images = []
        filter_id_str = ''.join(self.filter_ids)
        if self.tag is not None:
            gif_path = Path(self.out_dir, f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_{filter_id_str}_refl_{self.tag}.gif")
        else:
            gif_path = Path(self.out_dir, f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_{filter_id_str}_refl.gif")
        for img in self.imgs:
            print(f'Exporting image {img.filter_id} ')
            img_slice = Image.fromarray(np.uint8(img.get_image('raw')*255))
            # add text to the image
            img_slice = img_slice.convert("RGBA")
            text = f"{img.filter_id}: {img.cwl} nm"
            draw = ImageDraw.Draw(img_slice)
            font = ImageFont.load_default(size=24)
            draw.text((0, 0), text, fill=(255, 255, 0, 255), font=font)
            img_slice = img_slice.convert("RGB")
            images.append(img_slice)
        if images:        
            images[0].save(str(gif_path.resolve()), save_all=True, append_images=images[1:], duration=500, loop=0)

class StereoTools:
    """A class for stereo tools, such as camera calibration and stereo rectification.
    """
    def __init__(self, 
                 src: NDArray,
                 dst: NDArray) -> None:
        """Initialize the StereoTools class with source and destination frames.   

        :param src: Source image as a NumPy array.
        :type src: NDArray
        :param dst: Destination image as a NumPy array.
        :type dst: NDArray
        """
        self.src = src
        self.dst = dst
        self.src_mask = np.ones(self.src.shape[:2], dtype=np.uint8)*255
        self.dst_mask = np.ones(self.dst.shape[:2], dtype=np.uint8)*255
        self.pts_src = np.empty(1)
        self.pts_dst = np.empty(1)
        self.homography = np.zeros((3, 3))

        # show the two frame that will be used for rectification
        plt.style.use('default')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.src, vmin=0, vmax=1, interpolation='none')
        ax[0].set_title('Source Frame')
        ax[1].imshow(self.dst, vmin=0, vmax=1, interpolation='none')
        ax[1].set_title('Destination Frame')
        plt.tight_layout()
        plt.show()

    def cvtFrame2cv2(self):
        """Convert the source and destination frames to OpenCV format images.
        """
        src_img_uint = (np.clip(self.src,0,1) * 255).astype('uint8')
        if self.src.shape[-1] == 3:
            self.src = cv2.cvtColor(src_img_uint, cv2.COLOR_RGB2BGR)
        else:
            self.src = src_img_uint
        
        dst_img_uint = (np.clip(self.dst,0,1) * 255).astype('uint8')
        if self.dst.shape[-1] == 3:
            self.dst = cv2.cvtColor(dst_img_uint, cv2.COLOR_RGB2BGR)
        else:
            self.dst = dst_img_uint 

    def select_match_regions(self) -> None:
        """Select matching regions in the source and destination images.
        This is a placeholder for manual selection of matching regions.
        """               
        if not isinstance(self.src.dtype, float):
            self.cvtFrame2cv2() 

        prompt = "Select Src. Region for Optimal Rectification"
        src_box = cv2.selectROI(prompt, self.src)
        cv2.destroyWindow(prompt)

        # set pixels outside of box to NaN
        src_mask = np.zeros(self.src.shape[:2], dtype=np.uint8)
        src_mask[src_box[1]:src_box[1]+src_box[3], src_box[0]:src_box[0]+src_box[2]] = 255
        self.src_mask = src_mask

        prompt = "Select same Region in Dst."
        dst_box = cv2.selectROI(prompt, self.dst)
        cv2.destroyWindow(prompt)

        # set pixels outside of box to NaN
        dst_mask = np.zeros(self.dst.shape[:2], dtype=np.uint8)
        dst_mask[dst_box[1]:dst_box[1]+dst_box[3], dst_box[0]:dst_box[0]+dst_box[2]] = 255
        self.dst_mask = dst_mask

    def findMatches(self, show: bool=False) -> None:
        """Find matches between the source and destination images using SIFT.

        :param show: _description_, defaults to False
        :type show: bool, optional
        """
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        # histogram equalise using opencv the self.src image
        src_yuv = cv2.cvtColor(self.src, cv2.COLOR_BGR2YUV)
        dst_yuv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        src_yuv[:,:,0] = cv2.equalizeHist(src_yuv[:,:,0])
        dst_yuv[:,:,0] = cv2.equalizeHist(dst_yuv[:,:,0])

        # convert the YUV image back to RGB format
        src_eq = cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR)
        dst_eq = cv2.cvtColor(dst_yuv, cv2.COLOR_YUV2BGR)

        # find the keypoints and descriptors with SIFT
        try:
            # apply mask
            src_img = cv2.bitwise_and(src_eq, src_eq, mask=self.src_mask)
            kp_src, des_src = sift.detectAndCompute(src_img, None)
        except cv2.error as e:
            print(f"Error in SIFT detection for source image: {e}")
            print("Try running cvtFrame2cv2() to convert the images to OpenCV format.")
            return
        try:
            dst_img = cv2.bitwise_and(dst_eq, dst_eq, mask=self.dst_mask)
            kp_dst, des_dst = sift.detectAndCompute(dst_img,None)
        except cv2.error as e:
            print(f"Error in SIFT detection for destination image: {e}")
            print("Try running cvtFrame2cv2() to convert the images to OpenCV format.")
            return
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_src,des_dst,k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        if show:
            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(
                            cv2.cvtColor(src_eq, cv2.COLOR_RGB2BGR), kp_src,
                            cv2.cvtColor(dst_eq, cv2.COLOR_RGB2BGR), kp_dst,
                            good,None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            plt.imshow(img3),plt.show()

        print(f"Found {len(good)} good matches between source and destination regions.")
        
        pts_src = []
        pts_dst = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                pts_dst.append(kp_dst[m.trainIdx].pt)
                pts_src.append(kp_src[m.queryIdx].pt)

        pts_src = np.int32(pts_src)
        pts_dst = np.int32(pts_dst) 

        # save the points
        self.pts_src = pts_src
        self.pts_dst = pts_dst

    def findWarp(self, show: bool=False) -> None:
        """Find the homography between the source and destination images.
        """
        if self.src.dtype == float:
            self.cvtFrame2cv2()

        self.findMatches(show)

        h, mask = cv2.findHomography(self.pts_src, self.pts_dst, cv2.RANSAC)

        self.homography = h
        # Refine pts to select only inliers
        self.pts_src = self.pts_src[mask.ravel()==1]
        self.pts_dst = self.pts_dst[mask.ravel()==1]

        if show:
            # apply the homography
            wrp = cv2.warpPerspective(
                self.src, 
                self.homography, 
                (self.dst.shape[1], self.dst.shape[0])
            )
            # convert the src image to grayscale
            src_gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
            dst_gray = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
            wrp_gray = cv2.cvtColor(wrp, cv2.COLOR_BGR2GRAY)
            # make an RGB image from the source and destination pre-warp
            original = np.zeros_like(self.dst)
            original[:,:,0] = src_gray
            original[:,:,1] = dst_gray
            original[:,:,2] = dst_gray
            # make an RGB image with Red as source and blue and green as wrp
            warped = np.zeros_like(self.dst)
            warped[:,:,0] = wrp_gray
            warped[:,:,1] = dst_gray
            warped[:,:,2] = dst_gray
            # show the combined image
            plt.style.use('default')
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(original)
            ax[0].set_title('Src (Red) and Dst (Cyan) Images')
            ax[1].imshow(warped)
            ax[1].set_title('Wrp (Red) and Dst (Cyan) Images')
            plt.tight_layout()
            plt.show()

    def applyWarp(self, 
                  src_frame: Union[Img, RGB, HRC, WAC_RGB, MSC], 
                  dst_frame: Union[Img, RGB, HRC, WAC_RGB, MSC],
                  wrp_frame: Union[Img, RGB, HRC, WAC_RGB, MSC],
                  tag: str):
        """Apply the homography to the source frame to warp it to the 
        destination frame. Put the warped frame and unwarped destination
        frame into the wrp_frame. Add a tag describing the
        image feature used to perform rectification with, e.g. 'target', 'cliff'

        :param src_frame: Source frame to be warped
        :type src_frame: Union[Img, RGB, HRC, WAC_RGB, MSC]
        :param dst_frame: Destination frame to be warped to
        :type dst_frame: Union[Img, RGB, HRC, WAC_RGB, MSC]
        :param wrp_frame: Frame to put the warped source and unwarped destination
            frames into
        :type wrp_frame: Union[Img, RGB, HRC, WAC_RGB, MSC]
        :param tag: Tag describing the image feature used to perform rectification with
        :type tag: str
        :return: The warped frame with the source and destination images
            applied to it.
        :rtype: Union[Img, RGB, HRC, WAC_RGB, MSC]
        """
        if isinstance(src_frame, Img):
            src_mat = src_frame.image
        elif isinstance(src_frame, (RGB, HRC, WAC_RGB)):
            src_mat = src_frame.rgb_image
        elif isinstance(src_frame, MSC):
            src_mat = src_frame.stack

        if isinstance(dst_frame, Img):
            dst_mat = src_frame.image
        elif isinstance(dst_frame, (RGB, HRC, WAC_RGB)):
            dst_mat = dst_frame.rgb_image
        elif isinstance(dst_frame, MSC):
            dst_mat = dst_frame.stack

        # apply the homography
        wrp_mat = cv2.warpPerspective(
            src_mat, 
            self.homography, 
            (dst_mat.shape[1], dst_mat.shape[0])
        )

        # apply the masking of the warped image to the destination image
        mask_map = wrp_mat[:,:,0] == 0
        # where the mask is true, set the destination image to 0
        dst_mat[mask_map] = 0

        if isinstance(wrp_frame, Img):
            wrp_frame.image = wrp_mat
            wrp_frame.units = src_frame.units
            wrp_frame.dtype = src_frame.dtype
            wrp_frame.reset_stretch_coefficient('all')


        elif isinstance(wrp_frame, (RGB, HRC, WAC_RGB)):
            wrp_frame.rgb_image = wrp_mat
            # update the red, green and blue channels
            wrp_frame.red.image = wrp_mat[:,:,0]
            wrp_frame.green.image = wrp_mat[:,:,1]
            wrp_frame.blue.image = wrp_mat[:,:,2]
            # copy across the ccm
            wrp_frame.ccm = src_frame.ccm
            # update units and dtype
            wrp_frame.units = src_frame.units
            wrp_frame.dtype = src_frame.dtype
            wrp_frame.red.units = src_frame.red.units
            wrp_frame.green.units = src_frame.green.units
            wrp_frame.blue.units = src_frame.blue.units
            wrp_frame.reset_balance_vector('all')      

        elif isinstance(wrp_frame, MSC):
            # check that the filter ids of the src are in the warp frame
            wrp_filter_ids = wrp_frame.filter_ids
            src_filter_ids = src_frame.filter_ids
            if not all(band in wrp_filter_ids for band in src_filter_ids):
                raise ValueError("Source frame filter IDs are not in Warp frame")

            
            # make sure the stack dtype is float32
            wrp_frame.stack = np.zeros(
                (wrp_mat.shape[0], wrp_mat.shape[1], len(wrp_filter_ids)), 
                dtype=np.float32
            )

            # copy the warp source images into the source bands of the warp frame
            for i, band in enumerate(src_filter_ids):
                wrp_band_idx = wrp_filter_ids.index(band)
                # update both stack and Img's
                wrp_frame.stack[:,:,wrp_band_idx] = wrp_mat[:,:,i]              
                wrp_frame.imgs[wrp_band_idx].image = wrp_mat[:,:,i]
                # update warp frame properties to match the source frame
                wrp_frame.imgs[wrp_band_idx].units = src_frame.imgs[i].units
                wrp_frame.imgs[wrp_band_idx].dtype = src_frame.imgs[i].dtype
                wrp_frame.imgs[wrp_band_idx].refl_coeff = src_frame.imgs[i].refl_coeff
                wrp_frame.imgs[wrp_band_idx].refl_offset = src_frame.imgs[i].refl_offset
                wrp_frame.refl_coeffs[wrp_band_idx] = src_frame.imgs[i].refl_coeff
                wrp_frame.refl_offsets[wrp_band_idx] = src_frame.imgs[i].refl_offset
                wrp_frame.imgs[wrp_band_idx].reset_stretch_coefficient('all')
            
            # check that the filter ids of the destination are in the warp frame
            dst_filter_ids = dst_frame.filter_ids
            if all(band in wrp_filter_ids for band in dst_filter_ids):               
                # copy the warp destination images into the destination bands of the warp frame    
                for i, band in enumerate(dst_filter_ids):
                    wrp_band_idx = wrp_filter_ids.index(band)
                    # update both stack and Img's
                    wrp_frame.stack[:,:,wrp_band_idx] = dst_mat[:,:,i]
                    wrp_frame.imgs[wrp_band_idx].image = dst_mat[:,:,i]
                    # update warp frame properties to match the destination frame
                    wrp_frame.imgs[wrp_band_idx].units = dst_frame.imgs[i].units
                    wrp_frame.imgs[wrp_band_idx].dtype = dst_frame.imgs[i].dtype
                    wrp_frame.imgs[wrp_band_idx].refl_coeff = dst_frame.imgs[i].refl_coeff
                    wrp_frame.imgs[wrp_band_idx].refl_offset = dst_frame.imgs[i].refl_offset
                    wrp_frame.imgs[wrp_band_idx].reset_stretch_coefficient('all')
                    wrp_frame.refl_coeffs[wrp_band_idx] = dst_frame.imgs[i].refl_coeff
                    wrp_frame.refl_offsets[wrp_band_idx] = dst_frame.imgs[i].refl_offset
            
            if src_frame.camera == 'LRWAC' and dst_frame.camera == 'HRC':
                wrp_frame.camera = 'LRWHRC'
            elif src_frame.camera == 'LWAC' and dst_frame.camera == 'RWAC':
                wrp_frame.camera = 'LRWAC'
            wrp_frame.units = wrp_frame.imgs[0].units
            wrp_frame.dtype = wrp_frame.imgs[0].dtype
                
        wrp_frame.tag = tag
        # add the tag to the output directory
        wrp_frame.out_dir = wrp_frame.out_dir / tag
        # make sure the output directory exists
        wrp_frame.out_dir.mkdir(parents=True, exist_ok=True)

    def depth_map(self):
        """Generate a depth map from the source and destination images.
        TODO
        Current iteration just finds a basic disparity map, that is noisy.
        """
        if not isinstance(self.src, np.ndarray):
            self.cvtFrame2cv2()
        if not isinstance(self.dst, np.ndarray):
            self.cvtFrame2cv2()

        # if RGB images, convert to grayscale
        if len(self.src.shape) == 3 and self.src.shape[2] == 3:
            src = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        else:
            src = self.src
        
        if len(self.dst.shape) == 3 and self.dst.shape[2] == 3:
            dst = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
        else:
            dst = self.dst

        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=5)
        disparity = stereo.compute(src, dst)

        # where -16 set to 0
        disparity[disparity < 0] = 0

        # filter out sal and pepper noise
        disparity = cv2.medianBlur(disparity, 5)

        plt.imshow(disparity,'viridis')
        plt.colorbar()
        plt.title('Disparity Map')
        plt.show()
        return disparity