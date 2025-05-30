# A Python library for processing and analysing images from the
# Aberystwyth University PanCam Emulator, AUPE.
#
# Roger Stabbins
# Natural History Museum, London
# 9/5/2025

# from copy import deepcopy
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from roipoly import RoiPoly
from scipy.stats import linregress
import scipy.optimize
from scipy.interpolate import interp1d
from spectral import envi
import PIL.Image

import colour
from colour.characterisation import CCS_COLOURCHECKERS
from colour_checker_detection import detect_colour_checkers_inference
from colour_checker_detection.detection.common import sample_colour_checker, as_int32_array

LEVEL_DICT = {
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

class AupeInfo:
    """A class to hold the AUPE information for a given dataset.
    This includes the filter positions, filter ids, cwl and fwhm.
    """    
    def __init__(self, filepath: Path):
        """Holds AUPE information not included in the 
        image metadata, namely mapping filter positions and ids to
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
        # read the filepath csv file into the object
        # expect header lines of version and date
        header = pd.read_csv(filepath, nrows=2, usecols=[0,1], index_col=0)
        self.aupe_info_version = header.loc['version'].values[0]
        self.aupe_info_date = header.loc['date'].values[0]
        # read the data
        aupe_info = pd.read_csv(filepath, index_col=0, header=3)
        self.filter_pos = aupe_info.index.to_list()
        self.filter_id = aupe_info['filter_id'].to_dict()
        self.cwl = aupe_info['cwl'].to_dict()
        self.fwhm = aupe_info['fwhm'].to_dict()

        # cam number -> camera does not typically change between AUPE versions.
        self.cam_dict = {
                2: 'HRC',
                0: 'LWAC',
                1: 'RWAC'}
        
        # self.load_flat_fields() # TODO
        # self.load_bias_frames() # TODO
        
    def inverse_filter_id(self):
        """Invert the filter id dictionary to get the filter id from the filter
        position.
        """
        # invert the filter id dictionary
        inv_filter_id = {v: k for k, v in self.filter_id.items()}
        return inv_filter_id
    
    def inverse_cwl(self):
        """Invert the cwl dictionary to get the cwl from the filter position.
        """
        # invert the cwl dictionary
        inv_cwl = {v: k for k, v in self.cwl.items()}
        return inv_cwl
    
    def inverse_fwhm(self):
        """Invert the fwhm dictionary to get the fwhm from the filter position.
        """
        # invert the fwhm dictionary
        inv_fwhm = {v: k for k, v in self.fwhm.items()}
        return inv_fwhm
    
    def filter_ids2pos(self, 
                    filter_ids: List[str]) -> List[str]:
        """Convert the filter ids to filter positions
        """
        # convert the filter ids to filter positions
        filter_pos_lut = self.inverse_filter_id()
        filter_pos = [filter_pos_lut[filter_id] for filter_id in filter_ids]
        return filter_pos

    def set_filter_ids(self, 
                    camera: Literal['HRC', 'LWAC', 'RWAC', 'LRWAC'],
                    frame_type: Literal['RGB', 'MSC']) -> List[str]:
        """Set the filter ids for the given camera and frame type.
        """
        # set the filter ids to use according to the camera and frame type
        filter_ids = []
        if camera == 'HRC':
            if frame_type == 'RGB':
                filter_ids = ['HR0', 'HR0', 'HR0'] # initialise with same filter id
            elif frame_type == 'Single':
                filter_ids = ['HR0'] # just load the raw HRC frame
            elif frame_type == 'MSC':
                filter_ids = ['HR0', 'HR0', 'HR0'] # treat HRC as a multispectral imager
            else:
                raise ValueError(f"Unknown frame type {frame_type} for HRC camera")
        elif camera == 'LWAC':
            if frame_type == 'RGB':
                filter_ids = ['L1R', 'L2G', 'L3B']
            elif frame_type == 'MSC':
                filter_ids = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']
        elif camera == 'RWAC':
            if frame_type == 'RGB':
                filter_ids = ['R1R', 'R2G', 'R3B']
            elif frame_type == 'MSC':
                filter_ids = ['G07', 'G08', 'G09', 'G10', 'G11', 'G12']
        elif camera == 'LRWAC':
            if frame_type == 'MSC':
                filter_ids = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06',
                              'G07', 'G08', 'G09', 'G10', 'G11', 'G12']
        else:
            raise ValueError(f"Unknown camera {camera}")
            # TODO - add support for NavCams
        
        return filter_ids
    
class AupeIO:
    '''Class for loading an AUPE image from a given directory, or sol, scene,
    trial (optional) specification, for a given camera and frame type.

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
    '''
    def __init__(self, 
                 camera: Literal['HRC', 'LWAC', 'RWAC'],
                 frame_type: Literal['Single', 'RGB', 'MSC'],
                 sol: str,
                 scene: str, 
                 trial: str='',
                 filter_ids: List[str]=[''], # optionally specify the filter_ids to use (note - not filter_pos codes)
                 campaign_dir: Path=Path('..','data'),
                 aupe_info_path: Path=Path('.','data','aupe_info.csv')):
        
        self.camera = camera
        self.frame_type = frame_type

        self.campaign_dir = campaign_dir
        self.sol = sol
        self.scene = scene

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

        # set the list of filters to load for given camera and frame type
        self.aupe_info = AupeInfo(aupe_info_path)
        if filter_ids[0] != '':
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
        
        self.filter_ids = filter_ids # e.g. 'G01', 'G02', 'L1R', etc
        self.filter_pos = self.aupe_info.filter_ids2pos(filter_ids) # e.g. 'LWAC1', 'LWAC2', etc
        
        # initialise the lists of input image filepaths
        self.input_files = []

        # get all the images in the directory
        png_files = list(self.scene_dir.glob("*.png"))

        # grab the files that match the filter pos codes
        for filter_pos in self.filter_pos: # note order preserved
            # get the files that match the filter pos code
            filter_pos_files = [path for path in png_files if filter_pos+'_' in path.name]
            # add the files to the input files list
            self.input_files += filter_pos_files

    def load_frame(self):
        """Load the frame from the input files, and return the frame object.
        """        

        # if there are no files, skip
        if len(self.input_files) == 0:
            print(f"No files found for {self.camera} {self.frame_type} {self.sol} {self.scene} {self.trial}")
            return None

        input_file_dicts = self.file_dicts()     

        if self.frame_type == 'Single':
            if len(input_file_dicts) > 1:
                raise ValueError(f"Multiple files found for single frame type: {input_file_dicts}")
            frame = Img(input_file_dicts[0], self.aupe_info)
            return frame
        
        elif self.frame_type == 'RGB':
            if self.camera == 'HRC':
                frame = HRC(input_file_dicts, self.aupe_info)
                return frame
            elif self.camera == 'LWAC' or self.camera == 'RWAC':
                frame = WAC_RGB.from_filedicts(input_file_dicts, self.aupe_info)
                return frame
        
        elif self.frame_type == 'MSC':
            frame = MSC(input_file_dicts, self.aupe_info)
            return frame
        else:
            raise ValueError(f"Unknown frame type {self.frame_type}")
    
    def file_dicts(self) -> List:
        """For the given filepath, return a dictionary giving:
        - full file path
        - file name
        - filter id
        - sol
        - scene
        - trial
        - output directory

        :param filepath: file path to the image
        :type filepath: Path
        :return: File information needed to process the image
        :rtype: Dict
        """        
        input_file_dicts = []
        for i, input_file in enumerate(self.input_files):
            file_dict = {}
            file_dict['filepath'] = input_file
            file_dict['filter_id'] = self.filter_ids[i]
            file_dict['trial'] = self.trial
            file_dict['scene'] = self.scene
            file_dict['sol'] = self.sol
            file_dict['out_dir'] = self.out_dir
            input_file_dicts.append(file_dict)
        
        return input_file_dicts

class CalibrationTarget:
    """Calibration Target Class for hosting reference and observed calibration
    target patch values, and the colour correction matrix.
    """    
    def __init__(self,
                illuminant: Literal[
                     'A', 'B', 'C', 
                     'D50', 'D55', 'D65', 'D75', 
                     'ICC D50']='ICC D50',
                colour_checker: Literal[
                    'ColorChecker24 - After November 2014',
                    'ColorChecker 1976',
                    "ColorChecker 2005",
                    "BabelColor Average",
                    "ColorChecker24 - Before November 2014",
                    "ColorChecker24 - After November 2014",
                    "ColorCheckerSG - Before November 2014",
                    "ColorCheckerSG - After November 2014",
                    "TE226 V2"]='ColorChecker24 - After November 2014',
                ) -> None:
        """Initialise the calibration target class

        :param illuminant: Illuminant to use for the calibration target
        :type illuminant: Literal[
            'A', 'B', 'C',
            'D50', 'D55', 'D65', 'D75',
            'ICC D50']
        :param colour_checker: Colour checker to use for the calibration target
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
        :rtype: None
        """        
        # reference values
        self.illuminant = illuminant
        self.colour_checker = colour_checker
        self.patch_ref_xyY = self.load_ref_vals('xyY', illuminant, colour_checker)
        self.patch_ref_XYZ = self.load_ref_vals('XYZ', illuminant, colour_checker)
        self.patch_ref_sRGB = self.load_ref_vals('sRGB', illuminant, colour_checker)
        self.patch_names = list(CCS_COLOURCHECKERS['ColorChecker24 - After November 2014'].data.keys())
        self.rows = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014'].rows
        self.cols = CCS_COLOURCHECKERS['ColorChecker24 - After November 2014'].columns
        self.patch_ref_refl = self.load_spectral_data()  # high resolution spectral reflectance data
        # observed values
        self.target_outline = np.zeros((4,2))
        self.patch_rois = None  # TODO figure out how to get patch rois in the original image - should be way to invert via target_outline 
        self.ccm = np.zeros((3,3))
        self.patch_obs_drgb = None # do we need this?
        self.patch_obs_srgb = None  # do we need this?   
        # TODO
        self.patch_obs_refl = None  # np.ndarray
        
    def load_ref_vals(self, 
                space: Literal['xyY', 'XYZ', 'sRGB'],
                illuminant: Literal[
                    'A', 'B', 'C', 'D50', 'D55', 'D65', 'D75', 'ICC D50'
                ]='ICC D50',
                colour_checker: Literal[
                    'ColorChecker24 - After November 2014',
                    'ColorChecker 1976',
                    "ColorChecker 2005",
                    "BabelColor Average",
                    "ColorChecker24 - Before November 2014",
                    "ColorChecker24 - After November 2014",
                    "ColorCheckerSG - Before November 2014",
                    "ColorCheckerSG - After November 2014",
                    "TE226 V2"]='ColorChecker24 - After November 2014'
                ) -> NDArray:
        """Load reference values for the calibration target patches
        via the colour science python library.
        
        :param space: Colour space to use for the reference values
        :type space: Literal['xyY', 'XYZ', 'sRGB']
        :rtype: NDArray
        """     
        # load reference values from the colour science library   
        ref_ct = CCS_COLOURCHECKERS[colour_checker]

        # get xyY values
        ref_ct_xyY = list(ref_ct.data.values())
        if space == 'xyY':
            return ref_ct_xyY
        
        # get XYZ values
        ref_ct_XYZ = colour.xyY_to_XYZ(ref_ct_xyY)
        if space == 'XYZ':
            return ref_ct_XYZ

        # update the illuminant
        illuminant_ccs = colour.CCS_ILLUMINANTS[
            "CIE 1931 2 Degree Standard Observer"][illuminant]

        # get sRGB values
        ref_ct_RGB = colour.XYZ_to_sRGB(ref_ct_XYZ, illuminant_ccs, apply_cctf_encoding=False)
        if space == 'sRGB':
            return ref_ct_RGB
        else:
            raise ValueError(f"Unknown space {space} for reference colourchecker data")

    def load_spectral_data(self):
        """Load the high resolution spectral reflectance data for the 
        calibration target patches.

        Note this assumes that the patch columns match the order of the 
        colour checker patches in the colour science library.

        Note that the reflectance values are given in percentage (0 - 100), so 
        we convert them to reflectance values (0 - 1) by dividing by 100.
        """        
        filepath = Path('.', 'data', 'colorchecker_spectra.csv')
        # read the csv file into a pandas dataframe
        spectral_data = pd.read_csv(filepath, index_col=0, header=1)
        # convert the dataframe to a dictionary
        patch_ref_refl = {}
        patch_ref_refl['wavelengths'] = spectral_data.index.to_numpy()
        patch_ref_refl['reflectance'] = np.clip(spectral_data.to_numpy() / 100, 0,1) # convert to reflectance values, and clamp to 0-1 range
    
        return patch_ref_refl

    def sample_patch_ref_refl(self,
                              frame) -> NDArray:
        """Sample the reference patch reflectance values, given by the reference
        csv file, with the transmission profiles of the bands of the given frame
        (must be an MSC frame).
        
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
        # i.e. R[cwl] = sum_lambda(R[lambda] * T_cwl[lambda]) / sum_lambda(T_cwl[lambda])
        patch_refl_vals = np.divide(
                    np.matmul(frame.response_functions.T, patch_refl).T,
                    np.sum(frame.response_functions, axis=0))
        
        # store in a list
        patch_refl_vals = [patch_refl_vals[:,i] for i, filter_id in enumerate(frame.filter_ids)]
        # convert the list to an NDArray
        patch_refl_vals = np.array(patch_refl_vals)
        # TODO check this format matches the observed value format
        return patch_refl_vals

    def get_observed_vals(self, 
                          image: NDArray,
                          show: bool=False) -> NDArray:
        """Extract the patch values from the frame, and return the observed values

        :param frame: _description_
        :type frame: Literal['Single';, 'RGB', 'MSC']
        :return: An array giving the values of each patch in the image for 
        each channel of the frame
        :rtype: NDArray
        """        

        # get the approximate width and height of the calibration target in pixels
        q = self.target_outline
        width = np.abs(q[0][0] - q[3][0]).astype(np.int32)
        height = np.abs(q[0][1] - q[1][1]).astype(np.int32)

        print(f"Width: {width} Height: {height}")

        samples = int(np.floor(np.sqrt(0.2*(width * height)//24)))

        rectangle = as_int32_array([
                            [0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]])

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
        
        # draw where the patches are on the sampled calibration target
        if show:
             # Using the additional data to plot the colour checker and masks.
            masks_i = np.zeros(patch_data.colour_checker.shape)
            for i, mask in enumerate(patch_data.swatch_masks):
                masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
            
            colour.plotting.plot_image(
                colour.cctf_encoding(
                    np.clip(patch_data.colour_checker + masks_i * 0.25, 0, 1)));

        return patch_data.swatch_colours
    
    def Gauss(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def get_observed_vals_stack(self,
                                frame,
                                method: Literal['mean', 'gauss-fit'],
                                show: bool=False) -> Tuple[NDArray, NDArray]:
        """Extract the patch values from each band of an MSC stack.

        :param stack: The image stack containing the calibration target
        :type stack: NDArray
        :return: An array giving the values of each patch in the image for
        each channel of the frame
        :rtype: NDArray
        """
        # get the approximate width and height of the calibration target in pixels
        q = self.target_outline
        width = np.abs(q[0][0] - q[3][0]).astype(np.int32)
        height = np.abs(q[0][1] - q[1][1]).astype(np.int32)

        print(f"Target Width: {width} Height: {height}")
        samples = int(np.floor(np.sqrt(0.2*(width * height)//24)))
        rectangle = as_int32_array([
                            [0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]])
        
        # we use the colour detection library to sample the patches.
        # Note that we set the reference values to None, as we don't want
        # the algorithm to check the orientation of the patches, as the
        # frame we are using might not be an approximate of the colour checker
        # colours. The orientation should have been determined in the
        # find_calibration_target method.
        obs_vals = []
        obs_ave = []
        obs_std = []
        for band in frame.imgs:
            cwl = band.cwl
            # TODO replace this with a method that gives control of the sampling
            # e.g. perform statistics on the ROI of each patch.
            patch_data = sample_colour_checker(
                            band.image, 
                            self.target_outline, 
                            rectangle, 
                            samples,
                            working_width=width,
                            working_height=height,
                            reference_values=None)
            
            obs_vals.append(patch_data.swatch_colours)

            swatch_ave = []
            swatch_std = []
            if show:
                fig, ax = plt.subplots(self.rows, self.cols, sharey=True, figsize=(self.cols, self.rows))
            for p, mask in enumerate(patch_data.swatch_masks):
                # get the pixel values covered by the mask in the patch+data.colour_checker
                # range of colour checker
                ct_max = patch_data.colour_checker.max()
                patch = patch_data.colour_checker[mask[0]:mask[1], mask[2]:mask[3], ...]
                # implement diferent methods for evaluating the patch value
                ave = np.mean(patch, axis=(0,1))
                std = np.std(patch, axis=(0,1))
                if method == 'mean':
                    swatch_ave.append(ave)
                    swatch_std.append(std)
                
                if method == 'gauss-fit':
                    ydata, xdata = np.histogram(
                                                patch, 
                                                bins=patch.size//2, 
                                                range=(ave-4*std, ave+4*std)
                                                )
                    
                    parameters, covariance = scipy.optimize.curve_fit(
                                                    self.Gauss, 
                                                    xdata[:-1],
                                                    ydata,
                                                    [10, ave, std]
                                                    )

                    fit_A = parameters[0]
                    fit_B = parameters[1]
                    fit_C = parameters[2]

                    swatch_ave.append(fit_B)
                    swatch_std.append(fit_C)

                    if show:
                        fit_y = self.Gauss(xdata[:-1], fit_A, fit_B, fit_C)
                        r = p // len(patch) # current column                    
                        c = p % len(patch) # current row
                        col = colour.cctf_encoding(np.clip(self.patch_ref_sRGB[p], 0,1))
                        # alpha=1.0
                        alpha = (fit_B / ct_max)**2
                        ax[r][c].plot(xdata[:-1], ydata, 'o', c=col,label='data', alpha=alpha)
                        ax[r][c].plot(xdata[:-1], fit_y, '-', c=col, label='fit', alpha=alpha)
                        # remove axis ticks
                        ax[r][c].set_axis_off()
                        # show mean
                        ax[r][c].axvline(fit_B, ls='--')
                        ax[r][c].axvline(ave, ls='-.')
                        ax[r][c].set_title(f"{self.patch_names[p]}", c=col, fontsize='x-small')

            if show:
                fig.suptitle(f"{band.filter_id} {cwl} nm Gauss-Fits")
                fig.tight_layout()

            obs_ave.append(np.array(swatch_ave))
            obs_std.append(np.array(swatch_std))

        # convert the lists to ndarrays
        obs_ave = np.array(obs_ave)
        obs_std = np.array(obs_std)

        return obs_ave, obs_std

    def calibrate_reflectance(self,
                            frame,
                            method: Literal['mean', 'gauss-fit'] = 'gauss-fit',
                            show: bool=False) -> None:
        """Calibrate the reflectance values of the patches in each band
        of the frame, setting the reflectance correction coefficients
        for each band of the frame.

        :param frame: The MSC frame to calibrate the reflectance values for
        :type frame: MSC
        :param show: plot intermediary steps, defaults to False
        :type show: bool, optional
        """        

        # get the observed values for each band of the frame        
        obs_ave, obs_std = self.get_observed_vals_stack(frame, method, show)
        # get the reference values for each band of the frame
        ref_refl = self.sample_patch_ref_refl(frame)
        # compute the reflectance correction coefficients for each band of the frame
        refl_coeffs = []
        refl_offsets = []

        if show:
            ncols=3
            nrows=frame.n_bands//ncols
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,sharey=True, figsize=(2*3, 2*frame.n_bands//3))            
        for b, band in enumerate(frame.imgs):
            x = obs_ave[b,:]
            y = ref_refl[b,:]
            result = linregress(x, y)
            # store the slope and intercept of the regression line
            refl_coeffs.append(result.slope)
            refl_offsets.append(result.intercept)
            # TODO propagate uncertainties of the regression line
            if show:                
                c = b % ncols # corrent column
                r = b // ncols # current row
                if nrows == 1:
                    this_ax = ax[c]
                else:
                    this_ax = ax[r][c]
                col = colour.cctf_encoding(np.clip(self.patch_ref_sRGB, 0,1))
                this_ax.scatter(x, y, c=col)
                this_ax.plot(x, result.intercept + result.slope*x, 'r')
                # set x label to obs band
                this_ax.set_xlabel(f'Power {frame.units}')
                # set y label to ref band
                this_ax.set_ylabel(f'Reflectance')                

                this_ax.set_title(f'{band.filter_id} {band.cwl} nm')
                # # remove ticks
                # this_ax.set_xticklabels([])
                # this_ax.set_yticklabels([])
        if show:
            fig.suptitle(f'Reflectance Calibration for {frame.camera} {frame.sol} {frame.scene} {frame.trial}')
            fig.tight_layout()

        # set the reflectance correction coefficients and offsets for each band of the frame
        for b, band in enumerate(frame.imgs):
            band.refl_coeff = refl_coeffs[b]
            band.refl_offset = refl_offsets[b]
            print(f"{band.filter_id} {band.cwl} nm: coeff={refl_coeffs[b]:0.3}1/DN/s, offset={refl_offsets[b]:0.3}")

    def compute_ccm(self, 
                    observed_vals: NDArray,
                    reference_vals: NDArray) -> NDArray:
        """Compute the colour correction matrix for the calibration target,
        from the given observed and reference values.

        :param observed_vals: Array of observed values for each patch
        :type observed_vals: NDArray
        :param reference_vals: Array of reference values for each patch
        :type reference_vals: NDArray
        :return: 3x3 Colour correction matrix
        :rtype: NDArray
        """     
        # TODO make checks on the observed and reference values arrays
        ccm = colour.matrix_colour_correction(observed_vals, reference_vals)
        self.ccm = ccm
        return ccm

    def find_target_outline(self, rgb_image) -> bool:
        """Automatically find the Calibration Target 
        using the colour checker detection algorithm

        :param image: The image containing the calibration target
        :type image: NDArray
        :return: True if the target was found, False otherwise
        :rtype: bool
        """
        # run the colour checker detection algorithm
        # decoded_image = colour.cctf_decoding(rgb_image)
        decoded_image = rgb_image

        # this algorithm finds the colour checker values of the image supplied.
        # We want to get the patch locations though, so that we can draw
        # them on other images - e.g. we find the patch locations in an RGB
        # image, and then draw them on the multispectral image.
        print("Searching for colour checker...")
        colour_checker_data = detect_colour_checkers_inference(
                                    decoded_image, 
                                    additional_data=True)
        
        # check if the run was successful
        if colour_checker_data == ():
            print("No colour checker found")
            print("Searching for colour checker in cropped image...")
            # if the first search fails, we try to find it again in a smaller subset of the image
            cropped_image = decoded_image[150:-150, 150:-150]
            colour_checker_data = detect_colour_checkers_inference(
                                    cropped_image, 
                                    additional_data=True)
            if colour_checker_data == ():
                # # if this also fails, we resort to drawing the colour checker manually
                # print('Manually draw out the calibration target quadrilateral')
                # result = self.draw_target_outline(rgb_image)
                # if result is False:
                print("No calibration target found")
                return False
            else:
                print(f"Found {len(colour_checker_data)} colour checkers")
                # get the first one, as we only expect one
                colour_checker_data = colour_checker_data[0]
                # from this we get the quadrilateral that contains the calibration target
                # we need to offset the by the image reduction of 150 pixels
                colour_checker_data.quadrilateral[:, 0] += 150
                colour_checker_data.quadrilateral[:, 1] += 150
                self.target_outline = colour_checker_data.quadrilateral
        else:   
            print(f"Found {len(colour_checker_data)} colour checkers")
            # get the first one, as we only expect one
            colour_checker_data = colour_checker_data[0]
            # from this we get the quadrilateral that contains the calibration target
            self.target_outline = colour_checker_data.quadrilateral
        
        # this is all the information we need to extract the patch values down the line, if we repurpose the code provided
        # in the colour checker detection library

        return True

    def draw_target_outline(self, image: NDArray) -> bool:
        """Draw the target outline manually on the RGB image using
        the roipoly library.
        :param image: The image containing the calibration target
        :type image: NDArray
        :return: True if the target was found, False otherwise
        :rtype: bool
        """   

        # TODO predraw the approximate area using OpenCV ROI select,
        # so that we can zoom in on the target before drawing the more precise
        # polyroi outline of the target.

        ct_roi = cv2.selectROI("Select Calibration Target ROI", np.flip(image, 2))
        cv2.destroyWindow("Select Calibration Target ROI")
    
        default_backend = mpl.get_backend()
        mpl.use('Qt5Agg')  # need this backend for RoiPoly to work 
        
        fig = plt.figure(figsize=(10,10), dpi=80)

        # switch order of roi to (y, x, h, w)
        ct_roi = (ct_roi[1], ct_roi[0], ct_roi[3], ct_roi[2])           
        
        ct_img = image[ct_roi[0]:ct_roi[0]+ct_roi[2], ct_roi[1]:ct_roi[1]+ct_roi[3]]

        if len(ct_img) == 0:
            print("No ROI selected")
            plt.close()
            mpl.use(default_backend)  # reset backend
            return False     

        plt.imshow(ct_img, origin='upper')
        plt.title(f'Draw quadrilateral around the calibration target')

        my_roi = RoiPoly(fig=fig) # draw new ROI in red color
        plt.close()
        mpl.use(default_backend)  # reset backend

        # Get the coords for the ROIs
        # offset the coords by the ROI location
        quad_roi_x = [x + ct_roi[1] for x in my_roi.x]
        quad_roi_y = [y + ct_roi[0] for y in my_roi.y]
        points = np.array([quad_roi_x, quad_roi_y]).T[0:4]

        if len(points) != 4:
            return False
        else:            
            self.target_outline = points
            return True

    def show_target_outline(self, image: NDArray):
        """Show the target quadrilateral on the image

        :param image: The image containing the calibration target
        :type image: NDArray
        """
        # draw the quadrilateral on the image
        annotated_image = image.copy()
        points = np.int32(self.target_outline)

        # check the format of the image, and convert uint8 if neccesary
        if annotated_image.dtype != np.uint8:
            annotated_image = (annotated_image * 255).astype(np.uint8)

        annotated_image = cv2.polylines(annotated_image, 
                                        [points], 
                                        isClosed=True, 
                                        color=(255, 0, 0), 
                                        thickness=2)
        # show the image
        plt.imshow(annotated_image)
        plt.show()

    def extract_ccm(self, frame, show: bool=False):
        drgb_image = frame.get_image('bpu') # get vals from raw image
        obs_ct_dRGB_vals = self.get_observed_vals(drgb_image, show=show)
        # get the reference values
        ref_ct_sRGB_vals = self.patch_ref_sRGB
        # compute the colour correction matrix
        ccm = self.compute_ccm(obs_ct_dRGB_vals, ref_ct_sRGB_vals)
        frame.ccm = ccm

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
            fig, ax = plt.subplots(1, 1)            
            cols = np.clip(self.patch_ref_sRGB, 0,1)
            for patch in range(len(self.patch_names)):
                ax.scatter(
                        self.patch_ref_sRGB[patch,0], 
                        cor_ct_sRGB_vals[patch,0], 
                        color=cols[patch,:].flatten(),
                        edgecolor='red'
                        )
                ax.scatter(
                        self.patch_ref_sRGB[patch,1], 
                        cor_ct_sRGB_vals[patch,1], 
                        color=cols[patch,:].flatten(),
                        edgecolor='green'
                        )
                ax.scatter(
                        self.patch_ref_sRGB[patch,2], 
                        cor_ct_sRGB_vals[patch,2], 
                        color=cols[patch,:].flatten(),
                        edgecolor='blue')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('Reference sRGB')
            ax.set_ylabel('Corrected sRGB')
            # set square axis
            ax.set_aspect('equal', adjustable='box')
            # set title
            ax.set_title('Corrected vs Reference sRGB')
            # set x and y limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        return ccm

    def find_target_and_compute_ccm(self, 
                    frame, 
                    method: Literal['auto', 'manual']='auto',
                    show: bool=False) -> NDArray:
        """Find the calibration target in the given frame, and compute the
        colour correction matrix from the observed values and reference values.

        :param frame: The image containing the calibration target
        :type frame: RGB
        :return: 3x3 Colour correction matrix
        :rtype: NDArray
        """
        # find the calibration target in the image
        approx_balance_rgb = frame.get_image('99b')
        if method == 'manual':
            # draw the target outline manually
            result = self.draw_target_outline(approx_balance_rgb)
        elif method == 'auto':
            result = self.find_target_outline(approx_balance_rgb)
        else:
            raise ValueError(f"Unknown method {method} for finding calibration target")

        if result is False:
            print("No calibration target found")
            return False
            # print("Please draw the calibration target outline manually")
            # result = self.draw_target_outline(approx_balance_rgb)

        if show:
            self.show_target_outline(approx_balance_rgb)

        ccm = self.extract_ccm(frame, show=show)
  
        return True

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
        # read the metadata from the image file using the PIL exif reader
        img = PIL.Image.open(self.filepath)
        metadata = img.info

        self.pan = float(metadata['AU_pan'])
        self.tilt = float(metadata['AU_tilt'])
        self.exposure = float(metadata['AU_exposureTime'])
        self.timestamp = metadata['AU_timestampUTC']        
        self.camera = aupe_info.cam_dict[int(metadata['AU_camNum'])]

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
            raise FileNotFoundError(f"Filter response file {response_file} does not exist")
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
        print(f"Stretching image using {LEVEL_DICT[stretch_method]}")
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
        self.balance_vector = {
            'raw': np.zeros(3),
            'bpu': np.zeros(3),
            'bpb': np.zeros(3),
            '99b': np.zeros(3),
            '99u': np.zeros(3),
            'wps': np.zeros(3),
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

        # get the ccm directory
        proc_dir = self.out_dir.parent.parent.parent.parent.parent
        ccm_dir = Path(proc_dir, sol, scene, trial, camera, 'RGB')
        # check the ccm dir exists
        if not ccm_dir.exists():
            raise FileNotFoundError(f"CCM directory {ccm_dir} does not exist")
        # load the ccm from a csv file
        filename = f"{sol}_{scene}_{trial}_{camera}_ccm.csv"
        # check the file exists
        if not Path(ccm_dir, filename).exists():
            raise FileNotFoundError(f"CCM file {filename} does not exist in {ccm_dir}")
        ccm_df = pd.read_csv(Path(ccm_dir, filename), header=None)
        self.ccm = ccm_df.to_numpy()
        print(f"CCM loaded from {ccm_dir}/ccm.csv")

    def export_ccm(self):
        """Export the colour correction matrix to a csv file
        """        
        # save the ccm to a csv file
        ccm_df = pd.DataFrame(self.ccm)
        # add the camera, sol, scene, trial to the filename
        # TODO figure if there is any other metadata we can apply - e.g. in shade, in sun, indoors etc.
        filename = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_ccm.csv"
        ccm_df.to_csv(Path(self.out_dir, filename), index=False, header=False)
        print(f"CCM saved to {self.out_dir}/{filename}")

    def apply_ccm(self):
        """Apply the colour correction matrix to the RGB image
        """
        drgb_image  = self.get_image('bpu')
        # check if the ccm is set
        if self.ccm is None:
            print("No colour correction matrix set")
            # search for the latest calibration target in the 
        srgb_image = colour.apply_matrix_colour_correction(drgb_image, self.ccm)

        # check if the iamge range is 0 - 1.
        # # apply 0 - 1 normalisation and clipping
        # srgb_image = srgb_image.astype(np.float64) / np.max(srgb_image)
        # apply encoding
        # srgb_image = colour.cctf_encoding(srgb_image)
        srgb_image = np.clip(srgb_image, 0.0, 1.0)

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
            raise ValueError(f"Unknown stretch method: {method}")

        self.balance_vector[method] = np.array([r_stretch, g_stretch, b_stretch])

    def reset_balance_vector(self, 
                    method: Literal['all', 'raw', 'bpb', 'bpu', 'wps', '99b', '99u']='all'):
        """Reset the balance vector to all ones
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
        elif method == 'raw':
            self.balance_vector[method] = np.ones(3)
        else:
            self.balance_vector[method] = np.zeros(3)            

    def apply_balance_vector(self, colour_correction: Literal['raw', 'bpb', 'bps', 'wps', '99p']='raw'):

        print(f"Stretching image using {LEVEL_DICT[colour_correction]}")
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
        title = f"{self.sol} {self.scene} {self.trial} {self.camera} RGB ({LEVEL_DICT[colour_correction]})"

        disp_img = self.get_image(colour_correction)
        
        # # apply encoding
        # disp_img = colour.cctf_encoding(disp_img)

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
                     show: bool=False):
        """Export the image to an 8-bit RGB image file, using the stretch method

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        """   
        
        title = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_RGB_{colour_correction}.png"

        print(f"Exporting image using {LEVEL_DICT[colour_correction]}")
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
            fig, ax = plt.subplots(1,1, figsize=(8, 4))  
            ax.imshow(disp_img, vmin=0, vmax=255, interpolation='none')
            ax.axis('off')
            # set the title
            fig.suptitle(title)
    
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
        
        # relabel the channels of the hrc r,g,b images, and update other
        # attributes
        self.red.channel = 'HRCR'
        self.red.cwl = aupe_info.cwl['HRCR']
        self.red.fwhm = aupe_info.fwhm['HRCR']  
        self.red.filter_id = aupe_info.filter_id['HRCR']
        self.green.channel = 'HRCG'
        self.green.cwl = aupe_info.cwl['HRCG']
        self.green.fwhm = aupe_info.fwhm['HRCG']
        self.green.filter_id = aupe_info.filter_id['HRCG']
        self.blue.channel = 'HRCB'
        self.blue.cwl = aupe_info.cwl['HRCB']
        self.blue.fwhm = aupe_info.fwhm['HRCB']
        self.blue.filter_id = aupe_info.filter_id['HRCB']

    def debayer(self, 
                method: Literal[
                    'simple', 
                    'edge-aware', 
                    'variable-number-of-gradients']='edge-aware'):
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

    def hrc2wac_ccm(self, 
                     wac_frame: RGB, 
                     method: Literal['auto', 'manual']='auto') -> NDArray:
        """Find the CCM that translates the HRC dRGB colour values to the LWAC
        d_eRGB colour values.
        """        
        # get the calibration target patch values from the HRC image
        hrc_cal_targ = CalibrationTarget()
        hrc_drgb = self.get_image('99b') # get vals from raw image
        if method == 'auto':            
            hrc_cal_targ.find_target_outline(hrc_drgb)
        elif method == 'manual':
            hrc_cal_targ.draw_target_outline(hrc_drgb)
        else:
            raise ValueError(f"Unknown method: {method}")             
        drgb_image = self.get_image('bpu') # get vals from raw image
        hrc_ct_drgb = hrc_cal_targ.get_observed_vals(drgb_image)

        # get the hrc drgb -> srgb ccm
        hrc_ct_ref_srgb = hrc_cal_targ.patch_ref_sRGB
        hrc_ccm = hrc_cal_targ.compute_ccm(hrc_ct_drgb, hrc_ct_ref_srgb)

        # get the calibration target patch values from the RWAC image
        wac_cal_targ = CalibrationTarget()
        wac_dRGB = wac_frame.get_image('99b') # get vals from raw image
        if method == 'auto':            
            wac_cal_targ.find_target_outline(wac_dRGB)
        elif method == 'manual':
            wac_cal_targ.draw_target_outline(wac_dRGB)
        else:
            raise ValueError(f"Unknown method: {method}")
        # get the observed values from the calibration target
        wac_dRGB = wac_frame.get_image('bpu')
        wac_ct_drgb = wac_cal_targ.get_observed_vals(wac_dRGB)

        # compute the CCM from the HRC to RWAC patch values
        hrc2wac_ccm = wac_cal_targ.compute_ccm(hrc_ct_drgb, wac_ct_drgb)

        # get the wac drgb -> srgb ccm
        wac_ct_ref_srgb = wac_cal_targ.patch_ref_sRGB
        wac_ccm = wac_cal_targ.compute_ccm(wac_ct_drgb, wac_ct_ref_srgb)        

        # hrc ccm is then product of the hrc2wac_ccm and the wac ccm
        # # check error of the ccm
        # error = np.abs(hrc_ccm - hrc2wac_ccm @ wac_ccm)
        # print(error)

        # save the ccm to a csv file
        # get the output directory
        out_dir = Path('.', 'data', 'ccms', self.camera)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save the ccm to a csv file
        ccm_df = pd.DataFrame(hrc2wac_ccm)
        # add the camera, sol, scene, trial to the filename
        filename = f"ccm_HRC2{wac_frame.camera}.csv"
        ccm_df.to_csv(Path(out_dir, filename), index=False, header=False)
        print(f"CCM saved to {out_dir}/{filename}")

        return hrc2wac_ccm
    
    def load_hrc2wac2srgb_ccm(self, wac_frame: RGB):
        """Load the HRC to WAC to sRGB CCM from a csv file
        """
        # get the ccm directory
        ccm_dir = Path('.', 'data', 'ccms', self.camera)
        # check the ccm dir exists
        if not ccm_dir.exists():
            raise FileNotFoundError(f"CCM directory {ccm_dir} does not exist")
        # load the ccm from a csv file
        filename = f"ccm_HRC2{wac_frame.camera}.csv"
        # check the file exists
        if not Path(ccm_dir, filename).exists():
            raise FileNotFoundError(f"CCM file {filename} does not exist in {ccm_dir}")
        ccm_df = pd.read_csv(Path(ccm_dir, filename), header=None)
        hrc2wac_ccm = ccm_df.to_numpy()
        print(f"CCM loaded from {ccm_dir}/{filename}")

        # load the wac to srgb ccm
        wac_ccm = wac_frame.ccm

        # set the hrc ccm
        self.ccm = hrc2wac_ccm @ wac_ccm
        
class WAC_RGB(RGB):
    def __init__(self, rgb_imgs: Tuple[Img, Img, Img]):
        super().__init__(rgb_imgs)

class MSC:
    """A Class for hosting and processing MultiSpectral Cubes. Key functions
    include MSC loading and reflectance calibration.
    """    
    def __init__(self, 
                 msc_path_dicts: List[Dict],
                 aupe_info: AupeInfo):
        self.n_bands = len(msc_path_dicts)
        self.aupe_info = aupe_info
        self.imgs = [Img(path_dict, aupe_info) for path_dict in msc_path_dicts] # TODO change to preserve dict order
        self.filter_ids = [path_dict['filter_id'] for path_dict in msc_path_dicts]
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
        self.trial = self.imgs[0].trial
        self.scene = self.imgs[0].scene
        self.sol = self.imgs[0].sol
        self.timestamp = self.imgs[0].timestamp
        self.pan = self.imgs[0].pan
        self.tilt = self.imgs[0].tilt
        self.out_dir = self.imgs[0].out_dir
        self.refl_coeffs = np.zeros(self.n_bands)
        self.refl_offset = np.zeros(self.n_bands)
        self.false_rgb = None
        self.calibration_target = None

    def plot_exposures(self):
        """Plot the exposures as a function of wavelength
        """    
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
            'reflectance_offset': self.refl_offset
        })
        # add the camera, sol, scene, trial to the filename
        filename = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_refl_coeffs.csv"
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

    def export_2_envi(self):
        """Export the Reflectance calibrated image stack to an ENVI file format.
        :return: path to the exported ENVI file
        :rtype: Path
        """   

        # save the reflectance coefficients to a csv file
        self.export_reflectance_coefficients()

        envi.save_image(
            str(self.out_dir / f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_refl.hdr"),
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


class StereoTools:
    """A class for stereo tools, such as camera calibration and stereo rectification.
    """
    def __init__(self, 
                 src_frame: Union[Img, RGB, HRC, WAC_RGB],
                 dst_frame: Union[Img, RGB, HRC, WAC_RGB]) -> None:
        """Initialize the StereoTools class with source and destination frames.   
        """
        self.src = src_frame
        self.dst = dst_frame
        self.pts_src = np.empty(1)
        self.pts_dst = np.empty(1)
        self.homography = np.zeros((3, 3))

    def cvtFrame2cv2(self):
        """Convert the source and destination frames to OpenCV format images.
        """
        src_img = self.src.get_image('99u')
        src_img_uint = (np.clip(src_img,0,1) * 255).astype('uint8')
        if isinstance(self.src, RGB) or isinstance(self.src, HRC) or isinstance(self.src, WAC_RGB):
            self.src = cv2.cvtColor(src_img_uint, cv2.COLOR_RGBA2BGR)
        else:
            self.src = src_img_uint

        # prepare the images        
        dst_img = self.dst.get_image('99u')
        dst_img_uint = (np.clip(dst_img,0,1) * 255).astype('uint8')
        if isinstance(self.dst, RGB) or isinstance(self.dst, HRC) or isinstance(self.dst, WAC_RGB):
            self.dst = cv2.cvtColor(dst_img_uint, cv2.COLOR_RGBA2BGR)
        else:
            self.dst = dst_img_uint            

    def findMatches(self, show: bool=False) -> None:
        """Find matches between the source and destination images using SIFT.

        :param show: _description_, defaults to False
        :type show: bool, optional
        """
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        try:
            kp_src, des_src = sift.detectAndCompute(self.src,None)
        except cv2.error as e:
            print(f"Error in SIFT detection for source image: {e}")
            print("Try running cvtFrame2cv2() to convert the images to OpenCV format.")
            return
        try:
            kp_dst, des_dst = sift.detectAndCompute(self.dst,None)
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
            if m.distance < 0.55*n.distance:
                good.append([m])
        
        if show:
            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(
                            self.src,kp_src,
                            self.dst,kp_dst,
                            good,None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            plt.imshow(img3),plt.show()
        
        pts_src = []
        pts_dst = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.55*n.distance:
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
        if not isinstance(self.src, np.ndarray):
            self.cvtFrame2cv2()
        self.findMatches(show)

        h, mask = cv2.findHomography(self.pts_src, self.pts_dst, cv2.RANSAC)

        self.homography = h
        # Refine pts to select only inliers
        self.pts_src = self.pts_src[mask.ravel()==1]
        self.pts_dst = self.pts_dst[mask.ravel()==1]


    def applyWarp(self, 
                  src_frame: Union[Img, RGB, HRC, WAC_RGB, MSC], 
                  dst_frame: Union[Img, RGB, HRC, WAC_RGB, MSC],
                  wrp_frame: Union[Img, RGB, HRC, WAC_RGB, MSC]) -> np.ndarray:
        """Apply the homography to the source frame to warp it to the destination frame.
        The destination frame should be a prepared src-bands +  dst-bands MSC.

        Placeholder -- we assume the destination frame is an all-band MSC.
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

        if isinstance(wrp_frame, Img):
            wrp_frame.image = wrp_mat

        elif isinstance(wrp_frame, (RGB, HRC, WAC_RGB)):
            wrp_frame.rgb_image = wrp_mat
            # update the red, green and blue channels
            wrp_frame.red.image = wrp_mat[:,:,0]
            wrp_frame.green.image = wrp_mat[:,:,1]
            wrp_frame.blue.image = wrp_mat[:,:,2]

        elif isinstance(wrp_frame, MSC):
            # check that the filter ids of the src are in the warp frame
            wrp_filter_ids = wrp_frame.filter_ids
            src_filter_ids = src_frame.filter_ids
            if not all(band in wrp_filter_ids for band in src_filter_ids):
                raise ValueError("Source frame filter IDs are not in Warp frame")
            # check that the filter ids of the destination are in the warp frame
            dst_filter_ids = dst_frame.filter_ids
            if not all(band in wrp_filter_ids for band in dst_filter_ids):
                raise ValueError("Destination frame filter IDs are not in Warp frame")
            
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
                wrp_frame.refl_offset[wrp_band_idx] = src_frame.imgs[i].refl_offset
                wrp_frame.imgs[wrp_band_idx].reset_stretch_coefficient('all')

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
                wrp_frame.refl_offset[wrp_band_idx] = dst_frame.imgs[i].refl_offset
            
            wrp_frame.camera = 'LRWAC'
                
        return wrp_frame

    