# A Python library for processing and analysing images from the
# Aberystwyth University PanCam Emulator, AUPE.
#
# Roger Stabbins
# Natural History Museum, London
# 9/5/2025

from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from roipoly import RoiPoly
import scipy
import PIL.Image

import colour
from colour.characterisation import CCS_COLOURCHECKERS
from colour_checker_detection import detect_colour_checkers_inference
from colour_checker_detection.detection.common import sample_colour_checker, as_int32_array

LEVEL_DICT = {
    'raw': 'raw image no stretch',
    'bps': 'brightest pixel stretch',
    '99p': '99th percentile stretch',
    'wps': 'white patch stretch',
    'ccm': 'colour correction matrix',
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
        self.aupe_info_version = None
        self.aupe_info_date = None
        self.filter_pos = None
        self.filter_id = None
        self.cwl = None
        self.fwhm = None
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
                    camera: Literal['HRC', 'LWAC', 'RWAC'],
                    frame_type: Literal['RGB', 'MSC']) -> List[str]:
        """Set the filter ids for the given camera and frame type.
        """
        # set the filter ids to use according to the camera and frame type
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
        else:
            raise ValueError(f"Unknown camera {camera}")
            # TODO - add support for NavCams
        
        return filter_ids
    
class AupeIO:
    '''Class for loading an AUPE image from a given directory, or sol, scene,
    trial (optional) specification, for a given camera and frame type.
    '''
    def __init__(self, 
                 camera: Literal['HRC', 'LWAC', 'RWAC'],
                 frame_type: Literal['RGB', 'MSC'],
                 sol: str,
                 scene: str, 
                 trial: str=None,
                 filter_ids: List[str]=None, # optionally specify the filter_ids to use (note - not filter_pos codes)
                 campaign_dir: Path=Path('..','..','data'),
                 aupe_info_path: Path=Path('..','data','aupe_info.csv')):
        
        self.camera = camera
        self.frame_type = frame_type

        self.aupe_info = AupeInfo(aupe_info_path)

        # set the list of filters to load for given camera and frame type
        if filter_ids is not None:
            # check if the given filter ids are valid
            for filter_id in filter_ids:
                if filter_id not in self.aupe_info.filter_id.values():
                    raise ValueError(f"Filter id {filter_id} not found in aupe info")
        elif frame_type == 'Single':
            # if frame type is single, then a filter id is required
            raise ValueError(f"Filter id required for single frame type")
        else:
            # otherwise, get the filter ids and positions for the given camera and frame type
            filter_ids = self.aupe_info.set_filter_ids(camera, frame_type)
        self.filter_ids = filter_ids
        self.filter_pos = self.aupe_info.filter_ids2pos(filter_ids)

        # set the input data directory
        self.campaign_dir = campaign_dir
        if not self.campaign_dir.exists():
            raise FileNotFoundError(f"Directory {self.campaign_dir} does not exist")
        self.sol = sol
        self.scene = scene
        if trial is not None:
            self.scene_dir = Path(campaign_dir, sol, scene, trial)
        else:
            self.scene_dir = Path(campaign_dir, sol, scene)
            trial = 'Trial1'
        self.trial = trial
        # check if the input directory exists
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Directory {self.scene_dir} does not exist")

        # set the output data directory
        self.out_dir = Path(self.campaign_dir,'..', 'processed', 
                                self.sol, 
                                self.scene, 
                                self.trial, 
                                self.camera,
                                self.frame_type)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # initialise the lists of input image filepaths
        self.input_files = []

        # grab the files that match the filter pos codes
        # get all the images in the directory
        png_files = list(self.scene_dir.glob("*.png"))

        for filter_pos in self.filter_pos: # note this should preserve order from specification in set_filter_ids
            # get the files that match the filter pos code
            filter_pos_files = [path for path in png_files if filter_pos in path.name]
            # add the files to the input files list
            self.input_files += filter_pos_files

    def load_frame(self):

        # if there are no files, skip
        if len(self.input_files) == 0:
            print(f"No files found for {self.camera} {self.frame_type} {self.sol} {self.scene} {self.trial}")
            return None

        input_file_dicts = self.file_dicts()        

        if self.frame_type == 'Single':
            if len(input_file_dicts) > 1:
                raise ValueError(f"Multiple files found for single frame type: {input_file_dicts}")
            frame = Img(input_file_dicts[0], self.aupe_info)
        elif self.frame_type == 'RGB':
            if self.camera == 'HRC':
                frame = HRC(input_file_dicts, self.aupe_info)
            elif self.camera == 'LWAC' or self.camera == 'RWAC':
                frame = WAC_RGB(input_file_dicts, self.aupe_info)
        elif self.frame_type == 'MSC':
            # TODO - add support for MSC
            # frame = MSC(input_file_dicts, self.aupe_info)
            pass
        else:
            raise ValueError(f"Unknown frame type {self.frame_type}")
        
        return frame
    
    def file_dicts(self) -> List:
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
        input_file_dicts = []
        for input_file in self.input_files:
            file_dict = {}
            file_dict['filepath'] = input_file
            file_dict['trial'] = self.trial
            file_dict['scene'] = self.scene
            file_dict['sol'] = self.sol
            file_dict['out_dir'] = self.out_dir
            input_file_dicts.append(file_dict)
        
        return input_file_dicts

class CalibrationTarget:
    def __init__(self):
        self.target_outline = None  # np.ndarray
        self.patch_ref_srgb = None  # np.ndarray
        self.patch_ref_xyY = None  # np.ndarray
        self.patch_ref_XYZ = None
        self.patch_ref_refl = None  # np.ndarray
        self.patch_obs_drgb = None
        self.patch_obs_srgb = None  # np.ndarray
        #
        self.ccm = None
        # TODO
        self.patch_names = []
        self.patch_reflectance = None  # np.ndarray
        self.patch_srgb = None  # pd.DataFrame
        self.patch_rois = None  # np.ndarray        
        
    def load_spectral_data(self):
        filepath = Path('..', 'data', 'colorchecker_spectra.csv')
        # read the csv file into a pandas dataframe
        cal_targ_df = pd.read_csv(filepath, index_col=0)

    def load_reference_vals(self, 
                format: Literal['xyY', 'XYZ', 'sRGB'],
                illuminant: Literal[
                    'A', 'B', 'C', 'D50', 'D55', 'D65', 'D75', 'ICC D50'
                ]='ICC D50') -> np.array:
        """Load reference values for the calibration target patches
        via the colour science python library.
        
        :param format: Colour space to use for the reference values
        :type format: Literal['xyY', 'XYZ', 'sRGB']
        :rtype: np.array
        """        

        # TODO allow for specification of illuminant - 
        # e.g. if cal targ is in shade or direct light.

        ref_ct = CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]

        # get xyY values
        ref_ct_xyY = list(ref_ct.data.values())
        self.patch_ref_xyY = ref_ct_xyY
        if format == 'xyY':
            return ref_ct_xyY
        
        # get XYZ values
        ref_ct_XYZ = colour.xyY_to_XYZ(ref_ct_xyY)
        self.patch_ref_XYZ = ref_ct_XYZ
        if format == 'XYZ':
            return ref_ct_XYZ

        # update the illuminant
        illuminant_ccs = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][illuminant]

        # get sRGB values
        ref_ct_RGB = colour.XYZ_to_RGB(
                               ref_ct_XYZ,
                                "sRGB",
                                illuminant_ccs)
        self.patch_ref_srgb = ref_ct_RGB
        if format == 'sRGB':
            return ref_ct_RGB
        else:
            raise ValueError(f"Unknown format {format} for reference colourchecker data")

    def draw_target_outline(self, image):
        """Draw the target outline manually on the RGB image using
        the roipoly library.
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


    def show_target_outline(self, image):
        """Show the target quadrilateral on the image

        :param image: The image containing the calibration target
        :type image: np.array
        """
        # draw the quadrilateral on the image
        annotated_image = image.copy()
        points = np.int32(self.target_outline)
        annotated_image = cv2.polylines(annotated_image, 
                                        [points], 
                                        isClosed=True, 
                                        color=(255, 0, 0), 
                                        thickness=2)
        # show the image
        plt.imshow(annotated_image)
        plt.show()

    def get_observed_vals(self, frame: Literal['Single', 'RGB', 'MSC']) -> np.array:
        """Extract the patch values from the frame, and return the observed values

        :param frame: _description_
        :type frame: Literal['Single';, 'RGB', 'MSC']
        :return: An array giving the values of each patch in the image for 
        each channel of the frame
        :rtype: np.array
        """        

        # get the approximate width and height of the calibration target in pixels
        q = self.target_outline
        width = np.abs(q[0][0] - q[3][0]).astype(np.int32)
        height = np.abs(q[0][1] - q[1][1]).astype(np.int32)

        print(f"Width: {width} Height: {height}")

        samples = int(np.floor(np.sqrt(0.5*(width * height)//24)))

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
                            frame, 
                            self.target_outline, 
                            rectangle, 
                            samples,
                            working_width=width,
                            working_height=height,
                            reference_values=None)

        return patch_data.swatch_colours

    def compute_ccm(self, 
                    observed_vals: np.array,
                    reference_vals: np.array) -> np.array:
        """Compute the colour correction matrix for the calibration target,
        from the given observed and reference values.

        :param observed_vals: Array of observed values for each patch
        :type observed_vals: np.array
        :param reference_vals: Array of reference values for each patch
        :type reference_vals: np.array
        :return: 3x3 Colour correction matrix
        :rtype: np.array
        """     
        # TODO make checks on the observed and reference values arrays
        ccm = colour.matrix_colour_correction(observed_vals, reference_vals)
        self.ccm = ccm
        return ccm

    def find_in_scene(self, rgb_image) -> Tuple:
        """Automatically find the Calibration Target 
        using the colour checker detection algorithm

        :param image: The image containing the calibration target
        :type image: np.array
        """
        # run the colour checker detection algorithm
        decoded_image = colour.cctf_decoding(rgb_image)

        # this algorithm finds the colour checker values of the image supplied.
        # We want to get the patch locations though, so that we can draw
        # them on other images - e.g. we find the patch locations in an RGB
        # image, and then draw them on the multispectral image.
        print("Searching for colour checker...")
        colour_checker_data = detect_colour_checkers_inference(
                                    decoded_image, 
                                    additional_data=True, 
                                    show=True)
        
        # check if the run was successful
        if colour_checker_data == ():
            print("No colour checker found")
            print("Searching for colour checker in cropped image...")
            cropped_image = decoded_image[150:-150, 150:-150]
            colour_checker_data = detect_colour_checkers_inference(
                                    cropped_image, 
                                    additional_data=True, 
                                    show=True)
            if colour_checker_data == ():
                print('Manually draw out the calibration target quadrilateral')
                result = self.draw_target_outline(rgb_image)
                if result is False:
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

        return self.target_outline

    def find_target_and_compute_ccm(self, 
                    frame, 
                    illuminant: Literal[
                        'A', 'B', 'C', 'D50', 'D55', 'D65', 'D75', 'ICC D50'
                    ]='ICC D50',
                    show: bool=False) -> np.array:
        """Find the calibration target in the given frame, and compute the
        colour correction matrix from the observed values and reference values.

        :param frame: The image containing the calibration target
        :type frame: RGB
        :return: 3x3 Colour correction matrix
        :rtype: np.array
        """
        # find the calibration target in the image
        approx_balance_rgb = frame.get_image('99p')
        result = self.find_in_scene(approx_balance_rgb)

        if result is False:
            print("No calibration target found")
            return False

        if show:
            self.show_target_outline(approx_balance_rgb)
            
        drgb_image = frame.get_image('raw', np.uint8) # get vals from raw image
        obs_ct_dRGB_vals = self.get_observed_vals(drgb_image)
        # get the reference values
        ref_ct_sRGB_vals = self.load_reference_vals('sRGB', illuminant=illuminant)
        # compute the colour correction matrix
        ccm = self.compute_ccm(obs_ct_dRGB_vals, ref_ct_sRGB_vals)

        frame.ccm = ccm
        
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
        self.camera = None
        self.pan = None
        self.tilt = None
        self.exposure = None
        self.capture_timestamp = None

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
        self.cwl = aupe_info.cwl[self.channel]
        self.fwhm = aupe_info.fwhm[self.channel]
        self.filter_id = aupe_info.filter_id[self.channel]

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

    def get_image(self, 
                  stretch_method: Literal['raw', 'bps', 'wps', '99p']=None
                  ) -> np.ndarray:
        """Get a copy of the image data, optionally applying the stretch method
        :param stretch_method: method for finding the stretch coefficient
        :type stretch_method: Literal['raw', 'bps', 'wps', '99p'], optional
        :return: image data
        :rtype: np.ndarray
        """
        if stretch_method is None:
            image = self.image.copy()
        else:
            image = self.apply_stretch(stretch_method)
        return image

    def apply_stretch(self, stretch_method: Literal['raw', 'bps', 'wps', '99p']='raw'):
        """Apply the stretch coefficient to the image, and return the stretched image.
        Stretched iamge is always in the range of 0.0 to 1.0.

        :param stretch_method: method for finding the stretch coefficient
        :type stretch_method: Literal['raw', 'bps', 'wps', '99p']
        :return: stretched image
        :rtype: np.ndarray
        """
        print(f"Stretching image using {LEVEL_DICT[stretch_method]}")
        if self.stretch[stretch_method]['factor'] is None:
            self.extract_stretch_coefficient(stretch_method)
        print(f'Applying stretch factor of {self.stretch[stretch_method]["factor"]}')
        disp_img = np.clip(self.image * self.stretch[stretch_method]['factor'], 0.0, 1.0)        
        
        return disp_img

    def extract_stretch_coefficient(self, 
                method: Literal['raw', 'bps', 'wps', '99p']='raw',
                wp_roi: Tuple[int, int, int, int]=None) -> float:
        """Get the stretch coefficient for the given image according to
        the selected method.
        'raw': no stretch, just divide by the max value for the bit-depth
        'bps': "brightest pixel stretch" - stretches such that the brightest pixel
        has a value of 1.0.
        '99p': "99th percentile stretch" - stretches such that the 99th percentile
        of the image is 1.0.
        'wps': "white patch stretch" - stretches such that the white patch of the
        MacBeth colorchecker has a value of 1.0.

        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99p']
        """      
        if method == 'raw':
            # set the stretch coefficient to 1.0 / max bit-depth value
            self.stretch['raw'] = {}
            if self.units == 'DN':
                self.stretch['raw']['factor'] = 1.0 / ((2**BIT_DEPTH) - 1)
            elif self.units == 'DN/s':
                self.stretch['raw']['factor'] = 1.0 / ((2**BIT_DEPTH) - 1) * self.exposure
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
        elif method == '99p':
            # get the 99th percentile pixel value and location
            pct_val = np.percentile(self.image, 99)            
            # set the stretch coefficient to 1.0 / bp_val
            self.stretch['99p'] = {}
            self.stretch['99p']['factor'] = 1.0 / pct_val
            self.stretch['99p']['roi'] = (0, 0, self.width, self.height)
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
                method: Literal['all', 'raw', 'bps', 'wps', '99p']='all'):
        """Reset the stretch coefficient for the given image according to
        the selected method.
        :param method: method for finding the stretch coefficient
        :type method: Literal['raw', 'bps', 'wps', '99p']
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
                '99p': {
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
        # check if the image is already in DN/s
        if self.units == 'DN/s':
            print("Image already in DN/s")
            return
        
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
        if self.stretch['99p']['factor'] is not None:
            self.stretch['99p']['factor'] = self.stretch['99p']['factor'] * self.exposure
 
    def flat_field(self):
        pass

    def bias_correction(self):
        pass

    def show_image(self, 
                   stretch_method: Literal['raw', 'bps', 'wps', '99p']='raw'
                ) -> Tuple[plt.Figure, plt.Axes]:
        """Display the image using matplotlib,
        and optionally show the histogram of the image data.
        """  

        title = f"{self.sol} {self.scene} {self.trial} {self.channel} {self.cwl}±{int(self.fwhm/2)} nm ({stretch_method})"

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

        # show the image at full resolution
        plt.imshow(disp_img, interpolation='none')
        plt.axis('off')
        # set the title
        plt.title(title)

        return fig, ax

    def export_image(self, stretch_method: Literal['raw', 'bps', 'wps', '99p']='bps'):
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
                 rgb_path_dict: Tuple[Dict, Dict, Dict],
                 aupe_info: AupeInfo):
        self.red = Img(rgb_path_dict[2], aupe_info)
        self.green = Img(rgb_path_dict[1], aupe_info)
        self.blue = Img(rgb_path_dict[0], aupe_info)
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)
        self.ccm = np.empty((3,3))
        self.balance_vector = {
            'raw': np.zeros(3),
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

    def exposure_correct(self):
        """Exposure correct each channel
        """        
        self.red.exposure_correction()
        self.green.exposure_correction()
        self.blue.exposure_correction()
        self.units = self.red.units
        self.dtype = self.red.dtype
        # update the rgb image
        self.rgb_image = np.stack([self.red.image, self.green.image, self.blue.image], axis=2)

        # update the balance vector
        for method in self.balance_vector.keys():
                self.balance_vector[method] = self.balance_vector[method] * self.red.exposure

    def flat_field(self):
        pass

    def bias_subtract(self):
        pass

    def apply_ccm(self):
        """Apply the colour correction matrix to the RGB image
        """
        drgb_image  = self.get_image('raw', np.uint8)
        # check if the ccm is set
        if self.ccm is None:
            print("No colour correction matrix set")
            # search for the latest calibration target in the 
        srgb_image = colour.apply_matrix_colour_correction(drgb_image, self.ccm)

        # # apply 0 - 1 normalisation and clipping
        # srgb_image = srgb_image.astype(np.float64) / np.max(srgb_image)
        # srgb_image = np.clip(srgb_image, 0.0, 1.0)

        return srgb_image

    def export_ccm(self):
        """Export the colour correction matrix to a csv file
        """
        # get the output directory
        out_dir = Path('..', 'data', 'ccms', self.camera)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save the ccm to a csv file
        ccm_df = pd.DataFrame(self.ccm)
        # add the camera, sol, scene, trial to the filename
        # TODO figure if there is any other metadata we can apply - e.g. in shade, in sun, indoors etc.
        filename = f"ccm_{self.camera}_{self.sol}_{self.scene}_{self.trial}.csv"
        ccm_df.to_csv(Path(out_dir, filename), index=False, header=False)
        print(f"CCM saved to {out_dir}/ccm.csv")

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
        ccm_dir = Path('..', 'data', 'ccms', camera)
        # check the ccm dir exists
        if not ccm_dir.exists():
            raise FileNotFoundError(f"CCM directory {ccm_dir} does not exist")
        # load the ccm from a csv file
        filename = f"ccm_{camera}_{sol}_{scene}_{trial}.csv"
        # check the file exists
        if not Path(ccm_dir, filename).exists():
            raise FileNotFoundError(f"CCM file {filename} does not exist in {ccm_dir}")
        ccm_df = pd.read_csv(Path(ccm_dir, filename), header=None)
        self.ccm = ccm_df.to_numpy()
        print(f"CCM loaded from {ccm_dir}/ccm.csv")

    def apply_balance_vector(self, method: Literal['raw', 'bps', 'wps', '99p']='raw'):

        print(f"Stretching image using {LEVEL_DICT[method]}")
        if (self.balance_vector[method] == np.zeros(3)).all():
            self.extract_balance_vector(method)
        r_disp_img = np.clip(self.red.image * self.balance_vector[method][0], 0.0, 1.0)        
        g_disp_img = np.clip(self.green.image * self.balance_vector[method][1], 0.0, 1.0)
        b_disp_img = np.clip(self.blue.image * self.balance_vector[method][2], 0.0, 1.0)
                
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
                'raw': np.zeros(3),
                'bps': np.zeros(3),
                'wps': np.zeros(3),
                '99p': np.zeros(3)
            }
        elif method == 'raw':
            self.balance_vector[method] = np.ones(3)
        else:
            self.balance_vector[method] = np.zeros(3)            

    def get_image(self,
                  colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm']='raw',
                  dtype: Literal[np.uint8, np.uint16, np.float32, np.float64]=None
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
            if self.balance_vector[colour_correction][0] == 0.0:
                self.extract_balance_vector(colour_correction)
            # apply the balance vector
            image = self.apply_balance_vector(colour_correction)
        
        if dtype != None:
            if image.dtype != dtype:
                # make conversion of dtype
                if dtype == np.uint8:
                    image = (np.clip(image / image.max(), 0,1) * 255).astype(np.uint8)
                elif dtype == np.uint16:
                    image = (np.clip(image / image.max(), 0,1) * 65535).astype(np.uint16)
                elif dtype == np.float32:
                    image = (np.clip(image / image.max(), 0,1)).astype(np.float32)
                elif dtype == np.float64:
                    image = (np.clip(image / image.max(), 0,1)).astype(np.float64)

        return image

    def show_image(self, 
                   colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm']='raw'):
        """Display the RGB image using matplotlib,
        and optionally show the histogram of the image data.
        """
        title = f"{self.sol} {self.scene} {self.trial} {self.camera} RGB ({LEVEL_DICT[colour_correction]})"

        disp_img = self.get_image(colour_correction, dtype=np.uint8)   
        
        # fig, ax = plt.subplots(1,2, figsize=(8, 4))  
        # disp = ax[0].imshow(disp_img)

        # # for each channel, show a histogram - retain original image units
        # ax[1].hist(disp_img[:,:,0].ravel(), bins=256, histtype='stepfilled', color='red', alpha=0.5)
        # ax[1].hist(disp_img[:,:,1].ravel(), bins=256, histtype='stepfilled', color='green', alpha=0.5)
        # ax[1].hist(disp_img[:,:,2].ravel(), bins=256, histtype='stepfilled', color='blue', alpha=0.5)
        # ax[1].tick_params(labelleft=False, left=False)
        # # label x axis with units
        # ax[1].set_xlabel(f"Pixel Value {self.units} ({colour_correction})")
        # # add title and labels
        # fig.suptitle(title)
        # plt.show()

        # show the image at full resolution
        fig, ax = plt.subplots(1,1, figsize=(8, 4))  
        ax.imshow(disp_img, interpolation='none')
        ax.axis('off')
        # set the title
        fig.suptitle(title)

        return fig, ax

    def export_image(self, colour_correction: Literal['raw', 'bps', 'wps', '99p', 'ccm']='bps'):
        """Export the image to a file, using the stretch method

        :param stretch_method: Stretch method to use, defaults to 'raw'
        :type stretch_method: Literal['raw', 'bps', 'wps'], optional
        """   
        
        title = f"{self.sol}_{self.scene}_{self.trial}_{self.camera}_RGB_{colour_correction}.png"

        print(f"Exporting image using {LEVEL_DICT[colour_correction]}")
        if colour_correction == 'raw':
            if self.units == 'DN/s':
                # convert to uint8
                disp_img = (self.rgb_image * self.red.exposure).astype(np.uint8) # note this doesn't actually work...
            else:
                disp_img = (self.rgb_image).astype(np.uint8)
        elif colour_correction == 'ccm':
            # apply the colour correction matrix
            disp_img = self.get_image('ccm', np.uint8)
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
        
class WAC_RGB(RGB):
    def __init__(self, rgb_path_dict: Tuple[Dict, Dict, Dict],
                 aupe_info: AupeInfo):
        super().__init__(rgb_path_dict, aupe_info)

class MSC:
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
