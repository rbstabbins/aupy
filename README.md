# aupy: AUPE Reflectance and Colour Calibration in Python

![AUPE Reflectance and Colour Calibration](./example_images/cover.png)

## About <a name = "about"></a>

This is a Python toolkit for processing AUPE ([Aberystwyth University PanCam Emulator](https://exomars.wales/facilities/aupe/)) images into colour and reflectance products, through calibration against in-scene images of the MacBeth/Gretag/ColorRite Colour Checker 24-patch colour calibration target.

Raw 8-bit png files, output by AUPE, are read and processed into reflectance units, and collected into multispectral cubes and exported to ENVI hdr/img format files, for analysis via standard spectral imaging software (e.g. [ENVI](https://www.nv5geospatialsoftware.com/docs/ProgrammingGuideIntroduction.html), [SpectralPython](https://www.spectralpython.net/), [WISER](https://ehlmann.caltech.edu/wiser/index.html)).

The toolkit uses the Python [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) and [Colour Science](https://www.colour-science.org/) libraries to assist with colour processing and image processing functions.

## Installation and Setup <a name = "setup"></a>

These instructions will get you a copy of the project up and running on your computer.

This guide has been written with an assumption of no prior experience with github or installing python libraries.

### Prerequisites

#### Code Editor
If you're new to code editing, I recommend using VisualStudioCode to edit and run your Python notebooks and libraries. You can install it here:
[VisualStudioCode](https://code.visualstudio.com/).

#### Downloading aupy
Next, we need to download this github repository, that contains the codebase (aupy.py), the template notebooks for performing processing tasks, and a set of various calibration files that are need to perform calibration.

Setup a new folder on your computer for your AUPE processing project.

If you are new to github and coding, you can just download a copy of this as a zip file.

![alt text](./example_images/image.png)

If you are comfortable with git repositories, feel free to clone this repository to stay up-to-date with developments and bug-fixes.

Unzip this into your project folder.

#### Environment Setup

Now we will get the code running.

Open VSCode, and when prompted to open a folder, navigate to the aupy repository you just unzipped. This setups VSCode to work with this repository.

We now need to setup a virtual environment for running the code in. In the repository there is an ```environment.yml`` file that gives a list of dependencies that need to be installed to run this code.

We use ```conda``` to setup and manage this virtual environment.

You can install ```miniconda``` to do this here: [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install)

In VSCode, navigate to the terminal at the bottom of the window. We will use the terminal to setup this environment.

First, we can check quickly that we are indeed in the correct directory, by running ```pwd```:

![alt text](./example_images/image-1.png)

This should print the path to the directory of this repository. [If you're having trouble navigating your folders in the terminal this might help](https://gomakethings.com/navigating-the-file-system-with-terminal/).

Check that conda has installed by typing ```conda -V```. This should print the version number. If instead it says something like ```command not found: conda```, then try installing miniconda again, and troubleshoot using the conda website.

We can now setup the virtual environment, by running

```
conda env create --name aupy --file environment.yml
```

This will install a clean Python distribution, as well as the Python libraries needed to run aupy.py.

## Usage <a name = "usage"></a>

The ```ries_scene_complete_template.ipynb``` provides a template for processing AUPE scenes. This notebook, and this toolkit, have been developed for processing data captured during the Canadian Space Agency x UK Space Agency field trip to the Ries Crater, May 2025.

To run this notebook and process the data, the raw data from the field trip needs to be downloaded and put in a repository called ```data``` in the project folder, next to the aupy repository.

```
project
├── aupy
│  ├── aupy.py
│  ├── data
│  │  ├── aupe_info.csv
│  │  └── ...
├── data
│  ├── SOL1
│  ├── SOL2
│  │  ├── sol2_0
│  │  │  ├── Trial1
│  │  │  │  └── ...
├── processed
│  ├── SOL1
│  ├── SOL2
│  │  ├── sol2_0
│  │  │  ├── Trial1
│  │  │  │  └── ...
```

The notebook is setup to demonstrate processing of scene SOL2/sol2_8.

When you open the notebook, you should be prompted to select a kernel for the notebook. Navigate to Python Environments, and you should see an option called 'aupy (Python 3.13.x...)'. Choose this kernel. If this isn't visible, try restarting Visual Studio Code - it may not have registered that you have setup your new conda environment.

Please follow through the notebook, running it on your computer, to understand the steps of processing, and how the aupy library is called and used.

Processed data is exported to a structured directory at ```processed```.

## Overview of Processing

Here is an overview of a typical processing routine, as shown in the template notebook.

The key parts are:
1. LWAC Processing
   1. RGB Approx. Processing
   2. RGB Reflectance & Colour Calibration
   3. Geology Reflectance Calibration
2. RWAC Processing
   1. RGB Approx. Processing
   2. RGB Reflectance & Colour Calibration
   3. Geology Reflectance Calibration
3. LWAC->RWAC Rectification
   1. Choose common feature to find optimal warp for feature distance
   2. Warp LWAC->RWAC to give LRWAC 12-band Cube from RWAC View
4. HRC Processing
   1. RGB Approx. Processing
   2. Reflectance transfer from RWAC Reflectance Calibration
5. RWAC->HRC Rectification
   1. Choose common feature to find optimal warp for distance
   2. Warp LRWAC 12-band Cube to HRC View

### LWAC Processing
   #### RGB Approx. Processing
   ![alt text](./example_images/image-2.png)
   #### RGB Reflectance & Colour Calibration
   ![alt text](./example_images/image-5.png)
   ![alt text](./example_images/image-4.png)
   ![alt text](./example_images/image-6.png)
   ![alt text](./example_images/image-7.png)
   #### Geology Reflectance Calibration
   ![alt text](./example_images/image-8.png)
   ![alt text](./example_images/image-9.png)
   ![alt text](./example_images/SOL2_sol2_8_Trial1_LWAC_G01G02G03G04G05G06_refl.gif)
### RWAC Processing
   #### RGB Approx. Processing
   ![alt text](./example_images/image-10.png)
   #### RGB Reflectance & Colour Calibration
   ![alt text](./example_images/image-11.png)
   #### Geology Reflectance Calibration
   ![alt text](./example_images/image-13.png)
   ![alt text](./example_images/image-12.png)
### LWAC->RWAC Rectification
   #### Choose common feature to find optimal warp for feature distance
   ![alt text](./example_images/image-14.png)
   ![alt text](./example_images/image-15.png)
   #### Warp LWAC->RWAC to give LRWAC 12-band Cube from RWAC View
   ![alt text](./example_images/image-16.png)
   ![alt text](./example_images/SOL2_sol2_8_Trial1_LRWAC_G01G02G03G04G05G06G07G08G09G10G11G12_refl_contact.gif)
### HRC Processing
   #### RGB Approx. Processing
   ![alt text](./example_images/image-17.png)
   #### Reflectance and Colour transfer from RWAC Reflectance Calibration
   ![alt text](./example_images/image-18.png)
### RWAC->HRC Rectification
   #### Choose common feature to find optimal warp for distance
   ![alt text](./example_images/image-19.png)
   ![alt text](./example_images/image-20.png)
   #### Warp LRWAC 12-band Cube to HRC View
   ![alt text](./example_images/SOL2_sol2_8_Trial1_RWAC_RGB_G11G05G01_99b.png)
    ![alt text](./example_images/SOL2_sol2_8_Trial1_LRWHRC_G01G02G03G04G05G06G07G08G09G10G11G12_refl_belowcontact.gif)

## Author

This library has been developed by Roger Stabbins at the Natural History Museum, London, UK, since May 2025.