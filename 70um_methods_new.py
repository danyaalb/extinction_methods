from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
import numpy as np
from regions import Regions
import astropy.io.fits as pyfits
import os
from reproject import reproject_interp
from scipy import ndimage, misc
from astropy.wcs import WCS
import pyregion
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve_fft
from scipy import ndimage, misc
from matplotlib.colors import LogNorm
from astropy.nddata import Cutout2D
import matplotlib as mpl
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy.io import fits
import numpy as np
matplotlib.rcParams['font.sans-serif'] = "georgia" 

class Person:
    def __init__(self, sub_dict, selected_names=None):
        """
        Takes in a dictionary of names and ellipse regions
        region: List of regions (e.g., read from a file or passed as a parameter)
        name: List of region names corresponding to each region
        selected_names: List of names to filter regions
        """
        self.region = []
        self.name = []
        
        if selected_names:
            for nawhme in selected_names:
                if name in sub_dict:
                    self.region.append(sub_dict[name])
                    self.name.append(name)
        else:
            self.region = list(sub_dict.values())
            self.name = list(sub_dict.keys())

#class used to mask and crop individual clouds
class IndividualMasks(Person):
    def reproject_and_save(self, file, output_filename_template):
        """
        Reprojects the mask, applies it to the data, and saves the masked FITS file for each region.

        Parameters:
        file: str
            Filename for the background file.
        output_filename_template: str
            Template for saving the masked FITS file (should include a placeholder for the region name).
        """
        # Open and unpack the FITS file
        file_data = open_files(file)
        hdu_file, wcs_file = file_data

        # Loop over each region
        for name, region in zip(self.name, self.region):
            # Convert the region to pixel coordinates
            pixel_region = region.to_pixel(wcs_file)
            
            # Create the mask for the region
            region_mask = pixel_region.to_mask(mode='center').to_image(hdu_file.data.shape)
            
            if region_mask is not None:
                ### 1. First Mask: Region with original data, background as NaN ###
                
                # Copy the original data and mask out the background (NaN for background)
                data_with_background_nan = hdu_file.data.copy()
                data_with_background_nan[~region_mask.astype(bool)] = np.nan  # Set background to NaN
                
                # Save the first FITS file (original data within mask, NaN for background)
                output_filename = output_filename_template.format(name)
                self.save_masked_fits(data_with_background_nan, wcs_file, output_filename)
                
    def save_masked_fits(self, data, wcs, output_filename):
        """
        Saves the masked data into a new FITS file.

        Parameters:
        data: ndarray
            Data from the original file with NaNs in the background.
        wcs: WCS object
            WCS of the original file.
        output_filename: str
            Filename of the new masked FITS file.
        """
        # Mask the data to find valid entries
        masked_data = np.ma.masked_invalid(data)

        # Check if there are any valid entries
        if masked_data.count() == 0:
            raise ValueError("The data contains only NaN values; nothing to save.")
        
        # Get the bounding box of the non-NaN data
        min_y, max_y = np.where(~masked_data.mask)[0].min(), np.where(~masked_data.mask)[0].max()
        min_x, max_x = np.where(~masked_data.mask)[1].min(), np.where(~masked_data.mask)[1].max()

        # Crop the data to the bounding box and convert to a regular NumPy array
        cropped_data = masked_data[min_y:max_y + 1, min_x:max_x + 1].data
        
        # Create a new PrimaryHDU with the cropped data
        hdu = fits.PrimaryHDU(cropped_data)

        # Update the header with the WCS info, adjusting it for the cropped data
        wcs_header = wcs.to_header()
        wcs_header['CRPIX1'] -= min_x  # Adjust CRPIX for x
        wcs_header['CRPIX2'] -= min_y  # Adjust CRPIX for y
        hdu.header.update(wcs_header)

        # Write the cropped data to a new FITS file
        hdu_masked = fits.HDUList([hdu])
        hdu_masked.writeto(output_filename, overwrite=True)
        print(f"Saved masked FITS file: {output_filename}")

#method used to mask clouds from full cmz file before smoothing
def reproj_mask(filename, mask_filename, output_filename, threshold=1):
    """
    Takes in a FITS file and mask filenames, applies the mask to the data, and saves the masked data to a new FITS file.
    
    Parameters:
    - filename: str, the path to the input FITS file
    - mask_filename: list of str, paths to mask files (can be region files or FITS files)
    - output_filename: str, the path to save the masked output FITS file
    - threshold: float, the minimum value to consider a pixel part of the mask
    
    Returns:
    - masked_data: numpy masked array containing the masked data
    """
    # Open the input FITS file and extract the HDU (data and header) and WCS
    file = open_files(filename)
    hdu, wcs = file
    header = hdu.header  # Extract the header to preserve WCS information

    # Initialize mask_filled to combine all masks
    mask_filled = np.zeros(hdu.data.shape, dtype=bool)

    # Handle case with multiple mask files
    for i in range(len(mask_filename)):
        if mask_filename[i].endswith('.reg'):  # If it's a region file
            regions = Regions.read(mask_filename[i])  # Load the region file
            for region in regions:
                # Convert the region to pixel coordinates
                pixel_region = region.to_pixel(wcs)

                # Combine the region mask with the filled mask
                mask_filled |= pixel_region.to_mask(mode='center').to_image(hdu.data.shape).astype(bool)
        else:
            mask_file = open_files(mask_filename[i])  # Access each FITS mask file by index
            hdu_mask, wcs_mask = mask_file

            # Reproject the mask and combine with existing mask
            result_reproj, _ = reproject_interp((hdu_mask.data, wcs_mask), wcs, shape_out=hdu.data.shape)
            
            # Apply threshold to the reprojected mask
            mask_outline = np.where(result_reproj == threshold, True, False)  # Threshold for mask
            
            # Fill holes in the mask and combine with existing mask
            mask_filled |= ndimage.binary_fill_holes(mask_outline)

    # Apply the final combined mask to the data
    masked_data = np.ma.masked_array(hdu.data, mask=mask_filled)

    # Save masked data to new FITS file with the original header and WCS
    masked_data_filled = np.ma.filled(masked_data, np.nan)  # Replace masked areas with NaN

    # Create a new Primary HDU to store the masked data
    hdu_new = fits.PrimaryHDU(data=masked_data_filled, header=header)

    # Write the new FITS file
    hdu_new.writeto(output_filename, overwrite=True)

    return masked_data

#class used to smooth 8um and 70um method
class smoothing(Person):
    @staticmethod
    def smooth8um(filename, output_filename):
        """
        Opens background file, smooths masked_data and writes it to a FITS file.
        """
        # Load the FITS file and extract the background data
        background_file = fits.open(filename)
        background = background_file[0].data
        h_hers70 = pyfits.getheader(filename)

        # Gaussian smoothing parameters
        r_eff = 3  # arcmin
        res = r_eff / np.sqrt(2)
        test_res = res * 60  # arcmin to arcseconds
        spitz_res = 1.98 / 2.355  # FWHM to sigma (8um channel FWHM=1.98 arcseconds)

        # Calculate sigma for Gaussian kernel
        sig_gauss = np.sqrt(test_res**2 - spitz_res**2)
        kern = sig_gauss / 1.2  # arcsec/pixel
        smoothed = convolve_fft(background, Gaussian2DKernel(x_stddev=kern), normalize_kernel=True, preserve_nan=False, allow_huge=True)

        pyfits.writeto(output_filename, smoothed, h_hers70, overwrite=True) 


    @staticmethod
    def smooth70um(filename, output_filename):
        """
        Opens 70um file, smooths it, and then writes it to a FITS file.
        """
        background_file = pyfits.open(filename) 

        ## copy the FITS data into a numpy array##
        background = background_file[0].data
        h_hers70 = pyfits.getheader(filename)

        r_eff = 3 #arcmin
        res = r_eff/np.sqrt(2)
        test_res = res * 60 #arcmin * (60"/arcmin), want in arcseconds

        # sigma = FWHM/2.355
        hers70_res = 6/2.355 #70um channel FWHM=6"

        sig_gauss = np.sqrt(test_res**2 - hers70_res**2 )
        kern = sig_gauss/(3.2) #arcsec/pix
        smoothed = convolve_fft(background, Gaussian2DKernel(x_stddev=kern), normalize_kernel=True, preserve_nan=False, allow_huge=True)

        pyfits.writeto(output_filename, smoothed, h_hers70, overwrite=True) 

#used to regrid one file (old_filename) to another (new filename)
def regridding(old_filename, new_filename, output_filename):
    # Open the old FITS file
    file = open_files(old_filename)
    hdu, wcs = file

    # Open the new FITS file
    new_file = open_files(new_filename)
    new_hdu, new_wcs = new_file

    # Reproject the old data to the new WCS
    rescaled_data, _ = reproject_interp((hdu.data, wcs), new_hdu.header)

    # Create a new HDU for the rescaled data with the new header
    rescaled_imagehdu = fits.PrimaryHDU(data=rescaled_data, header=new_hdu.header)

    # Optional: Verify the WCS from the rescaled image
    rescaled_wcs = WCS(rescaled_imagehdu.header)

    # Save the rescaled HDU to a new FITS file
    rescaled_imagehdu.writeto(output_filename, overwrite=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Gaussian2DKernel

class Person:
    def __init__(self, sub_dict, selected_names=None):
        self.region = []
        self.name = []
        
        if selected_names:
            for name in selected_names:
                if name in sub_dict:
                    self.region.append(sub_dict[name])
                    self.name.append(name)
        else:
            self.region = list(sub_dict.values())
            self.name = list(sub_dict.keys())

#class used to do extinction calculations
class plot_70um(Person):
    def abline(self, slope, intercept):
        """Plot a line from slope and intercept"""
        x_vals = np.arange(-1,30)
        y_vals = intercept + slope * x_vals
        return x_vals, y_vals
    def cloud_EmvsExt_NANS_and_SUB(self, cloud, f_fore=0.5):
        cloud_herschel_regrid_fits = pyfits.open("hersch_to_70um_mask_regrid_{}.fits".format(cloud))
        cloud_70um_fits = pyfits.open("70um_data_unsmoothed_mask_{}.fits".format(cloud))
        cloud_70um_smoothed_fits = pyfits.open('70um_data_smoothed_mask_{}.fits'.format(cloud))
        
        cloud_70um = cloud_70um_fits[0].data
        cloud_70um_smoothed = cloud_70um_smoothed_fits[0].data
        cloud_hers_regrid_70 = cloud_herschel_regrid_fits[0].data
        h_cloud_70um = pyfits.getheader("hersch_to_70um_mask_regrid_{}.fits".format(cloud))

        cloud_ext_N_nans = self.cloud_extinction_calc_70um(f_fore, cloud_70um_smoothed, cloud_70um)
        # Update this line with proper formatting
        pyfits.writeto('70umExtN_ffore{:.2f}_tonans_{}.fits'.format(f_fore, cloud), cloud_ext_N_nans, h_cloud_70um, overwrite=True)

        return
    def convolve(self, f_fore=0.5):
        higal_colden_res = 36 / 2.355  # sigma = FWHM/sqrt(8ln2)
        highal70_res = 6 / 2.355
        sig_gauss = np.sqrt(higal_colden_res**2 - highal70_res**2)
        kern = sig_gauss / 3.2  # arcsec/pix

        for cloud in self.name:
            self.cloud_EmvsExt_NANS_and_SUB(cloud, f_fore)

            nans_sub_file = '70umExtN_ffore{:.2f}_tonans_{}.fits'.format(f_fore, cloud)
            nans_sub_data = pyfits.open(nans_sub_file)[0].data

            hers70um = 'destripe_l000_blue_wgls_rcal_cropped.fits'
            h_nans = pyfits.getheader(nans_sub_file)
            h_hers70 = pyfits.getheader(hers70um)

            smoothed_nans = convolve_fft(nans_sub_data, Gaussian2DKernel(x_stddev=kern), normalize_kernel=True, preserve_nan=True, allow_huge=True)

            pyfits.writeto('ExtN70um_sources_to_nans_ffore{:.2f}_cutout_smoothed_conv36_{}.fits'.format(f_fore, cloud), 
                           smoothed_nans, h_nans, overwrite=True)

            # Regridding
            conv_file = pyfits.open('ExtN70um_sources_to_nans_ffore{:.2f}_cutout_smoothed_conv36_{}.fits'.format(f_fore, cloud))[0]
            cloud_hers_cutout = pyfits.open('hersch_colden_masks_{}.fits'.format(cloud))[0]
            regrid_array, regrid_footprint = reproject_interp(conv_file, cloud_hers_cutout.header)

            pyfits.writeto('ExtN70um_sources_to_nans_ffore{:.2f}_cutout_smoothed_conv36_regrid_isolated_{}.fits'.format(f_fore, cloud), 
                            regrid_array, cloud_hers_cutout.header, overwrite=True)

    def plot_extinction(self, f_fore=0.5):
        ext_list_70_conv = []
        em_list_70_conv = []

        num_clouds = len(self.name)
        num_columns = 4  # Set number of columns to 3
        num_rows = (num_clouds + num_columns - 1) // num_columns  # Calculate the number of rows needed

        # Create a grid of subplots with 3 columns
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(25, 5 * num_rows))

        # Flatten the axes array to make indexing easier when plotting
        axes = axes.flatten()

        for idx, cloud in enumerate(self.name):
            regrid_to_nans_file = 'ExtN70um_sources_to_nans_ffore{:.2f}_cutout_smoothed_conv36_regrid_isolated_{}.fits'.format(f_fore, cloud)
            regrid_to_nans_data = pyfits.open(regrid_to_nans_file)[0].data

            cloud_hers = 'hersch_colden_masks_{}.fits'.format(cloud)
            cloud_hers_data = pyfits.open(cloud_hers)[0].data

            em_list_70_conv.append(np.nanmedian(cloud_hers_data) / 10**22)
            ext_list_70_conv.append(np.nanmedian(regrid_to_nans_data) / 10**22)

            # Create the WCS object
            wcs = WCS(pyfits.open(regrid_to_nans_file)[0].header)

            ax = axes[idx]  # Select the correct subplot axis

             # Initialize a flag to run correlation calculation only once
            correlation_done = False
            # Generate the abline
            x = np.arange(-1,30)
            y = np.arange(-1,30)
            x_vals, y_vals = self.abline(0, 3.5) 

            y_vals = np.arange(-5,30)
            x_vals = np.full(np.shape(y_vals),16)

            # Scatter plot of the data
            ax.scatter(cloud_hers_data / 1E22, regrid_to_nans_data / 1E22, marker='+', color='k', alpha=0.5)

            # Plot the abline
            ax.plot(x, y, c='black')

            ax.hlines(0, -1, 30, linestyle='--', color='gray')
            ax.set_ylim(-30, 30)
            ax.set_xlim(0, 32)

            # Add x and y labels only on leftmost and bottommost subplots
            if idx % num_columns == 0:  # Leftmost column, show y-axis
                ax.set_ylabel(r"$N(H_{2})$ [$x10^{22} cm^{-2}$] from Herschel 70 $\mu$m regridded")
            else:  # Hide y-axis on other columns
                ax.set_yticklabels([])

            if idx >= num_columns * (num_rows - 1):  # Bottommost row, show x-axis
                ax.set_xlabel(r"$N(H_{2})$ [$x10^{22} cm^{-2}$] from Hi-GAL")
            else:  # Hide x-axis on other rows
                ax.set_xticklabels([])

            if not correlation_done:
                xy_thin = pd.DataFrame({'Herschel thin': cloud_hers_data.flatten(), 'Ext Col Den thin': regrid_to_nans_data.flatten()})
                optically_thin_corr = xy_thin['Herschel thin'].corr(xy_thin['Ext Col Den thin'])

                xy_all = pd.DataFrame({'Herschel': cloud_hers_data.flatten(), 'Ext Col Den': regrid_to_nans_data.flatten()})
                all_points_corr = xy_all['Herschel'].corr(xy_all['Ext Col Den'])

                # Set the flag to True so it doesn't run again
                correlation_done = True
            # Add cloud name in the top-right of each subplot
            ax.text(.70, .99, '{}'.format(cloud), ha='left', va='top', transform=ax.transAxes,fontsize=20)

            # Correlations
            ax.text(.12, .95, r'\textbf{\underline{Correlations}}', transform=ax.transAxes, ha='center', fontsize=15)
            ax.text(.245, .90, f'Optically Thin Points: {optically_thin_corr:.3f}', transform=ax.transAxes, ha='center', fontsize=15)
            ax.text(.155, .85, f'All Points: {all_points_corr:.3f}', transform=ax.transAxes, ha='center', fontsize=15)

        # Hide any unused subplots
        for ax in axes[num_clouds:]:
            ax.axis('off')

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


    def run_all(self, f_fore=0.5):
        # Call each method in the order you want them to execute
        self.convolve(f_fore)
        self.plot_extinction(f_fore)

#class that will make plots for analysis. Will add 8um flux ratios and differences to this class
class other_plots(Person):
    def plot_six_panel_cloud(self, sub_dict):
        # Get the cloud name from the keys of sub_dict
        #Cloud = list(sub_dict.keys())[0]  # Assuming sub_dict has at least one key
        for Cloud in sub_dict.keys():
                # Load your FITS files as before
                example_smoothed = pyfits.open('70um_data_smoothed_mask_{}.fits'.format(Cloud))[0].data
                wcs_smoothed = WCS(pyfits.open('70um_data_smoothed_mask_{}.fits'.format(Cloud))[0].header)

                example_70um = pyfits.open("70um_data_unsmoothed_mask_{}.fits".format(Cloud))[0].data
                wcs_70um = WCS(pyfits.open("70um_data_unsmoothed_mask_{}.fits".format(Cloud))[0].header)

                example_70um_masked = pyfits.open("70um_sources_removed_unsmoothed_mask_{}.fits".format(Cloud))[0].data
                example_70um_ExtN = pyfits.open('70umExtN_ffore0.50_tonans_{}.fits'.format(Cloud))[0].data
                example_70um_ExtN_conv_regrid = pyfits.open('ExtN70um_sources_to_nans_ffore0.50_cutout_smoothed_conv36_regrid_isolated_{}.fits'.format(Cloud))[0].data

                example_hershel = pyfits.open('hersch_colden_masks_{}.fits'.format(Cloud))[0].data
                wcs_herschel = WCS(pyfits.open('hersch_colden_masks_{}.fits'.format(Cloud))[0].header)

                # Proceed with plotting as before
                plt.rcParams.update({'font.size': 10})
                fig, ax = plt.subplots(2, 3, subplot_kw={'projection': wcs_70um}, figsize = (10,8), dpi = 300)
                #ax_5 = fig.add_subplot(2,3,5, projection=wcs_herschel)
                #ax_6 = fig.add_subplot(2,3,6, projection=wcs_herschel, sharex=ax_5, sharey=ax_5)

                
                mpl.rc('image', cmap='Blues_r')


                #match 70um and 70um masked colorscales 
                im_70um = ax[0,0].imshow(example_70um, cmap='bone')
                vmin_70um, vmax_70um = im_70um.get_clim()
                
                im_70um_masked = ax[0,1].imshow(example_70um_masked, vmin=vmin_70um, vmax=vmax_70um, cmap='bone')
                
                
                im_70um_smoothed = ax[0,2].imshow(example_smoothed, vmin=vmin_70um, vmax=vmax_70um, cmap='bone')
                
                

                #match 70um Ext and 70um Ext Conv/Regrid colorscales 
                #im_herschel = ax_6.imshow(example_hershel)
                
                
                im_herschel = ax[1,1].imshow(example_hershel)
                im_70um_ExtN_conv_regrid = ax[1,2].imshow(example_70um_ExtN_conv_regrid, vmin=vmin_70um, vmax=vmax_70um)
                vmin_hers, vmax_hers = im_herschel.get_clim()
                
                im_70um_ExtN = ax[1,0].imshow(example_70um_ExtN, vmin=vmin_hers, vmax=vmax_hers)
                #im_70um_ExtN_conv_regrid = ax_5.imshow(example_70um_ExtN_conv_regrid, vmin=vmin_hers, vmax=vmax_hers)
                vmin_70Ext, vmax_70Ext = im_70um_ExtN.get_clim()

                print("peak N(H2) = {}".format(np.nanmax(example_hershel)) )
        
    
                ###BEAUTIFY PLOTS####    

                for i in range(2):
                        for j in range(3):
                                lon = ax[i,j].coords['glon']
                                lat = ax[i,j].coords['glat']
                                lon.set_ticks_visible(False)
                                lon.set_ticklabel_visible(False)
                                lat.set_ticks_visible(False)
                                lat.set_ticklabel_visible(False)
                                lon.set_axislabel('')
                                lat.set_axislabel('')
                                ax[i,j].set_facecolor('grey')
                        
                        
                #ax[0,0] and ax[1,0] can keep labels and ticks, the rest dont need them

                lon = ax[0,0].coords['glon']
                lat = ax[0,0].coords['glat']
                lon.set_major_formatter('d.dd')
                lat.set_major_formatter('d.dd')
                lon.set_ticks_visible(True)
                lon.set_ticklabel_visible(True)
                lat.set_ticks_visible(True)
                lat.set_ticklabel_visible(True)
                lon.set_ticks_position('b')
                lon.set_ticklabel_position('b')
                lon.set_axislabel_position('b')
                lat.set_ticks_position('l')
                lat.set_ticklabel_position('l')
                lat.set_axislabel_position('l')
                
                ax[0,0].set_facecolor('gray')
                ax[0,0].set_xlabel('Galactic Longitude')
                ax[0,0].set_ylabel('Galactic Latitude')
                ax[0,0].set_facecolor('grey')


                lon = ax[1,0].coords['glon']
                lat = ax[1,0].coords['glat']
                lon.set_major_formatter('d.dd')
                lat.set_major_formatter('d.dd')
                lon.set_ticks_visible(True)
                lon.set_ticklabel_visible(True)
                lat.set_ticks_visible(True)
                lat.set_ticklabel_visible(True)
                lon.set_ticks_position('b')
                lon.set_ticklabel_position('b')
                lon.set_axislabel_position('b')
                lat.set_ticks_position('l')
                lat.set_ticklabel_position('l')
                lat.set_axislabel_position('l')
                
                ax[1,0].set_facecolor('gray')
                ax[1,0].set_xlabel('Galactic Longitude')
                ax[1,0].set_ylabel('Galactic Latitude')
                ax[1,0].set_facecolor('grey')

        
        
                lon = ax[1,1].coords['glon']
                lat = ax[1,1].coords['glat']
                lon.set_ticks_visible(False)
                lon.set_ticklabel_visible(False)
                lat.set_ticks_visible(False)
                lat.set_ticklabel_visible(False)
                lon.set_axislabel('')
                lat.set_axislabel('')
                ax[1,1].set_facecolor('grey')

                lon = ax[1,2].coords['glon']
                lat = ax[1,2].coords['glat']
                lon.set_ticks_visible(False)
                lon.set_ticklabel_visible(False)
                lat.set_ticks_visible(False)
                lat.set_ticklabel_visible(False)
                lon.set_axislabel('')
                lat.set_axislabel('')
                ax[1,2].set_facecolor('grey')
                
                plt.tight_layout(w_pad = 0.5, h_pad=6)
        
        
        
                cb1 = fig.colorbar(im_70um, ax=ax[0,0], location='top',pad=0.0)
                cb2 = fig.colorbar(im_70um_masked, ax=ax[0,1], location='top',pad=0.0)
                cb3 = fig.colorbar(im_70um_smoothed, ax=ax[0,2], location='top',pad=0.0)
                cb4 = fig.colorbar(im_70um_ExtN, ax=ax[1,0], location='top',pad=0.0)
                cb5 = fig.colorbar(im_70um_ExtN_conv_regrid, ax=ax[1,1], location='top',pad=0.0)
                cb6 = fig.colorbar(im_herschel, ax=ax[1,2], location='top',pad=0.0)

                
                cb1.set_label(label = r'[MJy/sr]', labelpad=-14, x=0.96, rotation=0, fontsize = 7)
                cb2.set_label(label = r'[MJy/sr]', labelpad=-14, x=0.96, rotation=0, fontsize = 7)
                cb3.set_label(label = r'[MJy/sr]', labelpad=-14, x=0.96, rotation=0, fontsize = 7)
                
                cb4.set_label(label = r'[cm$^{-2}$]', labelpad=-14, x=0.99, rotation=0, fontsize = 7)
                cb5.set_label(label = r'[cm$^{-2}$]', labelpad=-14, x=0.99, rotation=0, fontsize = 7)
                cb6.set_label(label = r'[cm$^{-2}$]', labelpad=-14, x=0.99, rotation=0, fontsize = 7)

                
                from string import ascii_lowercase as alc

                for m in range(len(fig.axes[6:])):
                        fig.axes[m].text(0.07, 0.92,'({})'.format(alc[m]), ha='center', va='center', transform=fig.axes[m].transAxes)
                
                
                
                ax[0,0].text(0.82, 0.95,r'Herschel 70$\mu m$', ha='center', va='center', transform=ax[0,0].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)  
                
                ax[0,1].text(0.79, 0.92,'Herschel 70$\mu m$\nSources Removed', ha='center', va='center', transform=ax[0,1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)
                ax[0,2].text(0.82, 0.92,'Herschel 70$\mu m$\nSmoothed', ha='center', va='center', transform=ax[0,2].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)
                ax[1,0].text(0.86, 0.95,r'70$\mu m$ ExtN', ha='center', va='center', transform=ax[1,0].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)
                ax[1,1].text(0.84, 0.92,"70$\mu m$ ExtN\nConv+Regrid", ha='center', va='center', transform=ax[1,1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)
                ax[1,2].text(0.82, 0.95,r'Herschel N(H2)', ha='center', va='center', transform=ax[1,2].transAxes,
                        bbox=dict(facecolor='white', edgecolor='k', alpha=1), fontsize = 8)

                plt.suptitle(f'Six-panel plot for {Cloud}', fontsize=16, y=0.90)
                plt.tight_layout(rect=[0, 0, 1, 0.90])
                # Show the plot
                plt.show()


