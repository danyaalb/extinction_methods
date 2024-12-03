# extinction_methods

Hi! 

This folder has the methods created for my attempt at making a new package, it is still in progress. I made a version that will take in a region file (.reg) and use that as a mask on the file used. For a very detailed description on the method itself see Lipman et. al 2024 (should be on archive). We can then find the relative distances of molecular clouds and find the best fit parabola (representing the X-ray lightcurve) from these distances. 

**extinction_flux_ratio_methods_regions.ipynb** is the code I wrote and tested on the small sample of clouds in the Sgr A region of the galactic center. The notebook includes the same method mentioned earlier and includes an additional analysis comparing the extinction of light from two different wavelengths. In this case, 70 micron data from *Herschel* and 8 micron from *Spitzer* are used. 

My next step is making a version that will take in numpy arrays so that you can mask a cloud more accurately. The version shown here are just ellipses around the clouds that have emitted X-rays. This method will be much more accurate if we make numpy arrays tracing the brightest points in the X-ray emission.
