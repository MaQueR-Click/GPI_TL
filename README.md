# STRUCTURAL PROJECT GPI 
## project of GPI using Python by Tim 

to launch the code the package require are "numpy mrcfile scipy matplotlib scikit-image"

3 script are require and 1 is optionnal 

launch in order 

1 ----> binning.py  put in input the wished mrcfile and below in parameter choose the binning factor.

2 ----> PDB_oppening.py put the PDB_id then the PDB_url, it extract the templates of 3 point of view in the folder 

3 ----> cross-correlation.py in input use the binned image, for the template_input use the 3 or 2 view of the template XY,XZ,YZ
            feature incorporate = normalization of the templates and the image, padding for image and template if wanted, normalized cross-correlation 
            
            parameter you can change - in preprocess image choose the padding of the image wanted
                                     - in process template change the sigma for gaussian smoothing of template
                                     - in templates-input = diameter of the particle, threshold wished
                                     - choose the matching pixel size 
                                     - in parameters angles step for NCC -name of the output directory  
                                     - sigma noise in the templates if wanted 
                                     - duplicated_pixel  select only one value of NCC for radius of 15 px  
                                     - 

4 ----> FFT_NCC.py in input use the binned image, for the template_input use the 3 or 2 view of the template XY,XZ,YZ
            feature incorporate = normalization of the templates and the image, padding for image and template if wanted ,normalized, convertion templates and image in fourier space and fourier space cross-correlation

            parameter you can change - in preprocess image choose the padding of the image wanted
                                     - in process template change the sigma for gaussian smoothing of template
                                     - in templates-input = diameter of the particle, threshold wished
                                     - choose the matching pixel size
                                     - in parameters angles step for NCC -name of the output directory  
                                     - sigma noise in the templates if wanted 
                                     - duplicated_pixel  select only one value of NCC for radius of 15 px  