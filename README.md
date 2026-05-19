# STRUCTURAL PROJECT GPI 
## project of GPI using Python by Tim 

to launch the code the package require are "numpy mrcfile scipy matplotlib scikit-image"
pip install numpy mrcfile scipy matplotlib scikit-image biopython

3 script are require and 1 is optionnal 

to launch you need to have all the script in the folder and be in the folder. 

1 ----> binning.py  put in input the wished mrcfile and below in parameter choose the binning factor.

        launch with the command in the terminal = python3 binning.py image.mrc binfactor # exemple python3 binning.py image.mrc 4  

2 ----> PDB_oppening.py put the PDB_id then the PDB_url, it extract the templates of 3 point of view in the folder 

        launch with the command in the terminal = python3 PDB_oppening.py pdb_id pixel_size # exemple python3 PDB_oppening.py 6BDF 2.64

3 ----> cross-correlation.py in input use the binned image, for the template_input use the 3 or 2 view of the template XY,XZ,YZ
            feature incorporate = normalization of the templates and the image, padding for image and template if wanted, normalized cross-correlation 
            
        launch with the command in the terminal = python cross-correlation.py binning_image.mrc "pixel_size" "pdb_template.mrc":"particles size":"threshold"
        # python3 cross-correlation.py binning_4_image.mrc 2.64 6BDF_XY_template.mrc:115:0.26 6BDF_XZ_template.mrc:150:0.30

            parameter you can change - in preprocess image choose the padding of the image wanted
                                     - in process template change the sigma for gaussian smoothing of template
                                     - in templates-input = diameter of the particle, threshold wished
                                     - choose the matching pixel size 
                                     - in parameters angles step for NCC -name of the output directory  
                                     - sigma noise in the templates if wanted 
                                     - duplicated_pixel  select only one value of NCC for radius of 15 px  
                                    

4 ----> FFT_NCC.py in input use the binned image, for the template_input use the 3 or 2 view of the template XY,XZ,YZ
            feature incorporate = normalization of the templates and the image, padding for image and template if wanted ,normalized, convertion templates and image in fourier space and fourier space cross-correlation

            launch with the command in the terminal = python FFT_NCC.py binning_image.mrc "pixel_size" "pdb_template.mrc":"particles size":"threshold"
            # python3 FFT_NCC.py binning_4_image.mrc 2.64 6BDF_XY_template.mrc:115:0.26 6BDF_XZ_template.mrc:150:0.30

            parameter you can change - in preprocess image choose the padding of the image wanted
                                     - in process template change the sigma for gaussian smoothing of template
                                     - in templates-input = diameter of the particle, threshold wished
                                     - choose the matching pixel size
                                     - in parameters angles step for NCC -name of the output directory  
                                     - sigma noise in the templates if wanted 
                                     - duplicated_pixel  select only one value of NCC for radius of 15 px  