from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
import shutil
from tifffile import imread, imsave
import os
import csv
from glob import glob
from scipy import signal

# ## **5.2. Error mapping and quality metrics estimation (optional)**
# ---
# <font size = 4>This section can only be used if you have generated a **Quality Control Dataset** that can be used to assess the quality of your Noise2VOID model. Here you need a matching pair of low signal to noise ratio (SNR) image (image that you want to denoise) and a high SNR image (ground truth).
#
# <font size = 4>This section will display Square Error maps and SSIM maps as well as calculating NRMSE and SSIM metrics for all the images provided in the "Source_QC_folder" and "Target_QC_folder" !
#
# <font size = 4>**The Square Error map** display the square of the difference between the normalized predicted and target or the source and the target. In this case, a smaller SE is better. A perfect agreement between target and prediction will lead to an image showing zeros everywhere.
#
# <font size = 4>**The SSIM (structural similarity)** is a common metric comparing whether two images contain the same structures. It is a normalized metric and an SSIM of 1 indicates a perfect similarity between two images. Therefore for SSIM, the closer to 1, the better. The SSIM maps calculates the SSIM metric in each pixel by considering the surrounding structural similarity in the neighbourhood of that pixel (currently defined as window of 11 pixels and with Gaussian weighting of 1.5 pixel standard deviation, see our Wiki for more info).
#

# In[ ]:


#@markdown ##Choose the folders that contain your Quality Control dataset

QC_model_name = "" #@param {type:"string"}
QC_model_path = "" #@param {type:"string"}
Source_QC_folder = "" #@param{type:"string"}
Target_QC_folder = "" #@param{type:"string"}

# Create a quality control/Prediction Folder
if os.path.exists(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction"):
  shutil.rmtree(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")

os.makedirs(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")

# Activate the pretrained model.
model_training = N2V(config=None, name=QC_model_name, basedir=QC_model_path)


# List Tif images in Source_QC_folder
Source_QC_folder_tif = Source_QC_folder+"/*.tif"
Z = sorted(glob(Source_QC_folder_tif))
Z = list(map(imread,Z))

print('Number of test dataset found in the folder: '+str(len(Z)))


# Perform prediction on all datasets in the Source_QC folder
for filename in os.listdir(Source_QC_folder):
  img = imread(os.path.join(Source_QC_folder, filename))
  predicted = model.predict(img, axes='YX', n_tiles=(2,1))
  os.chdir(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")
  imsave(filename, predicted)


def gauss(size, sigma):

    """This function is used to create a window for the calculation of ssim, according to Zhou et al.
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    Addendum: NOW ASSUMING 16 bits!!!

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Currently fixed patch size and sigma size
    size = 11
    sigma = 1.5
    window = gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    # L = 255 #bitdepth of image
    L = 65535 #bitdepth of image

    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    #if cs_map:
    #    return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
    #                (sigma1_sq + sigma2_sq + C2)),
    #            (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    #else:
    return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

def normalizeImageWithPercentile(img):
  # Normalisation of the predicted image for conversion to uint8 or 16bits image (necessary for SSIM)
  img_max = np.percentile(img,99.9,interpolation='nearest')
  img_min = np.percentile(img,0.1,interpolation='nearest')
  return (img - img_min)/(img_max - img_min) # For normalisation between 0 and 1.

def clipImageMinAndMax(img, min, max):
  img_clipped = np.where(img > max, max, img)
  img_clipped = np.where(img_clipped < min, min, img_clipped)
  return img_clipped

def normalizeByLinearRegression(img1, img2):
  # Perform the fit
  linreg = LinearRegression().fit(np.reshape(img1.flatten(),(-1,1)), np.reshape(img2.flatten(), (-1,1)))

  # Get parameters of the regression fit.
  alpha = linreg.coef_
  beta = linreg.intercept_
  # print('alpha: '+str(alpha))
  # print('beta: '+str(beta))

  return img1*alpha + beta

def ssim_index(img1,img2):
  # This function calculates a SSIM index between two images.
  # Note that the images need be suitably normalised as below.

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)

  L = 65535
  c1 = (0.01*L)**2
  c2 = (0.03*L)**2

  #Mean of the images
  mu_img1 = np.mean(img1)
  mu_img2 = np.mean(img2)

  #Variance of the images
  var_img1 = np.mean(np.square(img1-mu_img1))
  var_img2 = np.mean(np.square(img2-mu_img2))

  #Covariance of the images
  cov_img1vsimg2 = np.mean((img1-mu_img1)*(img2-mu_img2))

  Numerator = (2*mu_img1*mu_img2+c1)*(2*cov_img1vsimg2+c2)
  Denominator = (mu_img1**2+mu_img2**2+c1)*(var_img1+var_img2+c2)

  SSIM_index = Numerator/Denominator

  return SSIM_index

# Open and create the csv file that will contain all the QC metrics
with open(QC_model_path+"/"+QC_model_name+"/Quality Control/QC_metrics_"+QC_model_name+".csv", "w", newline='') as file:
    writer = csv.writer(file)

    # Write the header in the csv file
    writer.writerow(["image #","Prediction v. GT mSSIM","Input v. GT mSSIM", "Prediction v. GT NRMSE", "Input v. GT NRMSE"])

    # Let's loop through the provided dataset in the QC folders
    for i in os.listdir(Source_QC_folder):
      if not os.path.isdir(os.path.join(Source_QC_folder,i)):
        print('Running QC on: '+i)
      # -------------------------------- Target test data (Ground truth) --------------------------------
        test_GT = io.imread(os.path.join(Target_QC_folder, i))
        test_GT_norm = normalizeImageWithPercentile(test_GT) # For normalisation between 0 and 1.

      # -------------------------------- Source test data --------------------------------
        test_source = io.imread(os.path.join(Source_QC_folder,i))
        test_source_norm = normalizeImageWithPercentile(test_source) # For normalisation between 0 and 1.
      # Normalize the image further via linear regression wrt the normalised GT image
        test_source_norm = normalizeByLinearRegression(test_source_norm, test_GT_norm)

      # -------------------------------- Prediction --------------------------------
        test_prediction = io.imread(os.path.join(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction",i))
        test_prediction_norm = normalizeImageWithPercentile(test_prediction) # For normalisation between 0 and 1.
      # Normalize the image further via linear regression wrt the normalised GT image
        test_prediction_norm = normalizeByLinearRegression(test_prediction_norm, test_GT_norm)


      # -------------------------------- Calculate the metric maps and save them --------------------------------

      # Calculate the SSIM images based on the default window parameters defined in the function
        GTforSSIM = img_as_uint(clipImageMinAndMax(test_GT_norm,0, 1), force_copy = True)
        PredictionForSSIM = img_as_uint(clipImageMinAndMax(test_prediction_norm,0, 1), force_copy = True)
        SourceForSSIM = img_as_uint(clipImageMinAndMax(test_source_norm,0, 1), force_copy = True)

      # Calculate the SSIM maps
        img_SSIM_GTvsPrediction = ssim(GTforSSIM, PredictionForSSIM)
        img_SSIM_GTvsSource = ssim(GTforSSIM, SourceForSSIM)

      #Save ssim_maps
        img_SSIM_GTvsPrediction_32bit=np.float32(img_SSIM_GTvsPrediction)
        io.imsave(QC_model_path+'/'+QC_model_name+'/Quality Control/SSIM_GTvsPrediction_'+i,img_SSIM_GTvsPrediction_32bit)
        img_SSIM_GTvsSource_32bit=np.float32(img_SSIM_GTvsSource)
        io.imsave(QC_model_path+'/'+QC_model_name+'/Quality Control/SSIM_GTvsSource_'+i,img_SSIM_GTvsSource_32bit)

      # Calculate the Root Squared Error (RSE) maps
        img_RSE_GTvsPrediction = np.sqrt(np.square(test_GT_norm - test_prediction_norm))
        img_RSE_GTvsSource = np.sqrt(np.square(test_GT_norm - test_source_norm))

      # Save SE maps
        img_RSE_GTvsPrediction_32bit = np.float32(img_RSE_GTvsPrediction)
        img_RSE_GTvsSource_32bit = np.float32(img_RSE_GTvsSource)
        io.imsave(QC_model_path+'/'+QC_model_name+'/Quality Control/RSE_GTvsPrediction_'+i,img_RSE_GTvsPrediction_32bit)
        io.imsave(QC_model_path+'/'+QC_model_name+'/Quality Control/RSE_GTvsSource_'+i,img_RSE_GTvsSource_32bit)

      ######### SAVE THE METRIC MAPS HERE #########

      # -------------------------------- Calculate the metrics and save them --------------------------------

      # Calculate the mean SSIM metric
      #SSIM_GTvsPrediction_metrics = np.mean(img_SSIM_GTvsPrediction) # THIS IS WRONG, please compute the SSIM over the whole image and not in patches.
      #SSIM_GTvsSource_metrics = np.mean(img_SSIM_GTvsSource) # THIS IS WRONG, please compute the SSIM over the whole image and not in patches.
        index_SSIM_GTvsPrediction = ssim_index(GTforSSIM, PredictionForSSIM)
        index_SSIM_GTvsSource = ssim_index(GTforSSIM, SourceForSSIM)

      # Normalised Root Mean Squared Error (here it's valid to take the mean of the image)
        NRMSE_GTvsPrediction = np.sqrt(np.mean(img_RSE_GTvsPrediction))
        NRMSE_GTvsSource = np.sqrt(np.mean(img_RSE_GTvsSource))

        writer.writerow([i,str(index_SSIM_GTvsPrediction),str(index_SSIM_GTvsSource),str(NRMSE_GTvsPrediction),str(NRMSE_GTvsSource)])



# All data is now processed saved
Test_FileList = os.listdir(Source_QC_folder) # this assumes, as it should, that both source and target are named the same

plt.figure(figsize=(15,15))
# Currently only displays the last computed set, from memory
# Target (Ground-truth)
plt.subplot(3,3,1)
plt.axis('off')
img_GT = io.imread(os.path.join(Target_QC_folder, Test_FileList[-1]))
plt.imshow(img_GT)
plt.title('Target')

# Source
plt.subplot(3,3,2)
plt.axis('off')
img_Source = io.imread(os.path.join(Source_QC_folder, Test_FileList[-1]))
plt.imshow(img_Source)
plt.title('Source')

#Prediction
plt.subplot(3,3,3)
plt.axis('off')
img_Prediction = io.imread(os.path.join(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction/", Test_FileList[-1]))
plt.imshow(img_Prediction)
plt.title('Prediction')

#Setting up colours
cmap = plt.cm.CMRmap

#SSIM between GT and Source
plt.subplot(3,3,5)
plt.axis('off')
imSSIM_GTvsSource = plt.imshow(img_SSIM_GTvsSource, cmap = cmap, vmin=0, vmax=1)
plt.colorbar(imSSIM_GTvsSource,fraction=0.046, pad=0.04)
plt.title('Target vs. Source SSIM: '+str(round(index_SSIM_GTvsSource,3)))

#SSIM between GT and Prediction
plt.subplot(3,3,6)
plt.axis('off')
imSSIM_GTvsPrediction = plt.imshow(img_SSIM_GTvsPrediction, cmap = cmap, vmin=0,vmax=1)
plt.colorbar(imSSIM_GTvsPrediction,fraction=0.046, pad=0.04)
plt.title('Target vs. Prediction SSIM: '+str(round(index_SSIM_GTvsPrediction,3)))

#Root Squared Error between GT and Source
plt.subplot(3,3,8)
plt.axis('off')
imRSE_GTvsSource = plt.imshow(img_RSE_GTvsSource, cmap = cmap, vmin=0, vmax = 1)
plt.colorbar(imRSE_GTvsSource,fraction=0.046,pad=0.04)
plt.title('Target vs. Source NRMSE: '+str(round(NRMSE_GTvsSource,3)))

#Root Squared Error between GT and Prediction
plt.subplot(3,3,9)
plt.axis('off')
imRSE_GTvsPrediction = plt.imshow(img_RSE_GTvsPrediction, cmap = cmap, vmin=0, vmax=1)
plt.colorbar(imRSE_GTvsPrediction,fraction=0.046,pad=0.04)
plt.title('Target vs. Prediction NRMSE: '+str(round(NRMSE_GTvsPrediction,3)));
