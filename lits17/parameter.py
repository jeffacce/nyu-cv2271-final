# -----------------------Path related parameters---------------------------------------

# CT data path of the original training set
train_vol_path = '''/scratch/ec2684/cv/data/lits17/processed/train/nii/vol/unit_voxel/'''
train_seg_path = '''/scratch/ec2684/cv/data/lits17/processed/train/nii/seg/unit_voxel_01/'''

# ---------------------Training data to obtain relevant parameters-----------------------------------

# Use 64 consecutive slices as input to the network by default
size = 64
# Only use the liver and the upper and lower 20 slices of the liver as the training sample
expand_slice = 20
# Normalize the spacing of all data on the z-axis to 1mm
slice_thickness = 1 

# CT data gray scale truncation window
upper, lower = 200, -200 
# upper, lower = 300, -300
seed = 0
test = False
early_stopping = 10000
# -----------------------Network structure related parameters------------------------------------

drop_rate = 0.5  # dropout random drop probability

# ---------------------Network training related parameters--------------------------------------

gpu = '0'  # The serial number of the graphics card used

epochs = 1000

learning_rate = 1e-4
learning_rate_decay = [500,750]

alpha = 0.33  # Depth supervision attenuation coefficient

batch_size = 1
# num_workers = 3
num_workers = 1

pin_memory = True

cudnn_benchmark = True

n_classes = 3

# ----------------------Model test related parameters-------------------------------------

threshold = 0.5  # Threshold degree threshold

stride = 12  # Sliding sampling step

maximum_hole = 5e4  # Largest void area

# ---------------------CRF post-processing optimization related parameters----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # The number of expansions in three directions based on the predicted results

max_iter = 20  # Number of CRF iterations

s1, s2, s3 = 1, 10, 10  # CRF Gaussian kernel parameters

# ---------------------CRF post-processing optimization related parameters----------------------------------
