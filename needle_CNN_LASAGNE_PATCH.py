#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import time
import datetime
import os

import lasagne
from copy import deepcopy

import SimpleITK as sitk
import random
from skimage import exposure
from skimage.morphology import binary_closing

import sklearn.cross_validation

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
    This function creates chunk of data for CNN training.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def extract_patches(mri, label, half_patch_size=5, negative_subsample_ratio=500):
    """
    This function divide the MRI into a serie of patches.
    Each patch is classified as "positive" or "negative" on basis of the label.
    """
    assert mri.shape == label.shape
    
    z_size, x_size, y_size = mri.shape
    positive_patches = []
    negative_patches = []
    
    if np.sum(label) <= 0.5:
        return ([], [])

    for z in xrange(half_patch_size, z_size-half_patch_size):
        if label[z,:,:].sum() > 1000: continue # workaround to remove white slices into the label dataset
        for x in xrange(half_patch_size, x_size-half_patch_size):
            for y in xrange(half_patch_size, y_size-half_patch_size):
                if label[z,x,y] == 1:
                    positive_patches.append(mri[z-half_patch_size:z+half_patch_size+1,x-half_patch_size:x+half_patch_size+1,y-half_patch_size:y+half_patch_size+1])
                elif label[z,x,y] == 0:
                    negative_patches.append(mri[z-half_patch_size:z+half_patch_size+1,x-half_patch_size:x+half_patch_size+1,y-half_patch_size:y+half_patch_size+1])
    
    random.shuffle(negative_patches)
    number_of_negative_cases = int(len(negative_patches) / float(negative_subsample_ratio))
    selected_negative_patches = deepcopy(negative_patches[:number_of_negative_cases])
    del(negative_patches)
    
    return (positive_patches, selected_negative_patches)


def patchimg2differentview(patch):
    """
    This function creates a sort of 3D patch by building a patch containing the volume depicted in 3 views 
    """
    z_size, x_size, y_size = patch.shape
    
    X = np.zeros((z_size * 3, x_size, y_size), dtype=np.float32)
    
    counter = 0
    for z in xrange(z_size):
        X[counter,:,:] = patch[z,:,:]
        counter += 1
    for x in xrange(x_size):
        X[counter,:,:] = patch[:,x,:]
        counter += 1
    for y in xrange(y_size):
        X[counter,:,:] = patch[:,:,y]
        counter += 1
    
    return X.astype(np.float32)


def patches2CNNformat(patches, label, half_patch_size=5):
    """
    This function takes a list of patches and create a 4D matrix.
    The patch is represented as a sort of 3D view.
    """ 
    X = np.zeros((len(patches), 3 * ((2 * half_patch_size)+1), (half_patch_size*2)+1, (half_patch_size*2)+1), dtype=np.float32)
    y = np.zeros((len(patches)), dtype=np.int32) * -1
    
    for i, patch in enumerate(patches):
        X[i,:,:,:] = patchimg2differentview(patch)
        y[i] = label
    
    assert -1 not in y
    
    return X.astype(np.float32), y.astype(np.int32)


def get_resized_img(img, data_type = sitk.sitkFloat32):
    """
    This function resizes an image to a fixed shape.
    If data type is sitkFloat32 a linear interpolation is used, otherwise nearest neighbor interpolation is used.
    """
    
    size = img.GetSize()
    ratio = [1.0/i for i in img.GetSpacing()]
    new_size = [int(size[i]/ratio[i]) for i in range(3)]
    
    rimage = sitk.Image(new_size, data_type)
    rimage.SetSpacing((1,1,1))
    rimage.SetOrigin(img.GetOrigin())
    tx = sitk.Transform()
    
    interp = sitk.sitkLinear
    if data_type == sitk.sitkInt16:
        interp = sitk.sitkNearestNeighbor
    
    new_image = sitk.Resample(img, rimage, tx, interp, data_type)
    
    return sitk.GetArrayFromImage(new_image)


def needles2tips(needles, mri, number_of_slices=3):
    """
    This function accepts a list of volumes (each volumes containing a needle) and returns a single volume containing just the tips.
    Here there are a plenty of attempts of removing untrusted data.
    """
    tips = np.zeros_like(mri).astype(np.int32)
    #print(tips.shape)
    for needle in needles:
        needle = needle.astype(np.int32)
        #print("MIN %f, MAX %f" % (needle.min(), needle.max()))
        if np.sum(needle) < (np.shape(needle)[0] * np.shape(needle)[1] * np.shape(needle)[2]):
            #print("Valid needle")
            needle = binary_closing(needle, selem=np.ones((3,3,3)))
            needle[needle!=0]=1
            #print(" after closing: MIN %f, MAX %f " % (needle.min(), needle.max()))
            for z in range(np.shape(mri)[0]-1, 0, -1):
                if 200 > np.sum(needle[z,:,:]) > 0.5 and z-number_of_slices-1 >= 0:
                    #print(" valid slice %d" % z)
                    tmp = deepcopy(needle)
                    tmp[:z-number_of_slices-1,:,:] = 0
                    tips[tmp!=0] = 1
                    del(tmp)
                    break
    
    tips[tips!=0]=1
        
    return tips.astype(np.int32)


def pad_volume(img, half_patch_size=5):
    """
    This function pads a volume in order to extract the patches without losing the information at the border
    """
    npad = ((half_patch_size,half_patch_size),(half_patch_size,half_patch_size),(half_patch_size,half_patch_size))
    return np.lib.pad(img, npad, "constant", constant_values=0)


def join_Xy_posneg(X_pos, y_pos, X_neg, y_neg, test_size1=0.20, test_size2=0.33):
    """
    This function takes X_pos, y_pos, X_neg, y_neg and return X_train, X_val, X_test, y_train, y_val, y_test.
    Test size is first equal to test_size1% of X, then it is split and val size is taken away from it. val size is test_size2% of test size.
    """
    
    X_pos_train, X_pos_test, y_pos_train, y_pos_test = sklearn.cross_validation.train_test_split(X_pos, y_pos, test_size=test_size1)
    X_pos_test, X_pos_val, y_pos_test, y_pos_val = sklearn.cross_validation.train_test_split(X_pos_test, y_pos_test, test_size=test_size2)

    X_neg_train, X_neg_test, y_neg_train, y_neg_test = sklearn.cross_validation.train_test_split(X_neg, y_neg, test_size=test_size1)
    X_neg_test, X_neg_val, y_neg_test, y_neg_val = sklearn.cross_validation.train_test_split(X_neg_test, y_neg_test, test_size=test_size2)

    X_train = np.concatenate((X_neg_train, X_pos_train), axis=0).astype(np.float32)
    y_train = np.concatenate((y_neg_train, y_pos_train), axis=0).astype(np.int32)

    X_val = np.concatenate((X_neg_val, X_pos_val), axis=0).astype(np.float32)
    y_val = np.concatenate((y_neg_val, y_pos_val), axis=0).astype(np.int32)

    X_test = np.concatenate((X_neg_test, X_pos_test), axis=0).astype(np.float32)
    y_test = np.concatenate((y_neg_test, y_pos_test), axis=0).astype(np.int32)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def data_for_CNN(general_folder, half_patch_size=5):
    """
    This is the main function for data generation. It accepts the root folder and returns X_pos, y_pos, X_neg, y_neg.
    Also here there is some strange stuff for trying to exclude untrusted data.
    """
    
    folders_cases = os.listdir(general_folder)
    
    X_pos = np.ones((1, 3 * ((2 * half_patch_size)+1), (half_patch_size*2)+1, (half_patch_size*2)+1), dtype=np.float32) * -1.0
    y_pos = np.ones((1), dtype=np.int32) * -1.0
    
    X_neg = np.ones((1, 3 * ((2 * half_patch_size)+1), (half_patch_size*2)+1, (half_patch_size*2)+1), dtype=np.float32) * -1.0
    y_neg = np.ones((1), dtype=np.int32) * -1.0
    
    for folder_case in folders_cases:
        print("Patient #%s" % (folder_case))
        full_case_path = general_folder + os.sep + folder_case
        
        volumetric_files = os.listdir(full_case_path)
        
        assert "case.nrrd" in volumetric_files
        del(volumetric_files[volumetric_files.index("case.nrrd")])
        print(" %d needles file" %  (len(volumetric_files)))

        
        mri_sitk = sitk.ReadImage(full_case_path + os.sep + "case.nrrd")
        
        needles = []
        valid_needles = 0
        for volumetric_file in volumetric_files:
            label_sitk = sitk.ReadImage(full_case_path + os.sep + volumetric_file)
            label = sitk.GetArrayFromImage(label_sitk)
            background = int(np.around(np.percentile(label, 75), decimals=0))
            #print('Background %d' % background)
            filtered_label = np.zeros_like(label, dtype=np.int32)
            filtered_label[label!=background] = 1
            #print(np.sum(filtered_label))
            
            #label[label!=0]=1
            if np.sum(filtered_label)>0:
                needles.append(filtered_label)
                valid_needles += 1
                
        print(" %d valid needles" %  (valid_needles))
        
        tips = needles2tips(needles, sitk.GetArrayFromImage(mri_sitk))
        tips_sitk = sitk.GetImageFromArray(tips)
        tips_sitk.CopyInformation(mri_sitk)
        tips = get_resized_img(tips_sitk, sitk.sitkInt16)
        tips = pad_volume(tips, half_patch_size=half_patch_size)
        tips[tips!=0]=1
        
        for z in range(tips.shape[0]):
            if np.sum(tips[z,:,:]) > 500:
                tips[z,:,:] = np.zeros_like(tips[z,:,:], dtype=np.int32)
        
        tips[tips!=0]=1
        tips = tips.astype(np.int32)
        #assert np.sum(tips) > 0.5
        
        mri = get_resized_img(mri_sitk)
        mri = pad_volume(mri, half_patch_size=half_patch_size)
        
        # DEBUG
        #data_tag = str(datetime.datetime.now()).replace(" ", "_")
        #mri_sitk_DEBUG = sitk.GetImageFromArray(mri)
        #tips_sitk_DEBUG = sitk.GetImageFromArray(tips)
        #tips_sitk_DEBUG.CopyInformation(mri_sitk_DEBUG)
        #sitk.WriteImage(mri_sitk_DEBUG, "../DEBUG/mri_%s.nrrd" % data_tag)
        #sitk.WriteImage(tips_sitk_DEBUG, "../DEBUG/tips_%s.nrrd" % data_tag)
        
        positive_patches, negative_patches = extract_patches(mri, tips, half_patch_size=half_patch_size)
        if len(positive_patches) > 0:
            print(" %d positive patches and %d negative patches" % (len(positive_patches), len(negative_patches)))
            X_temp_pos, y_temp_pos = patches2CNNformat(positive_patches, 1, half_patch_size=half_patch_size)
            X_temp_neg, y_temp_neg = patches2CNNformat(negative_patches, 0, half_patch_size=half_patch_size)
            
            X_pos = np.concatenate((X_pos, X_temp_pos), axis=0)
            y_pos = np.concatenate((y_pos, y_temp_pos), axis=0)
            X_neg = np.concatenate((X_neg, X_temp_neg), axis=0)
            y_neg = np.concatenate((y_neg, y_temp_neg), axis=0)
    
    assert -1 in X_pos[0,:,:,:] and y_pos[0] == -1
    X_pos = X_pos[1:,:,:,:].astype(np.float32)
    y_pos = y_pos[1:].astype(np.int32)
    
    assert -1 in X_neg[0,:,:,:] and y_neg[0] == -1
    X_neg = X_neg[1:,:,:,:].astype(np.float32)
    y_neg = y_neg[1:].astype(np.int32)
    
    return X_pos, y_pos, X_neg, y_neg



# DEFINE USER PARAMETERS FOR DATA GENERATION, NETWORK AND FOR REAL CASE SEGMENTATION
half_patch_size = 5 # it will produce a patch having size (11,11,11). It can be view as (1+5+5,1+5+5,1+5+5)
root_folder_data = "../LabelMaps"
save_patches = True
storage_patches_file = "../patches.npz"
load_or_generate_data = "generate"

num_epochs = 550
batchsize = 32
train_or_load_network = "train"
save_network = True
network_pars_file = '../network_parms_needle.npz'

real_case_to_segment = '../LabelMaps/64/case.nrrd'
output_MRI = '../test_pat.nrrd'
output_label = '../test_label.nrrd'


# GENERATE OR LOAD PATCHES
if load_or_generate_data == "generate":
    X_pos, y_pos, X_neg, y_neg = data_for_CNN(root_folder_data, half_patch_size=half_patch_size)
elif load_or_generate_data == "load"
    saved_data = np.load(storage_patches_file)
    X_pos, y_pos, X_neg, y_neg = saved_data['arr_0'], saved_data['arr_1'], saved_data['arr_2'], saved_data['arr_3']


# SAVE PATCEHS
if save_patches:
    np.savez(storage_patches_file, X_pos, y_pos, X_neg, y_neg)


# GENERATE TRAINING, TESTING AND VALIDATION X AND y
X_train, X_val, X_test, y_train, y_val, y_test = join_Xy_posneg(X_pos, y_pos, X_neg, y_neg, test_size1=0.20, test_size2=0.33)


# NORMALIZE DATASET
m, s = X_train.mean(), X_train.std()

X_train -= m
X_train /= s

X_val -= m
X_val /= s

X_test -= m
X_test /= s

print(X_train.shape, X_test.shape, X_val.shape)

def build_cnn(single_entry_shape, input_var=None):
    """
    Build a CNN
    """
    
    network = lasagne.layers.InputLayer(shape=(None, single_entry_shape[0], single_entry_shape[1], 
                                               single_entry_shape[2]),
                                        input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=480, filter_size=(3, 3),  #120
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.HeNormal())
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=240, filter_size=(2, 2), #120
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
    network = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(network, p=.5),
            num_units=120, #120*2*2
           nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
    network = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(network, p=.5),
            num_units=60,
           nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# Prepare Theano variables for inputs and targets
single_entry_shape = X_train.shape[1:]
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Build network
network = build_cnn(single_entry_shape, input_var)

if train_or_load_network == "train":
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) # multiclass_hinge_loss
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                        dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        now = datetime.datetime.now()
        print("Epoch %d started on %s" % (epoch + 1, now.strftime("%Y-%m-%d %H:%M")))
    
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

elif train_or_load_network == "train":
    with np.load(network_pars_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

# SAVE NETWORK
if save_network:
    np.savez(network_pars_file, *lasagne.layers.get_all_param_values(network))

# REAL CASE SEGMENTATION
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

test_pat = get_resized_img(sitk.ReadImage(real_case_to_segment))
test_pat = test_pat[140:,15:180,80:230] # crop the image in order to speed up the process
test_pat = pad_volume(test_pat, half_patch_size=half_patch_size)

pat_sitk = sitk.GetImageFromArray(test_pat.astype(np.float32))
sitk.WriteImage(pat_sitk, output_MRI)

final_label = np.zeros_like(test_pat)
test_pat.shape

z_size, x_size, y_size = test_pat.shape

print("Slices to classify = %d" % (z_size))
for z in xrange(half_patch_size, z_size-half_patch_size):
    print(z - half_patch_size + 1),
    for x in xrange(half_patch_size, x_size-half_patch_size):
        for y in xrange(half_patch_size, y_size-half_patch_size):
            patient_patch_img = test_pat[z-half_patch_size:z+half_patch_size+1,x-half_patch_size:x+half_patch_size+1,y-half_patch_size:y+half_patch_size+1]
            patient_patch = patchimg2differentview(patient_patch_img).reshape((1,3*((2*half_patch_size)+1),(2*half_patch_size)+1,(2*half_patch_size)+1)).astype(np.float32)
            patient_patch -= m
            patient_patch /= s
            predicted_label = int(predict_fn(patient_patch)[0])
            
            final_label[z,x,y] = predicted_label

final_label_sitk = sitk.GetImageFromArray(final_label)
sitk.WriteImage(final_label_sitk, output_label)

print("Done!")

