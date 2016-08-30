#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, os
import  numpy as np
import pandas as pd
import pickle
from scipy import misc
from skimage import io
from skimage import color
from skimage import util
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import colorsys

from PIL import Image
from skimage.segmentation import slic, felzenszwalb, quickshift, mark_boundaries

def fun_get_image_rgb_lab_hsv_value(image_path):
    """The function should:
        - load an image according to its path
        - select part of it
        - do conversiont to lab
        - compute average rgb and lab and return this info.
    """
    try:
        # check file extension
        #ext = os.path.splitext(image_path)[-1].lower()
        im_rgb = misc.imread(image_path)

        # check is the image is mono channel
        if len(np.shape(im_rgb)) == 2:
            tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3), dtype=im_rgb.dtype)
            tmp[:,:,0] = im_rgb
            tmp[:,:,1] = im_rgb
            tmp[:,:,2] = im_rgb

            im_rgb = tmp

        # check if the image is an RGBA in which case we don't want the A channel
        if np.shape(im_rgb)[2] == 4:
            tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3))
            tmp = im_rgb[:,:,0:3]
            im_rgb = tmp

        # select the important area
        im_rgb = fun_select_image_area(im_rgb)
        im_lab = color.rgb2lab(im_rgb)
        im_hsv = color.rgb2hsv(im_rgb)

        r = np.mean(im_rgb[:,:,0])
        g = np.mean(im_rgb[:,:,1])
        b = np.mean(im_rgb[:,:,2])

        CIE_L = np.mean(im_lab[:,:,0])
        CIE_a = np.mean(im_lab[:,:,1])
        CIE_b = np.mean(im_lab[:,:,2])

        h = np.mean(im_hsv[:,:,0])
        s = np.mean(im_hsv[:,:,1])
        v = np.mean(im_hsv[:,:,2])

    except IOError:
        r,g,b, CIE_L, CIE_a, CIE_b, h, s, v = "Nan", "Nan", "Nan", "Nan", "Nan", "Nan","Nan", "Nan", "Nan"
        print image_path

    res = [r,g,b, CIE_L, CIE_a, CIE_b, h ,s, v]

    return res

def fun_get_image_rgb_lab_value(image_path):
    """The function should:
        - load an image according to its path
        - select part of it
        - do conversiont to lab
        - compute average rgb and lab and return this info.
    """
    try:
        # check file extension
        #ext = os.path.splitext(image_path)[-1].lower()
        im_rgb = misc.imread(image_path)

        # check is the image is mono channel
        if len(np.shape(im_rgb)) == 2:
            tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3), dtype=im_rgb.dtype)
            tmp[:,:,0] = im_rgb
            tmp[:,:,1] = im_rgb
            tmp[:,:,2] = im_rgb

            im_rgb = tmp

        # check if the image is an RGBA in which case we don't want the A channel
        if np.shape(im_rgb)[2] == 4:
            tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3))
            tmp = im_rgb[:,:,0:3]
            im_rgb = tmp

        # select the important area
        im_rgb = fun_select_image_area(im_rgb)
        im_lab = color.rgb2lab(im_rgb)

        r = np.mean(im_rgb[:,:,0])
        g = np.mean(im_rgb[:,:,1])
        b = np.mean(im_rgb[:,:,2])

        CIE_L = np.mean(im_lab[:,:,0])
        CIE_a = np.mean(im_lab[:,:,1])
        CIE_b = np.mean(im_lab[:,:,2])

    except IOError:
        r,g,b, CIE_L, CIE_a, CIE_b = "Nan", "Nan", "Nan", "Nan", "Nan", "Nan"
        print image_path

    res = [r,g,b, CIE_L, CIE_a, CIE_b]

    return res

def fun_load_image(image_path):
    """The function loads the image and return a numpy array of m x n x 3
    even if the image is single channel of RGBA 4 channels.
    """
    # check file extension
    im_rgb = misc.imread(image_path)

    # check is the image is mono channel
    if len(np.shape(im_rgb)) == 2:
        tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3), dtype=im_rgb.dtype)
        tmp[:,:,0] = im_rgb
        tmp[:,:,1] = im_rgb
        tmp[:,:,2] = im_rgb

        im_rgb = tmp

    # check if the image is an RGBA in which case we don't want the A channel
    if np.shape(im_rgb)[2] == 4:
        tmp = np.zeros((np.shape(im_rgb)[0], np.shape(im_rgb)[1], 3))
        tmp = im_rgb[:,:,0:3]
        im_rgb = tmp

    return im_rgb


def fun_get_average_color(image_data):
    """The funtion return the average color of an image.
    Input:
        image: numpy array

    output:
        rgb_av: numpy array
    """
    r_mean = np.mean(image_data[:,:,0])
    g_mean = np.mean(image_data[:,:,1])
    b_mean = np.mean(image_data[:,:,2])

    return r_mean, g_mean, b_mean

def fun_display_one_image_and_average_color(image_data):
    """The function does its name says
    """
    # to avoid problem with rgba images
    im = image_data

    # # # select the center and "interesting" part of the image
    im_crop = fun_select_image_area(im)

    # replace top left image corner by the area value cropped
    im[0:np.shape(im_crop)[0],0:np.shape(im_crop)[1],:] = im_crop
    im_patch = np.zeros((np.shape(im_crop)))
    for i in np.arange(np.shape(im)[2]):
        #print i,
        im_patch[:,:,i] = np.mean(im_crop[:,:,i])

    # replace the bottom right of the image by the mean value of the cropped area
    im[-np.shape(im_crop)[0]:,-np.shape(im_crop)[1]:,:] = im_patch

    # # # select bottom left of the image
    im_crop_bl = im[-np.shape(im_crop)[0]:,0:np.shape(im_crop)[1],:]

    # get average value of the botton left
    im_patch_tr = np.zeros(np.shape(im_crop_bl))
    for i in np.arange(np.shape(im)[2]):
        im_patch_tr[:,:,i] = np.mean(im_crop_bl[:,:,i])

    # replace the top right by the bottom left average
    im[0:np.shape(im_crop_bl)[0],-np.shape(im_crop_bl)[1]:,:] = im_patch_tr


    plt.imshow(im)
    plt.draw()
    #print np.shape(im)
    return (im_crop[:,:,0],im_crop[:,:,1],im_crop[:,:,2])


def fun_select_image_area(image_data):
    """The function select idealy the area whith information in it.

        Basically I'm defining a grid and take only the center as important area.
    """

    ss = np.shape(image_data)
    h = np.uint(np.linspace(0,ss[0],6))
    v = np.uint(np.linspace(0,ss[1],6))
    image_data_area = image_data[h[2]:h[3],v[2]:v[3],:]
    #image_data_area = image_data[h[1]:h[4],v[1]:v[4],:]

    return image_data_area

def create_3D_grid(x):
    """ Create a 3D grid from a vector of data in one dimension.
    The idea is to do someting equivalent as meshgrid, but for 3D and
    not only 2D.
    Args:
        x (float or [floats]): vector points
    Output:
        u, v, w (float or [floats]): coords
    """

    [u, v] = np.meshgrid(x, x)
    w = np.tile(np.ones(np.shape(u)),((np.size(x)),1))
    u = np.tile(np.reshape(u,(np.size(u),1)),(np.size(x),1))
    v = np.tile(np.reshape(v,(np.size(v),1)),(np.size(x),1))

    for ii in np.arange(0,np.size(x)):
        block = w[:,ii] * x[ii]
        w[: ,ii] = np.transpose(block)

    w = np.reshape(np.transpose(w),(np.size(u),1))

    return u, v, w

def fun_compare_colorsegmentation_and_display(image_data, number_segments=250, compactness_factor=10):
    """
    The function is a copy of what does this link http://scikit-image.org/docs/dev/auto_examples/plot_segmentations.html
    """
    segments_fz = felzenszwalb(image_data, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(image_data, n_segments=number_segments, compactness=compactness_factor, sigma=1)
    segments_quick = quickshift(image_data, kernel_size=3, max_dist=6, ratio=0.5)

    print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    fig.set_size_inches(8, 3, forward=True)
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

    ax[0].imshow(mark_boundaries(image_data, segments_fz,color=(1, 0, 0)))
    ax[0].set_title("Felzenszwalbs's method")
    ax[1].imshow(mark_boundaries(image_data, segments_slic,color=(1, 0, 0)))
    ax[1].set_title("SLIC")
    ax[2].imshow(mark_boundaries(image_data, segments_quick,color=(1, 0, 0)))
    ax[2].set_title("Quickshift")
    for a in ax:
        a.set_xticks(())
        a.set_yticks(())
    plt.show()

    #img = data_rgb

def fun_slic_and_display(image_data, number_segments=250, compactness_factor=10):
    """
    THe function applies the SLIC algorithm to an image and display in false and clustered color
    the different regions.
    """
    # now we get the colors from the slic information
    segments_seg = slic(image_data, n_segments = number_segments, compactness =compactness_factor, sigma=1, convert2lab=True)

    segments_colored = np.ones(image_data.shape)

    im_average_segment = fun_get_average_color_image_from_segmemted_image(image_data, segments_seg)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(mark_boundaries(image_data, segments_seg,color=(0, 0, 0)))
    plt.subplot(1,3,2)
    plt.imshow(segments_seg, cmap=plt.get_cmap('rainbow'))
    plt.subplot(1,3,3)
    plt.imshow(im_average_segment)

def fun_get_average_color_image_from_segmemted_image(image_data, segments_seg):
    """
    The function takes the image and its segmented version, it givec as output the image with the average color value per region.
    """
    #num_segments = np.max(segments_seg) +1
    num_segments = len(np.unique(segments_seg))
    segments = {}

    image_average_segment = np.empty(np.shape(image_data))

    # look by segments number
    for ii in range(len(np.unique(segments_seg))):
        (rows,cols)  = np.nonzero(segments_seg == ii )

        curr_values  = np.empty([len(rows),3])
        for idx in range(len(rows)):
            # take the rgb value per index N
            curr_values[idx,:] = image_data[rows[idx],cols[idx],:]

        av_rgb = np.mean(curr_values, axis = 0)

        image_average_segment[rows,cols,0] = av_rgb[0]# / 255.
        image_average_segment[rows,cols,1] = av_rgb[1]# / 255.
        image_average_segment[rows,cols,2] = av_rgb[2]# / 255.

    return image_average_segment

def fun_display_img_hsv(image_data):
    """
    The function display the image and a representation in hsv space to its side.
    """
    # conversion to hsv
    image_hsv = color.rgb2hsv(image_data)
    #print np.shape(image_hsv), image_hsv.dtype
    h = image_hsv[:,:,0].flatten()
    s = image_hsv[:,:,1].flatten()
    v = image_hsv[:,:,2].flatten()
    rgb_ = np.transpose(np.vstack([image_data[:,:,0].flatten(),
                                   image_data[:,:,1].flatten(),
                                   image_data[:,:,2].flatten()]))
    print rgb_.dtype, h.dtype, np.shape(rgb_)

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    # plot the original image
    ax1.imshow(image_data)
    ax1.axis('off')

    # conversion to polar coordinates
    xx = v * np.sin(2 * h * np.pi)
    yy = v * np.cos(2 * h * np.pi)
    ax2.scatter(xx, yy, s=100*v, facecolors=rgb_, marker='o')
    ax2.axis([-1.1,1.1,-1.1,1.1])
    ax2.grid('on')

    ax3.scatter(h * 360, v , s=100*v, facecolors=rgb_, marker='o')
    ax3.axis([-10, 370, -0.1, 1.1])
    ax3.grid('on')
    ax3.set_xlabel('h')
    ax3.set_ylabel('v')


    plt.show()

def fun_display_img_hsv2(image_data, hist):
    """
    The function display the image and a representation in hsv space to its side.

    This function is like the one above but with an hist as input parameter in bonus.
    """
    # conversion to hsv
    image_hsv = color.rgb2hsv(image_data)
    #print np.shape(image_hsv), image_hsv.dtype
    h = image_hsv[:,:,0].flatten()
    s = image_hsv[:,:,1].flatten()
    v = image_hsv[:,:,2].flatten()
    rgb_ = np.transpose(np.vstack([image_data[:,:,0].flatten(),
                                   image_data[:,:,1].flatten(),
                                   image_data[:,:,2].flatten()]))
    print rgb_.dtype, h.dtype, np.shape(rgb_)

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    # plot the original image
    ax1.imshow(image_data)
    ax1.axis('off')

    # conversion to polar coordinates
    xx = v * np.sin(2 * h * np.pi)
    yy = v * np.cos(2 * h * np.pi)
    hist = 100
    ax2.scatter(xx, yy, s=hist*1000, facecolors=rgb_, marker='o')
    ax2.axis([-1.1,1.1,-1.1,1.1])
    ax2.grid('on')

    ax3.scatter(h * 360, v , s=hist*1000, facecolors=rgb_, marker='o')
    #ax3.scatter(h * 360, v , s=100*v, facecolors=rgb_, marker='o')
    ax3.axis([-10, 370, -0.1, 1.1])
    ax3.grid('on')
    ax3.set_xlabel('h')
    ax3.set_ylabel('v')


    plt.show()

def fun_display_img_hsv_cluster(image_data, image_labels, hist, data):
    """
    The function tries to combine all previous visualization in once.
    """
    ax1 = plt.subplot2grid((4,3),(0,0))
    ax2 = plt.subplot2grid((4,3),(0,1))
    ax3 = plt.subplot2grid((4,3),(0,2))
    ax4 = plt.subplot2grid((4,3),(1,0), colspan = 3)
    ax5 = plt.subplot2grid((4,3),(2,0), colspan = 3)
    ax6 = plt.subplot2grid((4,3),(3,0), colspan = 3)

    ax1.imshow(image_data)
    ax1.axis('off')

    img_reconstruct = fun_get_average_color_image_from_segmemted_image(image_data, image_labels)
    ax3.imshow(img_reconstruct)
    ax3.axis('off')

    imtest1 = fun_create_image_from_cluster(data)
    imtest2 = fun_create_image_from_cluster_with_hist(data, hist)
    print np.shape(imtest1),np.shape(imtest2)
    ax5.imshow(imtest1)
    ax5.axis('off')
    ax6.imshow(imtest2)
    ax6.axis('off')
    plt.show()


def fun_display_half_img_slic(im1, im2, alpha = 0.5):
    """
    The function display an image made of half original half segmented.

    It is assumed that im1 and im2 have the same size of course.
    """

    im1[:,np.round(np.shape(im1)[1] * alpha):,:] = im2[:,np.round(np.shape(im1)[1] * alpha):,: ]

    return im1

def fun_create_image_from_cluster(data_rgb_cluster):
    """
    The function takes the data_rgb_cluster which are ideally an n x 3 numpy array.

    It reshapes the data to get and image representatio in order to display it.

    We limit the number of cluster to 8, for the images we have it seems to be not
    necessary to have more.
    """

    output_image_size = (100,800,3)

    # increase the size if there is less than 8 values
    data = np.zeros((8,3))
    if np.shape(data_rgb_cluster)[0] < 8:
        data[0:np.shape(data_rgb_cluster)[0],:] = data_rgb_cluster
    else:
        data = data_rgb_cluster[0:np.shape(data)[0],:]

    # reorder the lut cluster thing
    cc_rgb = data#np.random.random((8,3))
    cc_rgb = np.reshape(cc_rgb,(1,8,3))
    cc_hsv = color.rgb2hsv(cc_rgb)
    cc_h = cc_hsv[:,:,0].flatten()
    index_h = np.argsort(cc_h)

    img_cluster = []
    index_h = index_h[::-1]
    for ii in np.arange(8):
        #print ii, data[ii,:]

        # create the color patch
        color_patch = np.empty((100,100,3))
        color_patch[:,:,0] = data[index_h[ii],0]
        color_patch[:,:,1] = data[index_h[ii],1]
        color_patch[:,:,2] = data[index_h[ii],2]

        if len(img_cluster) == 0:
            img_cluster = color_patch
        else:
            img_cluster = np.hstack((img_cluster, color_patch))

    return img_cluster

def fun_create_image_from_cluster_with_hist(data_rgb_cluster, hist_cluster):
    """
    The function takes the data_rgb_cluster which are ideally an n x 3 numpy array.

    It reshapes the data to get and image representatio in order to display it.

    It used the hist_cluster that tells us the percentage of color in the image. We
    use this information to modulate the size of the patches.

    We limit the number of cluster to 8, for the images we have it seems to be not
    necessary to have more.
    """

    output_image_size = (100,800,3)

    # increase the size if there is less than 8 values
    data = np.zeros((8,3))
    if np.shape(data_rgb_cluster)[0] < 8:
        data[0:np.shape(data_rgb_cluster)[0],:] = data_rgb_cluster
    else:
        data = data_rgb_cluster[0:np.shape(data)[0],:]

    # reorder the lut cluster thing
    cc_rgb = data
    cc_rgb = np.reshape(cc_rgb,(1,8,3))
    cc_hsv = color.rgb2hsv(cc_rgb)
    cc_h = cc_hsv[:,:,0].flatten()
    index_h = np.argsort(cc_h)

    # create the final image
    hist_cluster = 800* np.round(hist_cluster*100)/100
    img_cluster = []
    index_h = index_h[::-1]
    for ii in np.arange(8):
        #print ii, data[ii,:], hist_cluster[ii]

        # create the color patch
        color_patch = np.empty((100,int(hist_cluster[index_h[ii]]),3))
        color_patch[:,:,0] = data[index_h[ii],0]
        color_patch[:,:,1] = data[index_h[ii],1]
        color_patch[:,:,2] = data[index_h[ii],2]

        if len(img_cluster) == 0:
            img_cluster = color_patch
        else:
            img_cluster = np.hstack((img_cluster, color_patch))

    return img_cluster

def fun_display_category_distribution(unique_item_count, unique_item_name, limit_size_for_training=1000, item_name= "category"):
    """
    The function display the number of item per category with an arrow pointing to
    some of the category names.
    """

    # display the categories
    #plt.figure()

    aa = np.argsort(unique_item_count)
    aa = aa[::-1]

    unique_item_count = np.asarray(unique_item_count)
    nb_item_above_limit = len(unique_item_count[unique_item_count > limit_size_for_training])
    nb_item_above_limit = np.round(nb_item_above_limit * 1.5)

    vec_x = np.linspace(1, len(unique_item_count)-1, nb_item_above_limit)
    vec_y = np.linspace(unique_item_count[aa[0]], unique_item_count[aa[-1]], nb_item_above_limit)
    #plt.plot(vec_x, vec_y,':.r')

    plt.plot(np.asarray(unique_item_count)[aa],'.-')
    plt.hlines(limit_size_for_training, 0, len(aa))

    for ii in np.arange(len(aa)):
        if unique_item_count[aa[ii]] > limit_size_for_training:
            the_text = str(unique_item_name[aa[ii]]) +" ("+str(unique_item_count[aa[ii]])+")"
            plt.annotate(str(the_text),
                         xy = (ii, unique_item_count[aa[ii]]),
                         xytext = (vec_x[ii], vec_y[ii]),
                         arrowprops = dict(facecolor='blue',shrink=0.02, width=1))

    plt.xlim([0,len(aa)])
    plt.ylabel("item occurence")
    plt.xlabel(item_name)
    plt.title('Number of item per '+item_name)
    plt.draw()

'''def fun_display_brand_distribution(unique_item_count, unique_item_name):
    """
    The function displays the number of item per brand.

    Highlight the name of the fist 20 brands.
    """
    plt.figure()
    #vec_pos = np.linspace(int(np.max(unique_item_count) * 1.1),
    #                      int(unique_item_count * 0.9), 20)

    aa = np.argsort(unique_item_count)
    aa = aa[::-1]

    plt.plot(np.asarray(unique_item_count)[aa],'.:')
    #plt.hlines(limit_size_for_training, 0, len(aa))

    #plt.plot([0, len(unique_item_count)],[unique_item_count[0], unique_item_count[-1]],color='k',linestyle='-',linewidht=2)
    #for ii in np.arange(20):
#        plt.annotate(unique_item_name[aa[ii]],
#                    xy = (0, unique_item_count[aa[ii]]),#
#                    xytext = (len(aa)/2, vec_pos[ii]),
#                    arrowprops=dict(facecolor='blue',shrink=0.02, width=1))
    plt.xlim([0,len(aa)])
    plt.xlabel("Unique brand")
    plt.ylabel("Occurence per brand")
'''

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    The function displays the confusion matrix cm as an image.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(target_names))
    #print tick_marks
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names, rotation=-45, va='bottom')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def fun_display_category_in_2D(X, Y, category_name, display_text = False, title = "a beautiful plot"):
    """
    The function display the categories in a 2 dimensional plot.
    """

    #plt.figure()
    # color-blind-friendly palette
    for x, y, name in zip(X, Y, category_name):
        color = 'blue' if "M" in name else 'red'
        marker = "v" if "M" in name else "^"
        plt.scatter(x, y, c=color)# markers=marker)

        if display_text == True:
            plt.text(x, y+0.005, name, horizontalalignment='center',
                                 verticalalignment='bottom')

        if name == "uncategorized":
            plt.scatter(x, y, marker='s',facecolor = 'magenta')
            plt.text(x,y,name, horizontalalignment='center',
                               verticalalignment='bottom',
                               color = 'magenta')

    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title(title)
    plt.axis([-1, 1, -1, 1])
    plt.draw()

def fun_display_category_in_triang2D(X, Y, category_name, title = "aNother beautiful plot"):
    """
    The function does almost list the one called fun_display_category_in_2D
    except that it uses some triangulation to show the data.
    """
    names_gender = []
    for name in category_name:
        names_gender.append(name[0])

    indices_M = [i for i, s in enumerate(names_gender) if s[0]=='M']
    indices_F = [j for j, s in enumerate(names_gender) if s[0]=='F']

    indices_M = np.asarray(indices_M)
    indices_F = np.asarray(indices_F)

    #plt.figure()

    triang = tri.Triangulation(X[indices_M], Y[indices_M])
    # Mask off unwanted triangles.
    xmid = X[triang.triangles].mean(axis=1)
    ymid = Y[triang.triangles].mean(axis=1)
    plt.triplot(triang, 'bs-')

    triang = tri.Triangulation(X[indices_F], Y[indices_F])
    # Mask off unwanted triangles.
    xmid = X[triang.triangles].mean(axis=1)
    ymid = Y[triang.triangles].mean(axis=1)
    plt.triplot(triang, 'or-')

    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title('another beautiful plot')
    plt.axis([-1, 1, -1, 1])
    plt.draw()
