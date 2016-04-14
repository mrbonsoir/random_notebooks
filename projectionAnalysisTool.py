import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import sys
#sys.path += ['/home/viorama/cv/bin']
#import stitcherOutput_pb2
#import stitcherAnalysisTool
import math

def fun_create_camera_matrix(fov, image_width, image_height):
    """Compute the camera matrix
    
       |fx 0  cx |  where fx = focal / width and cx = width / 2
    A  |0  fy cy |  where fy = focal / width and cy = width / 2 
       |0  0  1  |
    
    Inspired from cpp code:
    float_t focal = 0.5 / tanf((fov*(M_PI/180))/2);
    
    K(0,0) = focal*cameraImageSize.width;
    K(1,1) = focal*cameraImageSize.width;
    K(0,2) = (float_t)cameraImageSize.width / 2.0;
    K(1,2) = (float_t)cameraImageSize.height / 2.0;
    
    """ 
    K = np.eye(3)
    
    focal = 2 * 0.5 * np.tan(fov * (np.pi / 180.)/2)
    
    #print 'focal: %2.4f' % focal
    K[0,0] = focal*image_width
    K[1,1] = focal*image_width
    K[0,2] = image_width / 2.0
    K[1,2] = image_height / 2.0 

    return K

def fun_create_test_points(image_width, image_height, sample_step = 5):
    """The functions creates test point in XYZ space.
    It uses the images size to generate fake points that happen to be in the 
    camera frame.
    """

    step_point = sample_step
    XX, YY = np.meshgrid(np.linspace(- image_width / 2, image_width / 2, step_point),
                         np.linspace(- image_height / 2, image_height / 2, step_point))
    data_XYZ = np.zeros((3,1+np.size(XX)))
    data_XYZ[0,0:np.size(XX.flatten())] = XX.flatten()
    data_XYZ[1,0:np.size(XX.flatten())] = YY.flatten()
    data_XYZ[0,:] = data_XYZ[0,:] + image_width / 2   # suppose to be the center of the image
    data_XYZ[1,:] = data_XYZ[1,:] + image_height / 2  #
    data_XYZ[2,:] = 1

    return data_XYZ

def fun_create_Rotation_Matrix_Xaxis(angle_value):
    """The function create a rotation matrix around the X axis
    that can be used for test."""

    R = np.eye(3)
    R[1,1] =   np.cos(angle_value)
    R[1,2] = - np.sin(angle_value)
    R[2,1] =   np.sin(angle_value)
    R[2,2] =   np.cos(angle_value)

    return R

def fun_create_Rotation_Matrix_Yaxis(angle_value):
    """The function create a rotation matrix around the Y axis
    that can be used for test."""

    R = np.eye(3)
    R[0,0] =   np.cos(angle_value)
    R[0,2] =   np.sin(angle_value)
    R[2,0] = - np.sin(angle_value)
    R[2,2] =   np.cos(angle_value)

    return R

def fun_create_Rotation_Matrix_Zaxis(angle_value):
    """The function create a rotation matrix around the Z axis
    that can be used for test."""

    R = np.eye(3)
    R[0,0] =   np.cos(angle_value)
    R[0,1] = - np.sin(angle_value)
    R[1,0] =   np.sin(angle_value)
    R[1,1] =   np.cos(angle_value)

    return R

def fun_get_Rotation_Matrix_from_Report(index_image, dir_to_inspect):
    """The function loads a configuration file and returns the computed
    rotation matrix at the given index_image."""

    run = stitcherAnalysisTool.read_result(dir_to_inspect)
    dataSequence = run[0]
    R = np.reshape(dataSequence.stitcherRun[index_image].levels[0].rotationOutput.data,(3,3)) 

    return R


def fun_denormalized(data_XYZ_):
    """The function de-normalizes the point after the cameara matrix has been applied."""
    data_XYZ_dn = np.zeros(np.shape(data_XYZ_))

    data_XYZ_dn[0,:] = data_XYZ_[0,:] / data_XYZ_[2,:] 
    data_XYZ_dn[1,:] = data_XYZ_[1,:] / data_XYZ_[2,:] 
    data_XYZ_dn[2,:] = data_XYZ_[2,:]

    return data_XYZ_dn

def fun_apply_mat_to_points(M, Data_In):
    """The function simply does a matrix multiplication M (3x3) by another matrix Data (3xn) 
    
    The matrix M can represent a rotation, projection, inverse projection
    """
    
    Data_Out = np.dot(M, Data_In)
    
    return Data_Out

def fun_display_points_in_camera_frame(data_XYZ, frame, title='Camera frame'):
    """The function display the point in the camera frame
    data_XYZ can be only of size 2 x n
    frame is an image from which we will compute the shape.
    """
    
    im_width, im_height = np.shape(frame)[1], np.shape(frame)[0]
    
    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    # highlight the image area
    ax1.add_patch(patches.Rectangle(
                    (0,0),           # (x,y)
                    im_width,        # width
                    im_height,       # height
                    alpha = 0.1))
    plt.plot(data_XYZ[0,:], data_XYZ[1,:],'.b')
    plt.xlim(0 - im_width / 8, im_width+ im_width / 8)
    plt.ylim(0 - im_height / 8 , im_height +  im_height / 8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    
def fun_display_points_in_angle_texture_map(data_theta_phi, title='projection equirectangulare in (phi,theta) coordinates'):
    """The function display the point in the camera frame
    data_XYZ can be only of size 2 x n
    frame is an image from which we will compute the shape.
    """
    
    theta = data_theta_phi[0,:]
    phi   = data_theta_phi[1,:]
    
    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Rectangle((- np.pi, - np.pi/2), # (x,y)
                                   2 * np.pi,             # width
                                       np.pi,             # height
                                       alpha = 0.1))
    plt.plot(theta, phi,'.b')
    plt.xlim(- np.pi *1.1, np.pi*1.1)
    plt.ylim(- np.pi / 2 *1.1, np.pi/2 *1.1)
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.title(title)
    
def fun_display_point_in_uv_texture_map(data_uv, pano_frame, title='Equirectangular image texture'):
    """The function display uv points in the equirectangulare image projection that corresponds to 
    the texture displayed.
    
    pano-frame can be color or grayscale image."""
    
    pano_width, pano_height = np.shape(pano_frame)[1], np.shape(pano_frame)[0]
    
    plt.figure(figsize=(12,8))
    plt.imshow(pano_frame)
    plt.plot(data_uv[0,:], data_uv[1,:],'.',color=[1,0,1],markersize=4)

    plt.xlim(0, pano_width)
    plt.ylim(0, pano_height)
    

def fun_display_displacement_in_uv_texture(data_video_sequence_uv, pano_frame, color_value=[1,0,1]):
    """The function takes a set of uv point that correspond to the displacement if the same point after 
    several frames."""

    pano_width, pano_height = np.shape(pano_frame)[1], np.shape(pano_frame)[0]
    
    plt.figure(figsize=(12,8))
    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Rectangle((0, 0),   # (x,y)
                                     pano_width,        # width
                                     pano_height,       # height
                                       alpha = 0.1))


    plt.plot(data_video_sequence_uv[0,:], data_video_sequence_uv[1,:],'-',color=color_value, markersize=4)
    plt.xlim(0 - pano_width * 0.1 , pano_width * 1.1)
    plt.ylim(0 - pano_height * 0.1 , pano_height * 1.1)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Displacement in uv')


def fun_display_two_sets_of_points_in_camera_frame(data_XYZ_a, data_XYZ_b, frame, title='ideal camera image frame'):
    """The function displays two set of points to ideally illustrate how points are back projected to the camera 
    view frame"""
    
    im_width, im_height = np.shape(frame)[1], np.shape(frame)[0]

    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Rectangle((0,0), im_width, im_height, alpha = 0.1))
    plt.plot(data_XYZ_a[0,:], data_XYZ_a[1,:],'.b')
    plt.plot(data_XYZ_b[0,:], data_XYZ_b[1,:],'or')

    for ii in np.arange(np.shape(data_XYZ_a)[1]):
            plt.plot([data_XYZ_a[0,ii], data_XYZ_b[0,ii]],
                     [data_XYZ_a[1,ii], data_XYZ_b[1,ii]],'k')

    #plt.xlim(0 - im_width / 2, im_width+ im_width / 2)
    #plt.ylim(0 - im_height / 2 , im_height +  im_height / 2)
    
    plt.xlim(0 - im_width / 8, im_width+ im_width / 8)
    plt.ylim(0 - im_height / 8 , im_height +  im_height / 8)
    plt.title('ideal camera image')


def fun_normalize_XYZ(data_XYZ):
    """The function does what it should do."""
    # get norm for each vector
    norm_data_xyz = np.linalg.norm(data_XYZ,axis=0)

    # normalize each vector
    data_xyz = np.zeros(np.shape(data_XYZ))
    data_xyz = data_XYZ / np.tile(norm_data_xyz,(3,1))
    
    return data_xyz

def fun_xyz2theta_phi(data_xyz):
    """The function speaks by its name."""
    
    theta = np.arctan(data_xyz[0,:] / data_xyz[2,:])
    phi   = np.arctan(data_xyz[1,:] / np.sqrt(data_xyz[0,:]**2 + data_xyz[2,:]**2))

    data_theta_phi = np.vstack([theta, phi])

    return data_theta_phi

def fun_theta_phi2xyz(data_theta_phi):
    """The function speaks by its name."""
    
    data_xyz_ = np.zeros((3, np.shape(data_theta_phi)[1]))
    data_xyz_[0,:] = np.sin(data_theta_phi[0,:]) * np.cos(data_theta_phi[1,:])
    data_xyz_[1,:] = np.sin(data_theta_phi[1,:])
    data_xyz_[2,:] = np.cos(data_theta_phi[0,:]) * np.cos(data_theta_phi[1,:])

    return data_xyz_


def fun_theta_phi2uv(data_theta_phi, pano_width, pano_height):
    """Once again just read the name of the function."""
    data_uv = np.zeros(np.shape(data_theta_phi))
    data_uv[0,:] = data_theta_phi[0,:] * pano_width  / (2 * np.pi) + pano_width / 2
    data_uv[1,:] = data_theta_phi[1,:] * pano_height / ( np.pi) + pano_height / 2
    
    return data_uv

def fun_uv2theta_phi(data_uv, frame_width, frame_height):
    """As above but different"""
    data_theat_phi = np.zeros((np.shape(data_uv)))
    data_theat_phi[0,:] = (data_uv[0,:] - frame_width / 2) * (2*np.pi) / frame_width
    data_theat_phi[1,:] = (data_uv[1,:] - frame_height / 2) * (np.pi / frame_height)

    return data_theat_phi
