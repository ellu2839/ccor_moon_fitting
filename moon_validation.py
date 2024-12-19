import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
import netCDF4 as nc
from astropy.coordinates import Distance, SkyCoord, solar_system_ephemeris, get_body, EarthLocation
from astropy.wcs.utils import skycoord_to_pixel
from astropy.constants import au
import xarray as xr
import os
from astropy.wcs import WCS
import astropy
import sunpy
import sunpy.map
from astropy.time import Time
import astropy.units as u
from astropy.time import TimeDelta
from sunpy.net import attrs as a
from sunpy.net import hek
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.time import parse_time
from astropy.coordinates import SkyCoord, CylindricalRepresentation
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a
from astropy.io import fits
from astropy import constants
from reproject import reproject_interp
from PIL import Image
import matplotlib.cm as cm
import math
import glob
from astroquery.vizier import Vizier
from astroquery.jplhorizons import Horizons
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com, centroid_2dg
from scipy.ndimage import gaussian_filter
from skimage.measure import label
import netCDF4 as nc
from skimage.draw import disk
import imageio
from skimage.feature import canny
import cv2
import math

CCOR_FOV = 5.5

def get_body_distance(date, body1, body2):
    b1 = get_body(body1, date).cartesian
    if body2 == 'ccor':
        b2 = ccor_map.observer_coordinate.cartesian
        print(b2)
    else:
        b2 = get_body(body2, date).cartesian
    distance = (b1 - b2).norm()

    return distance.to(u.AU)



def detect_moon_edges(fits_file, ext=1, sigma1=3, sigma2=15, canny_low=0, canny_high=300):
    img_data = fits.getdata(fits_file, ext)
    img_data = np.nan_to_num(img_data, nan=0.0, posinf=np.max(img_data), neginf=np.min(img_data))
    
    # normalize to [0, 255] for opencv
    img_normalized = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))).astype(np.uint8)
    
    # Contrast Limited Adaptive Histogram Equalization (better for low contrast images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_normalized)
    
    # Gaussian blur for feature enhancement
    blur1 = cv2.GaussianBlur(img_clahe, (sigma1, sigma1), 0)
    blur2 = cv2.GaussianBlur(img_clahe, (sigma2, sigma2), 0)
    edge_enhanced = cv2.subtract(blur1, blur2)
    edge_enhanced = cv2.normalize(edge_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Canny edge detection
    edges = cv2.Canny(edge_enhanced, canny_low, canny_high)
    
    return edges

def find_moon(edges, max_radius=60, min_radius=30):
    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    structures = []
    filtered_contours = []

    # only look for contours between 30-60pix, otherwise it'll pick up pylon stuff
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if min_radius <= radius <= max_radius:
            structures.append((int(x), int(y), int(radius)))
            filtered_contours.append(contour)

    return structures, filtered_contours

def find_moon_centroid(edges):
    structures, filtered_contours = find_moon(edges, max_radius=60, min_radius=40)

    if not structures:
        print(f"no structures found within the 40 - 60 pixel range")
        return None, None, None

    # get contour by max area
    max_contour = max(filtered_contours, key=cv2.contourArea)
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(max_contour)

    return int(circle_x), int(circle_y), int(radius)


with open(f"SMN_ANGLE_assessment.log", "w") as f:
    for file in sorted(glob.glob('/Users/elysia.lucas/Downloads/CCOR_0B_20241002/*.fits')):
        print(file)
        f.write(file)
        f.write('\n')

        # edge-detect moon image processing
        edges = detect_moon_edges(file)
        centroid_x, centroid_y, moon_radius = find_moon_centroid(edges)
        # skip images where edge detection doesn't pick up the moon
        if moon_radius == None:
            continue

        # get ccor map for plotting later
        data, header = fits.getdata(file, ext=1), fits.getheader(file, 1)
        ccor_map = sunpy.map.Map(data, header)


        solar_system_ephemeris.set('jpl') 
        img_scale  = [800, 5000]

        fit_manually = True
        show_roi     = False
        make_diff    = False
        wcskey       = ''

        radius_bounds  = [230.,1100.]     # inner (=pylon) and outer (=vignetting) radii in pixels for masking everything that is not considered data
        occulter_pos   = [1007, 973]      # center of the occulter (by eye)
        vignetting_pos = [1030, 960]      # center of the vignetting circle (by eye)
        pylon_pos_x    = [930, 1075]      # Pylon lower and upper x-positions (by eye)

        fov_radius = 5.5                  # field of view in degrees to search for stars
        sigma = 3.                        # sigma level that the star signal needs to be above the nearby background
        box_size = 11                     # the box size around the star
        gmag = 7                          # lower brightness limit for the stars
        circle_size = 10        

        # Convert EME2000 coordinates to a Heliographic Stonyhurst observer coordinate
        xyz = [header['EPHVEC_X'],header['EPHVEC_Y'],header['EPHVEC_Z']]
        cartrep = astropy.coordinates.CartesianRepresentation(*xyz, unit=u.m)
        gcrs = astropy.coordinates.GCRS(cartrep, obstime=header['DATE-OBS'])
        hgs_oc = gcrs.transform_to(sunpy.coordinates.HeliographicStonyhurst(obstime=header['DATE-OBS']))

        header['HGLN_OBS'] = hgs_oc.lon.value
        header['HGLT_OBS'] = hgs_oc.lat.value
        header['DSUN_OBS'] = hgs_oc.radius.value

        # Chris' new best fit parameters as of 12/18/24
        best_fit_param = [1.138, 41.59, 13.55, -6.473e-07, 1.001, -5.943e-03, 0.9077]
        rot_angle_from_pc_matrix = np.rad2deg(np.arcsin(header['PC2_1']))
        rot_angle = rot_angle_from_pc_matrix + best_fit_param[0]
        pc1_1 =  np.cos(rot_angle * u.deg) 
        pc1_2 = -np.sin(rot_angle * u.deg)
        pc2_1 =  np.sin(rot_angle * u.deg)
        pc2_2 =  np.cos(rot_angle * u.deg)
        for k, v in header.items():
           if k.startswith('CRPIX1'):
               header[k] = header[k] + best_fit_param[1]
           elif k.startswith('CRPIX2'):
               header[k] = header[k] + best_fit_param[2]
           elif k.startswith('PC1_1'):
               header[k] = pc1_1.value
           elif k.startswith('PC1_2'):
               header[k] = pc1_2.value
           elif k.startswith('PC2_1'):
               header[k] = pc2_1.value
           elif k.startswith('PC2_2'):
               header[k] = pc2_2.value
           elif k.startswith('PV2_0'):
               header[k] = best_fit_param[3]
           elif k.startswith('PV2_1'):
               header[k] = best_fit_param[4]
           elif k.startswith('PV2_2'):
               header[k] = best_fit_param[5]
           elif k.startswith('PV2_3'):
               header[k] = best_fit_param[6]

        # Fix the units for what astropy expects
        for k, v in header.items():
            if k.startswith('CUNIT') and v.startswith('deg'):
                header[k] = 'degree'
                    
        # Alter the header if we want to use anything but the first WCS system
        if wcskey != '':
            keys_to_alter = ['WCSAXES','CDELT1','CDELT2','CUNIT1','CUNIT2','CRVAL1','CRVAL2',
                             'CRPIX1','CRPIX2','CTYPE1','CTYPE2','WCSNAME','PC1_1', 'PC1_2',
                             'PC2_1', 'PC2_2', 'PV1_1', 'PV1_2', 'PV1_3', 'PV2_0', 'PV2_1', 
                             'PV2_2', 'PV2_3']
            for key in keys_to_alter:
                header[key] = header[key+wcskey]

        ccor_map = sunpy.map.Map(data, header)

        # Get moon location
        sun_to_satellite = ccor_map.observer_coordinate.transform_to('hcrs')
        satellite_to_sun = SkyCoord(-sun_to_satellite.spherical, obstime=sun_to_satellite.obstime, frame='hcrs')

        vv = Vizier(columns=['**'], row_limit=-1, column_filters={'Gmag': '<'+str(gmag)}, timeout=1200)
        vv.ROW_LIMIT = -1
        result = vv.query_region(satellite_to_sun, radius=fov_radius*u.deg, catalog='I/345/gaia2')
        ccor_itrs = ccor_map.observer_coordinate.transform_to('itrs')
        el = EarthLocation.from_geocentric(x=ccor_itrs.x, y=ccor_itrs.y, z=ccor_itrs.z)
        obj_pos = get_body('moon', time=Time(header['DATE-OBS'], scale='utc', format='isot'), location=el)
        # moon location in pixels
        crds_obj_pix = skycoord_to_pixel(obj_pos, ccor_map.wcs, origin=0, mode='wcs')
        moon_radius = (3476/2.) * u.km
        obj_dist = obj_pos.separation_3d(ccor_map.observer_coordinate.transform_to('gcrs'))  # earth-centric coordinate system
        # moon radius in pixels
        obj_radius_pix = np.arctan2(moon_radius, obj_dist).to(u.deg).value/header['CDELT1']

        # used for SM angle calculations
        gbd_se = get_body_distance(ccor_map.date, 'sun', 'earth')
        gbd_sm = get_body_distance(ccor_map.date, 'sun', 'moon')
        gbd_em = get_body_distance(ccor_map.date, 'earth', 'moon')


        ccor_geostationary_distance = (35786. * u.km).to(u.AU)
        EM_ANGLE = 2*np.pi - header['SM_ANGLE'] - header['SN_ANGLE']

        moon_to_ccor_distance = np.sqrt((gbd_em ** 2) + (ccor_geostationary_distance ** 2) + (2 * gbd_em * ccor_geostationary_distance * np.cos(EM_ANGLE)))

        print('SM_ANGLE')
        # SM_ANGLE: is moon in FOV or not
        DSUN_OBS = (header['DSUN_OBS']*u.m).to(u.AU)
        obj_dist_au = obj_dist.to(u.AU)

        SM_ANGLE = (np.arccos((obj_dist_au**2 + DSUN_OBS**2 - gbd_sm**2)/
                             (2*obj_dist_au*DSUN_OBS))).value
        header_sm_angle = (header['SM_ANGLE']*u.rad).value

        # determine error between Chris' method and geometric method, and whether or not moon should be in FOV
        f.write('SM_ANGLE\n')
        f.write('calculated ' + str(SM_ANGLE)+ ' deg\n')
        f.write('header ' + str(header['SM_ANGLE']) + ' deg\n')
        f.write('calculated has ' + str(np.abs(SM_ANGLE - header_sm_angle)/header_sm_angle*100) + ' percent error from header\n')
        f.write('header has ' + str(np.abs(header_sm_angle - SM_ANGLE/SM_ANGLE*100)) + ' percent error from calculated\n')
        minimum_moon_angle = (header_sm_angle - header['M_RADIUS'])*u.rad.to(u.deg)
        if minimum_moon_angle < CCOR_FOV:
            f.write(f'Moon should be in FOV: ({header_sm_angle} - {header["M_RADIUS"]*u.rad.to(u.deg)}) < {CCOR_FOV}\n')
            print(f'Moon should be in FOV: ({header_sm_angle} - {header["M_RADIUS"]*u.rad.to(u.deg)}) < {CCOR_FOV}\n')
        else:
            f.write(f'Moon should NOT be in FOV: ({header_sm_angle} - {header["M_RADIUS"]*u.rad.to(u.deg)}) >= {CCOR_FOV}\n')
            print(f'Moon should NOT be in FOV: ({header_sm_angle} - {header["M_RADIUS"]*u.rad.to(u.deg)}) >= {CCOR_FOV}\n')
        f.write('\n')


        # print('SN_ANGLE')
        # SN ANGLE is earth in FOV or not
        # earth_dist_au = (35786*u.km).to(u.AU)
        # print(earth_dist_au)
        # print(DSUN_OBS)
        # print(gbd_se)
        # SN_ANGLE = ((earth_dist_au**2 + DSUN_OBS**2 - gbd_se**2) / (2 * earth_dist_au * DSUN_OBS))
        # #SN_ANGLE = np.arccos((earth_dist_au**2 + DSUN_OBS**2 - gbd_se.value**2)/
        #                      # (2*earth_dist_au*DSUN_OBS))
        # f.write('SN_ANGLE\n')
        # f.write('calculated ' + str(SN_ANGLE*u.rad.to(u.deg)) + ' deg\n')
        # f.write('header ' + str(header['SN_ANGLE']*u.rad.to(u.deg)) + ' deg\n')
        # f.write('calculated has ' + str(np.abs(SN_ANGLE*u.rad.to(u.deg) - header['SN_ANGLE']*u.rad.to(u.deg))/header['SN_ANGLE']*u.rad.to(u.deg)*100) + ' percent error from header\n')
        # f.write('header has ' + str(np.abs(header['SN_ANGLE']*u.rad.to(u.deg) - SN_ANGLE*u.rad.to(u.deg))/SN_ANGLE*u.rad.to(u.deg)*100) + ' percent error from calculated\n')
        # minimum_earth_angle = (header['SN_ANGLE']*u.rad.to(u.deg) - header['E_RADIUS'])*u.rad.to(u.deg)
        # if minimum_earth_angle < CCOR_FOV:
        #     f.write(f'Earth should be in FOV: ({header["SN_ANGLE"]*u.rad.to(u.deg)} - {header["E_RADIUS"]*u.rad.to(u.deg)}) < {CCOR_FOV}\n')
        # else:
        #     f.write(f'Earth should NOT be in FOV: ({header["SN_ANGLE"]*u.rad.to(u.deg)} - {header["E_RADIUS"]*u.rad.to(u.deg)}) >= {CCOR_FOV}\n')
        # f.write('\n')
        # f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # f.write('\n')



        # make ccor images with edge detected results
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f'CCOR L0b')
        fig.tight_layout(h_pad=-50)
        ax = fig.add_subplot(projection=ccor_map)
        ccor_map.plot(clip_interval=(0.1, 99.9)*u.percent, vmin=0.0, vmax=16000.0) # max is usually 16160-16224
        pix_distance_to_moon = math.degrees(header['SM_ANGLE']/header['CDELT1'])
        print('pix distance to moon', pix_distance_to_moon)


        # center of the sun
        plt.plot(header['CRPIX1'], header['CRPIX2'], 'mo')

        # line between center of sun and moon
        plt.plot([header['CRPIX1'], crds_obj_pix[0]], [header['CRPIX2'], crds_obj_pix[1]], 'r')

        # Center of the moon
        plt.scatter([crds_obj_pix[0]], [crds_obj_pix[1]], color="yellow", s=5, marker='o')

        # circle around moon
        location_obj_predicted = CircularAperture(crds_obj_pix, r=obj_radius_pix)
        location_obj_predicted.plot(color='gold', label='Moon loc (adjusted wcs)' if minimum_moon_angle < CCOR_FOV else None)
        ax.set_xlim([0, header['NAXIS1']])
        ax.set_ylim([0, header['NAXIS2']])

        chris_x, chris_y = crds_obj_pix

        # distance between Chris' centroid and edge detection centroid
        centroid_diff = np.sqrt((chris_x - centroid_x)**2 + (chris_y - centroid_y)**2)

        # plot image processing edge detection moon boundary
        masked_edges = np.ma.masked_where(edges <= 0.1, edges)
        plt.imshow(masked_edges, cmap="gray_r", alpha=0.7, origin='lower')
        circle = plt.Circle((centroid_x, centroid_y), obj_radius_pix, color="red", fill=False, lw=3, label="Moon loc (edge detection)")
        circle.set_linestyle('dashed')
        plt.gca().add_artist(circle)
        plt.scatter([centroid_x], [centroid_y], color="red", s=5, marker='*')

        plt.legend()

        print(f"CRPIX1: {header['CRPIX1']}, type: {type(header['CRPIX1'])}")
        print(f"CRPIX2: {header['CRPIX2']}, type: {type(header['CRPIX2'])}")
        print(f"crds_obj_pix: {crds_obj_pix}, type: {type(crds_obj_pix)}")
        print(f"obj_radius_pix: {obj_radius_pix}, type: {type(obj_radius_pix)}")
        print(f"centroid_x: {centroid_x}, type: {type(centroid_x)}")
        print(f"centroid_y: {centroid_y}, type: {type(centroid_y)}")
        print(f"moon_radius: {moon_radius}, type: {type(moon_radius)}")

        # TODO: comment this out for full image, not just centered on moon
        plt.xlim(crds_obj_pix[0] - 100, crds_obj_pix[0] + 100)
        plt.ylim(crds_obj_pix[1] - 100, crds_obj_pix[1] + 100)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if minimum_moon_angle < CCOR_FOV:
            # add text for Moon in FOV on the left side
            ax.text(x_min + 0.02*(x_max - x_min), y_min + 0.02*(y_max - y_min),
                    f'Moon in FOV: {np.round(minimum_moon_angle, 2)} deg',
                    color='red', fontsize=12, ha='left', va='bottom')

            # Calculate the centroid difference
            rcd = float(f"{centroid_diff:.3g}")

            # add text for centroid difference on the right side
            ax.text(x_max - 0.02*(x_max - x_min), y_min + 0.02*(y_max - y_min),
                    f'centroid difference: {rcd} pix',
                    color='green', fontsize=12, ha='right', va='bottom')


        plt.savefig(os.path.join('/Users/elysia.lucas/Data/CCOR/ccor_smn_images/', f'{os.path.basename(file).split(".")[0]}'))
        plt.close()



def create_mp4_from_images(directory, output_filename, original_fps=24, desired_fps=2):

    # repeat frames for lower fps
    repeat_factor = max(1, math.ceil(original_fps / desired_fps))

    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png'))])

    # read an image to determine size for saving
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape
    size = (width, height)
    
    # get the video writer
    output_path = os.path.join(directory, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, original_fps, size)

    # add repeated frames to slow down effective fps
    for file in image_files:
        image = cv2.imread(file)
        for k in range(repeat_factor):
            video_writer.write(image)

    video_writer.release()
    print(f"MP4 created: {output_path}")


create_mp4_from_images("/Users/elysia.lucas/Data/CCOR/ccor_smn_images", "/Users/elysia.lucas/Data/CCOR/ccor_moon_validation_newparams_small.mp4")

