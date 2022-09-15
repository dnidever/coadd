#!/usr/bin/env python

import os
import time
import numpy as np
import time
import shutil
from datetime import datetime
import subprocess
import tempfile
import logging
import re
from glob import glob
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.wcs import WCS
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
import warnings
from dlnpyutils import utils as dln
import struct
from itertools import zip_longest
from itertools import accumulate
from io import StringIO
from . import utils

# Ignore these warnings
warnings.simplefilter('ignore', category=AstropyWarning)
#warnings.filterwarnings(action="ignore", message=r'FITSFixedWarning:*')

def make_parser(fieldwidths,fieldtypes=None):
    # https://stackoverflow.com/questions/4914008/how-to-efficiently-parse-fixed-width-files
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool flags for padding fields
    if fieldtypes is None:
        flds = tuple(zip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one        
        slcs = ', '.join('line[{}:{}]'.format(i, j) for pad, i, j in flds if not pad)
    else:
        tdict = {'s':'str','d':'int','f':'float'}
        ftypes = [tdict[ft] for ft in fieldtypes]
        flds = tuple(zip_longest(ftypes,pads, (0,)+cuts, cuts))[:-1]  # ignore final one        
        slcs = ', '.join('{}(line[{}:{}])'.format(ftype, i, j) for ftype, pad, i, j in flds if not pad)        
    parse = eval('lambda line: ({})\n'.format(slcs))  # Create and compile source code.
    # Optional informational function attributes.
    parse.size = sum(abs(fw) for fw in fieldwidths)
    if fieldtypes is None:
        parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                   for fw in fieldwidths)
    else:
        parse.fmtstring = ' '.join('{}{}'.format(a[0],a[1]) for a in zip(fieldwidths,fieldtypes))
    return parse

def fileinfo(files):
    """
    Gather information about files like their corner coordinates, etc. 
    This is used by the various tiling-related programs. 
 
    Parameters
    ----------
    files : str or list
      The FITS file names. 
 
    Returns
    -------
    info : table
       The structure with information for each file. 
 
    Example
    -------

    info = fileinfo(files)
 
    By D.Nidever  Jan. 2017 
    Translated to Python by D. Nidever,  April 2022
    """

    nfiles = np.array(files).size
    if nfiles==1 and (type(files)==str or type(files)==np.str or type(files)==np.str_):
        files = [files]

    # Create structure 
    dt = [('file',(np.str,300)),('dir',(np.str,300)),('base',(np.str,100)),('ext',(np.str,10)),('exists',bool),
          ('size',int),('nx',int),('ny',int),('filter',(np.str,50)),('exptime',float),
          ('dateobs',(np.str,30)),('mjd',float),('pixscale',float),('cenra',float),('cendec',float),
          ('ra_range',(float,2)),('dec_range',(float,2)),
          ('vertices_ra',(float,4)),('vertices_dec',(float,4)),('nextend',int),('exteninfo',object)]
    info = np.zeros(nfiles,dtype=np.dtype(dt))
    # File loop 
    for i in range(nfiles): 
        info['file'][i] = files[i]
        info['dir'][i] = os.path.dirname(os.path.abspath(files[i]))
        info['base'][i] = utils.fitsext(files[i],basename=True)
        info['ext'][i] = utils.fitsext(files[i])
        info['exists'][i] = os.path.exists(files[i])
        if info['exists'][i]==False:
            continue
        hdu = fits.open(files[i])
        nhdu = len(hdu)
        info['nextend'][i] = nhdu


        ra_range = np.array([np.inf,-np.inf])
        dec_range = np.array([np.inf,-np.inf])
        extendt =  [('extnum',int),('nx',int),('ny',int),('pixscale',float),('cenra',float),('cendec',float),
                   ('vertices_ra',(float,4)),('vertices_dec',(float,4)),('wcs',object)]
        exteninfo = np.zeros(len(hdu),dtype=np.dtype(extendt))
        for j in range(nhdu):
            exteninfo['extnum'][j] = j
            nx = hdu[j].header.get('NAXIS1')
            ny = hdu[j].header.get('NAXIS2')
            if nx is not None and ny is not None and nx>0 and ny>0:
                exteninfo['nx'][j] = nx
                exteninfo['ny'][j] = ny
                wcs = WCS(hdu[j].header)
                pcoo = wcs.pixel_to_world([exteninfo['nx'][j],exteninfo['nx'][j]+1],
                                          [exteninfo['ny'][j],exteninfo['ny'][j]+1])
                pixscale = 3600*pcoo[0].separation(pcoo[1]).deg/np.sqrt(2)
                exteninfo['pixscale'][j] = pixscale 
                coo = wcs.pixel_to_world(exteninfo['nx'][j]//2,exteninfo['ny'][j]//2)
                cenra1 = coo.ra.deg
                cendec1 = coo.dec.deg
                exteninfo['cenra'][j] = cenra1 
                exteninfo['cendec'][j] = cendec1 
                vcoo = wcs.pixel_to_world([0,exteninfo['nx'][j]-1,exteninfo['nx'][j]-1,0],
                                          [0,0,exteninfo['ny'][j]-1,exteninfo['ny'][j]-1])
                vra = vcoo.ra.deg
                vdec = vcoo.dec.deg
                exteninfo['vertices_ra'][j] = vra
                exteninfo['vertices_dec'][j] = vdec
                exteninfo['wcs'][j] = wcs
                # Figure out maximum RA/DEC ranges for the full exposure
                ra_range[0] = np.min(np.hstack((ra_range[0],vra)))
                ra_range[1] = np.max(np.hstack((ra_range[1],vra)))
                dec_range[0] = np.min(np.hstack((dec_range[0],vdec)))
                dec_range[1] = np.max(np.hstack((dec_range[1],vdec)))
        info['ra_range'][i] = ra_range
        info['dec_range'][i] = dec_range        
        info['exteninfo'][i] = exteninfo
        if files[i][-7:]=='fits.fz':
            head = hdu[0].header
            #head.extend(hdu[1].header)
            ## Fix the NAXIS1/NAXIS2 in the header
            #if 'ZNAXIS1' in head:
            #    head['NAXIS1'] = head['ZNAXIS1']
            #if 'ZNAXIS2' in head:
            #    head['NAXIS2'] = head['ZNAXIS2']
        else: 
            head = hdu[0].header
        hdu.close()
        if head['NAXIS'] != 0:
            info['nx'][i] = head['NAXIS1']
            info['ny'][i] = head['NAXIS2']
        try:
            info['filter'][i] = getfilter(files[i],noupdate=True,silent=True)
        except:
            info['filter'][i] = head.get('filter')
        info['exptime'][i] = getexptime(head=head,silent=True)
        info['dateobs'][i] = head.get('date-obs')
        info['mjd'][i] = utils.date2jd(info['dateobs'][i],mjd=True)

        # Single image with no extensions
        if nhdu==1:
            wcs = WCS(head)
            pcoo = wcs.pixel_to_world([info['nx'][i],info['nx'][i]+1],[info['ny'][i],info['ny'][i]+1])
            pixscale = 3600*pcoo[0].separation(pcoo[1]).deg
            info['pixscale'][i] = pixscale 
            coo = wcs.pixel_to_world(info['nx'][i]//2,info['ny'][i]//2)
            cenra1 = coo.ra.deg
            cendec1 = coo.dec.deg
            info['cenra'][i] = cenra1 
            info['cendec'][i] = cendec1 
            vcoo = wcs.pixel_to_world([0,info['nx'][i]-1,info['nx'][i]-1,0],[0,0,info['ny'][i]-1,info['ny'][i]-1])
            vra = vcoo.ra.deg
            vdec = vcoo.dec.deg
            info['vertices_ra'][i] = vra 
            info['vertices_dec'][i] = vdec
        # Images in extensions
        else:
            cenra1 = np.mean(ra_range)
            cenra1 = np.mean(dec_range)            
            info['cenra'][i] = cenra1
            info['cendec'][i] = cendec1
            info['pixscale'][i] = np.mean(exteninfo['pixscale'])
            info['vertices_ra'][i] = [ra_range[0],ra_range[1],ra_range[1],ra_range[0]]
            info['vertices_dec'][i] = [dec_range[0],dec_range[0],dec_range[1],dec_range[1]]
            
    return Table(info)


def getgain(filename=None,head=None):
    """
    This gets the GAIN information from a FITS files 
 
    Parameters
    ----------
    filename : str, optional
       FITS filename 
    head : header, optional
      Use this header array instead of reading FITS file. 

    Returns
    -------
    gain : float
      The gain in e/ADU 
       If the gain is not found in the header then None
       is returned. 
    keyword : str
      The actual header keyword used. 
 
    Example
    -------
    
    gain,keyword = getgain(filename)
 
    By D.Nidever  May 2008 
    Translated to Python by D. Nidever, April 2022
    """

    if filename is None and head is None:
        raise ValueError('filename or head must be input')
    nfiles = dln.size(filename)
    
    # Can't use input HEAD if multiple fits files input 
    if nfiles > 1:
        head = None
     
    # More than one name input 
    if nfiles > 1:
        gain = np.zeros(nfiles,float)
        keyword = np.zeros(nfiles,(np.str,50))
        for i in range(nfiles): 
            gain1,keyword1 = getgain(filename[i])
            gain[i] = gain1
            keyword[i] = keyword1
        return gain,keyword
     
    # No header input, read from fits file 
    if head is None:
        # Check that the file exists
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        if filename[-7:]=='fits.fz':
            head = readfile(filename,1,header=True)
        else:
            head = readfile(filename,header=True)        
     
    gain = head.get('GAIN')
    egain = head.get('EGAIN')   # imacs 
     
    # Use GAIN 
    if gain is not None:
        keyword = 'GAIN' 
    # Use EGAIN 
    if gain is None and egain is not None:
        gain = egain 
        keyword = 'EGAIN' 
             
    # No GAIN 
    if gain is None and egain is None:
        print('NO GAIN FOUND')
        gain = None
        keyword = None

    return gain,keyword


def getrdnoise(filename=None,head=None):
    """
    This gets the RDNOISE information from a FITS files 
 
    Parameters
    ----------
    filename : str, optional
       FITS filename 
    head : header, optional
      Use this header array instead of reading FITS file. 

    Returns
    -------
    rdnoise : float
      The rdnoise in electrons/read
       If the rdnoise is not found in the header then None
       is returned. 
    keyword : str
      The actual header keyword used. 
 
    Example
    -------
    
    rdnoise,keyword = getrdnoise(filename)
 
    By D.Nidever  May 2008 
    Translated to Python by D. Nidever, April 2022
    """

    if filename is None and head is None:
        raise ValueError('filename or head must be input')
    nfiles = dln.size(filename)
    
    # Can't use input HEAD if multiple fits files input 
    if nfiles > 1:
        head = None
     
    # More than one name input 
    if nfiles > 1:
        rdnoise = np.zeros(nfiles,float)
        keyword = np.zeros(nfiles,(np.str,50))
        for i in range(nfiles): 
            rdnoise1,keyword1 = getrdnoise(filename[i])
            rdnoise[i] = rdnoise1
            keyword[i] = keyword1
        return rdnoise,keyword
     
    # No header input, read from fits file 
    if head is None:
        # Check that the file exists
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        if filename[-7:]=='fits.fz':
            head = readfile(filename,1,header=True)
        else:
            head = readfile(filename,header=True)        

    rdnoise = head.get('RDNOISE')                
    readnois = head.get('READNOIS')   # swope
    enoise = head.get('ENOISE')   #    imacs
    
    # Use RDNOISE
    if rdnoise is not None:
        readnoise = rdnoise
        keyword = 'RDNOISE' 
    # Use READNOIS
    if rdnoise is None and readnois is not None:
        readdnoise = readnois
        keyword = 'READNOIS'
    # Use ERDNOISE
    if rdnoise is None and readnois is None and enoise is not None:
        readnoise = enoise
        keyword = 'ENOISE' 
        
    # No RDNOISE 
    if rdnoise is None and readnois is None and enoise is None:
        print('NO READNOISE FOUND')
        readnoise = None
        keyword = None

    return readnoise,keyword


def getexptime(filename=None,head=None,silent=False):
    """
    This gets the EXPTIME information from a FITS files 
 
    Parameters
    ----------
    filename : str, optional
       FITS filename 
    head : header, optional
       Use this header array instead of reading FITS file. 
    silent : boolean, optional
       Do not print anything to the screen.

    Returns
    -------
    exptime : float
      The exposure time information in seconds.
       If the exptime is not found in the header then None
       is returned. 
 
    Example
    -------
    
    exptime = getexptime(filename)
 
    By D.Nidever  May 2008 
    Translated to Python by D. Nidever, April 2022
    """

    if filename is None and head is None:
        raise ValueError('filename or head must be input')
    nfiles = dln.size(filename)
    
    # Can't use input HEAD if multiple fits files input 
    if nfiles > 1:
        head = None
     
    # More than one name input 
    if nfiles > 1:
        exptime = np.zeros(nfiles,float)
        for i in range(nfiles): 
            exptime[i] = getexptime(filename[i])
        return exptime
     
    # No header input, read from fits file 
    if head is None:
        # Check that the file exists
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        if filename[-7:]=='fits.fz':
            head = readfile(filename,1,header=True)
        else:
            head = readfile(filename,header=True)        

    exptime = head.get('EXPTIME')                
    
    # No EXPTIME 
    if exptime is None:
        if silent==False:
            print('NO EXPTIME FOUND')

    return exptime

def getpixscale(filename,head=None):
    """
    Get the pixel scale for an image. 
 
    Parameters
    ----------
    file    FITS filename 
    =head   The image header for which to determine the pixel scale. 
    /stp    Stop at the end of the program. 
 
    Returns
    -------
    scale   The pixel scale of the image in arcsec/pix. 
    =error  The error if one occurred. 
 
    Example
    -------

    scale = getpixscale('ccd1001.fits')
 
    BY D. Nidever   February 2008 
    Translated to Python by D. Nidever,  April 2022
    """

    scale = None  # bad until proven good 

     
    # No header input, read from fits file 
    fpack = 0 
    if head is None:
        # Check that the file exists
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')

        # Open the file
        #hdu = fits.open(filename)
        
        # Fpack or regular fits
        if filename[-7:]=='fits.fz':
            fpack = 1 
            exten = 1 
        else: 
            fpack = 0 
            exten = 0 
         
        # Read the header
        if head is None:
            head = readfile(filename,exten=exten,header=True) 
         
        # Fix NAXIS1/2 in header 
        if fpack == 1:
            head['NAXIS1'] = head['ZNAXIS1']
            head['NAXIS2'] = head['ZNAXIS2']            

    # Does the image have a SCALE parameter
    if head.get('scale') is not None:
        scale = head['scale']
    # Does the image have a PIXSCALE parameter 
    if scale is None:
        pixscale = head.get('pixscale')
        if pixscale is not None:
            scale = pixscale
    # Does the image have a PIXSCALE1 parameter 
    if scale is None: 
        pixscale1 = head.get('pixscale1')
        if pixscale1 is not None:
            scale = pixscale1
     
    # Try the WCS 
    if scale is None:
        try:
            wcs = WCS(head)
             
            # Get the coordinates for two positions 
            #  separated by 1 pixel 
            #head_xyad,head,0.0,0.0,ra1,dec1,/degree 
            #head_xyad,head,1.0,0.0,ra2,dec2,/degree 
            #dist = sphdist(ra1,dec1,ra2,dec2,/deg)*3600. 
            #scale = dist
            c1 = wcs.pixel_to_world(0,0)
            c2 = wcs.pixel_to_world(1,0)            
            dist = c1.separation(c2).arcsec
            scale = dist
            
            if scale == 0.0: 
                scale = None 
        except:
            raise ValueError('No WCS in header')
                
    # Couldn't determine the pixel scale 
    if scale == None:
        error = 'WARNING! COULD NOT DETERMINE THE PIXEL SCALE' 
        print(error)

    return scale
 
def getfilter(filename=None,setup=None,head=None,numeric=False,noupdate=False,
              silent=False,filtname=None,fold_case=False):
    """
    This gets filter information for an image 
    using the "filter" file. 
    The "short" filter names that are returned 
    are not necessarily "unique".  The filter names 
    "I c6028", "T", "T2" all have a short filter 
    name of "T". 
 
    If a filter is not found in the filter list "filters" 
    then a new short name is created for it (based on the 
    first word in the string) and added to the list. 
 
    Parameters
    ----------
    filename : str, optional
       FITS filename 
    setup : dict
       The setup information contained in the photred.setup file.
    head : header, optional
       Use this header array instead of reading FITS file. 
    numeric : boolean, optional
       Return a numeric value instead of letter.  Default is False.
    filtname : str or list, optional
       Input the filter name explicitly instead of giving the 
              FITS filename. 
    noupdate : boolean, optional
       Don't update the "filters" file if the filter is not found.
         Default is to update.
    fold_case : boolean, optional
       Ignore case. The default is for it to be case sensitive. 
    silent : boolean, optional
       Don't print anything to the screen.  Default is False.
 
    Returns
    -------
    The short filter name is output. 
 
    Example
    -------

    filter = getfilter(fitsfile,numeric=numeric,noupdate=noupdate, 
                                 filtname=filtname,fold_case=fold_case)
 
    By D.Nidever  February 2008 
    Translated to Python by D. Nidever, April 2022
    """

    # This is what the "filters" file looks like:
    # 'M Washington c6007'    M
    # 'M Washington k1007'    M
    # 'M'                     M
    # 'I c6028'               T
    # 'I Nearly-Mould k1005'  T
    # 'T'                     T
    
    nfiles = dln.size(filename)
    nfiltname = dln.size(filtname)
    # Not enough inputs
    if filename is None and filtname is None:
        raise ValueError('Need filename or filtname')
     
    # Can't use input HEAD if multiple fits files or filter names input 
    if (nfiles > 1 or nfiltname > 1): 
        head = None 
     
    # More than one FITS filename input 
    if (nfiles > 1): 
        filters = np.zeros(nfiles,(np.str,50))
        for i in range(nfile): 
            filters[i] = getfilter(filename[i],numeric=numeric,
                                   noupdate=noupdate,fold_case=fold_case) 
        return filters 
     
    # More than one filter name input 
    # ONLY if no FITS filename input 
    if (nfiles == 0 and nfiltname > 1): 
        filters = np.zeros(nfiltname,(np.str,50))
        for i in range(nfiltname): 
            filters[i] = getfilter(filtname=filtname[i],numeric=numeric,
                                   noupdate=noupdate,silent=silent,fold_case=fold_case) 
        return filters 
     
     
    # Does the "filters" file exist? 
    if os.path.exists('filters')==False:
        if setup is None:
            raise ValueError('No setup file')
        scriptsdir = setup['scriptsdir']
        if scriptsdir is None:
            raise ValueError('NO SCRIPTSDIR')
        if os.path.exists('filters'): os.remove('filters')
        shutil.copyfile(scriptsdir+'/filters','./filters')
     
    # Load the filters
    lines = dln.readlines('filters',noblank=True)
    gd, = np.where(np.char.array(lines).strip() != '') 
    if len(gd) == 0: 
        raise ValueError('NO FILTERS')
    lines = np.char.array(lines)[gd]
    longnames = [l.split("'")[1] for l in lines]
    shortnames = [l.split("'")[2].strip() for l in lines]
     
    # Get the filter information from the header 
    if (nfiles > 0): 
        # Header NOT input, read FITS files 
        if head is None:
            # Does it have the ".fits" of ".fits.fz" ending
            ext = os.path.splitext(os.path.basename(filename))[1]
            if ext != '.fits' and filename[-7:] != 'fits.fz': 
                raise ValueError(filename+' IS NOT A FITS FILE')
             
            # Make sure the file exists 
            if os.path.exists(filename)==False:
                raise ValueError(filename+' NOT FOUND')
             
            # Read the header 
            if filename[-7:] == 'fits.fz': 
                head = readfile(filename,exten=1,header=True)
            else: 
                head = readfile(filename,header=True)
         
        # Problem with header
        if head is None:
            if silent==False:
                print(filename+' - Problem loading FITS header')
            return '' 
         
        filtname = head.get('FILTER')
        if filtname is None:
            raise ValueError('NO FILTER INFORMATION IN '+filename+' HEADER')
         
    # Get the filter name from "filtname" 
    else: 
        filtname = str(filtname[0]).strip()
     
     
    # Match filter
    ind, = np.where(np.char.array(longnames).lower()==filtname.lower())    
     
    # Match found 
    if len(ind) > 0:     
        filt = shortnames[ind[0]] 
         
        # Numeric value 
        if numeric: 
            snames,ui = np.unique(shortnames,return_index=True)  # unique short names
            nui = len(ui)
            numnames = (np.arange(nui)+1).astype(str) # numbers for the unique short names 
            gg, = np.where(snames == filt)   # which short name 
            numname = numnames[gg[0]] 
            return numname 
         
        return filt
         
    # No match found 
    else:
        if silent==False:
            print('NO FILTER MATCH')
         
        # Add it to the "filters" file 
        if noupdate==False:
             
            # The IRAF task is called "ccdsubset" 
            ## CCDSUBSET -- Return the CCD subset identifier. 
            ## 
            ## 1. Get the subset string and search the subset record file for the ID string. 
            ## 2. If the subset string is not in the record file define a default ID string 
            ##    based on the first word of the subset string.  If the first word is not 
            ##    unique append a integer to the first word until it is unique. 
            ## 3. Add the new subset string and identifier to the record file. 
            ## 4. Since the ID string is used to generate image names replace all 
            ##    nonimage name characters with '_'. 
            ## 
            ## It is an error if the record file cannot be created or written when needed. 
             
            # Get first word of the ID string 
            newshortname = filtname.split()[0]
             
            # Is this already a "short" filter name
            # string comparison
            ind, = np.where(np.char.array(shortnames).lower()==newshortname.lower())
             
            # Yes, already a "short" name 
            # Append integer until unique 
            if len(ind) > 0: 
                #newshortname = newshortname+'_' 
                # Loop until we have a unique name 
                flag = 0 
                integer = 1 
                while (flag == 0): 
                    sinteger = str(integer)
                    ind, = np.where(np.char.array(shortnames).lower()==(newshortname+sinteger).lower())                    
                     
                    # Unique 
                    if len(ind) == 0: 
                        newshortname = newshortname+sinteger 
                        flag = 1 
                    # Not unique, increment integer 
                    else: 
                        integer += 1
                     
             
            # Make sure the variable is okay 
            #newshortname = IDL_VALIDNAME(newshortname,/convert_all) 
             
            # Make sure it doesn't have any weird characters 
            # such as '*', '!', '$'
            newshortname = newshortname.replace('*','_')
            newshortname = newshortname.replace('!','_')
            newshortname = newshortname.replace('$','_')
            newshortname = newshortname.replace('__','_')
            # Add new filter to the "filters" file 
            newline = "'"+filtname+"'     "+newshortname
            with open('filters','wa') as f:
                f.write(newline)
            #dln.writelines('filters',newline,append=True)
            #WRITELINE,'filters',newline,/append 
            print('Adding new filter name to "filters" list')
            print('Filter ID string:  ',filtname)
            print('Filter short name: ',newshortname)
             
             
            # Numeric value 
            if numeric:
                # Reload filters
                lines = dln.readlines('filters')
                lines = [l.strip() for l in lines]
                lines = np.char.array(lines)
                gd, = np.where(lines != '')
                ngd = len(gd)
                lines = lines[gd]
                longnames = [l.split("'")[1] for l in lines]
                shortnames = [l.split("'")[2].strip() for l in lines]                
                 
                snames,ui = np.unique(shortnames,return_index=True)
                nui = len(ui)
                numnames = (np.arange(nui)+1).astype(str)  # numbers for the unique short names 
                 
                gg, = np.where(snames == newshortname)  # which short name 
                numname = numnames[gg[0]] 
                return numname 
             
        # Don't update 
        else: 
            print('NO FILTER MATCH')
            return '' 


# Make meta-data dictionary for an image:
def makemeta(fluxfile=None,header=None):
    '''
    This creates a meta-data dictionary for an exposure that is used by many
    of the photometry programs.  Either the filename or the header must be input.
    Note that sometimes in multi-extension FITS (MEF) files the information needed
    is both in the primary header and the extension header.  In that case it is best
    to combine them into one header and input that to makemeta().  This can easily
    be accomplished like this:
      
       head0 = fits.getheader("image1.fits",0)
       head = fits.getheader("image1.fits",1)
       head.extend(head0,unique=True)
       meta = makemeta(header=head)

    Parameters
    ----------
    fluxfile : str, optional
             The filename of the FITS image.
    header : str, optional
           The header of the image.

    Returns
    -------
    meta : astropy header
        The meta-data dictionary which is an astropy header with additional
        keyword/value pairs added.

    Example
    -------

    Create the meta-data dictionary for `image.fits`

    .. code-block:: python

        meta = makemeta("image.fits")

    Create the meta-data dictionary from `head`.

    .. code-block:: python

        meta = makemeta(header=head)

    '''

    # You generally need BOTH the PDU and extension header
    # To get all of this information

    if (fluxfile is None) & (header is None):
        print("No fluxfile or headerinput")
        return
    # Initialize meta using the header
    if fluxfile is not None:
        header = readfile(fluxfile,header=True,exten=0)
    meta = header

    #- INSTCODE -
    if "DTINSTRU" in meta.keys():
        if meta["DTINSTRU"] == 'mosaic3':
            meta["INSTCODE"] = 'k4m'
        elif meta["DTINSTRU"] == '90prime':
            meta["INSTCODE"] = 'ksb'
        elif meta["DTINSTRU"] == 'decam':
            meta["INSTCODE"] = 'c4d'
        else:
            print("Cannot determine INSTCODE type")
            return
    else:
        print("No DTINSTRU found in header.  Cannot determine instrument type")
        return

    #- RDNOISE -
    if "RDNOISE" not in meta.keys():
        # Check DECam style rdnoise
        if "RDNOISEA" in meta.keys():
            rdnoisea = meta["RDNOISEA"]
            rdnoiseb = meta["RDNOISEB"]
            rdnoise = (rdnoisea+rdnoiseb)*0.5
            meta["RDNOISE"] = rdnoise
        # Check other names
        if meta.get('RDNOISE') is None:
            for name in ['READNOIS','ENOISE']:
                if name in meta.keys(): meta['RDNOISE']=meta[name]
        # Bok
        if meta['INSTCODE'] == 'ksb':
            meta['RDNOISE']= [6.625, 7.4, 8.2, 7.1][meta['CCDNUM']-1]
        if meta.get('RDNOISE') is None:
            print('No RDNOISE found')
            return
    #- GAIN -
    if "GAIN" not in meta.keys():
        try:
            gainmap = { 'c4d': lambda x: 0.5*(x.get('GAINA')+x.get('GAINB')),
                        'k4m': lambda x: x.get('GAIN'),
                        'ksb': lambda x: [1.3,1.5,1.4,1.4][x.get['CCDNUM']-1] }  # bok gain in HDU0, use list here
            gain = gainmap[meta["INSTCODE"]](meta)
            meta["GAIN"] = gain
        except:
            gainmap_avg = { 'c4d': 3.9845419, 'k4m': 1.8575, 'ksb': 1.4}
            gain = gainmap_avg[meta["INSTCODE"]]
            meta["GAIN"] = gain
    #- CPFWHM -
    # FWHM values are ONLY in the extension headers
    cpfwhm_map = { 'c4d': 1.5 if meta.get('FWHM') is None else meta.get('FWHM')*0.27, 
                   'k4m': 1.5 if meta.get('SEEING1') is None else meta.get('SEEING1'),
                   'ksb': 1.5 if meta.get('SEEING1') is None else meta.get('SEEING1') }
    cpfwhm = cpfwhm_map[meta["INSTCODE"]]
    meta['CPFWHM'] = cpfwhm
    #- PIXSCALE -
    if "PIXSCALE" not in meta.keys():
        pixmap = { 'c4d': 0.27, 'k4m': 0.258, 'ksb': 0.45 }
        try:
            meta["PIXSCALE"] = pixmap[meta["INSTCODE"]]
        except:
            w = WCS(meta)
            meta["PIXSCALE"] = np.max(np.abs(w.pixel_scale_matrix))

    return meta
