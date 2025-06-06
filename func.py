import numpy as np
from math import sqrt, pi #, fabs, copysign
import matplotlib.pyplot as plt
import matplotlib
import os
from warnings import warn


R_MOON = 1737.1
B0 = 3.85  # nT
# B0 = 2.6 #nT
B0Gs = B0 * 1e-5
n0 = 6.99  # cm-3
c = 3e10  # cm/s
e = 4.8e-10  # CGS
mp = 1.67e-24  # g
# rho0 = n0*mp



li = c / sqrt(4 * pi * n0 * (e ** 2) / mp)  # cm
likm = 1e-5 * li
Omegainv = mp * c / e / B0Gs  # s
Va = likm / Omegainv  # km/s
Enorm = mp * Va * Va * 6.24e21  # erg -> eV + Va km -> cm
mime = 1836
Te_eV = 9.8  # eV
Te_erg = Te_eV * 1.6e-12  # erg

Tp_eV = 2.4  # eV
Tp_erg = Tp_eV * 1.6e-12  # erg

pe = n0 * Te_erg
p_mag = B0Gs ** 2 / (8 * np.pi)
betae = pe / p_mag

pp = n0 * Tp_erg
betap = pp / p_mag



def read_and_paint_2d_plots(file_name):
    """
    basic fucntion for reading files and drawing 2d plots
    """
    fin = open(file_name, 'r')
    x = []
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []

    for line in fin:
        b = line
        b = b.split()
        if len(b) == 2:
            x.append(float(b[0]))
            f1.append(float(b[1]))
        elif len(b) == 4:
            x.append(float(b[0]))
            f1.append(float(b[1]))
            f2.append(float(b[2]))
            f3.append(float(b[3]))

        elif len(b) == 8:
            x.append(float(b[0]))
            f1.append(float(b[1]))
            f2.append(float(b[2]))
            f3.append(float(b[3]))
            f4.append(float(b[4]))
            f5.append(float(b[5]))
            f6.append(float(b[6]))
            f7.append(float(b[7]))

    if len(b) == 2:
        plt.title(file_name)
        plt.plot(x, f1, '-r')
    elif len(b) == 4:
        plt.subplot(311)
        plt.title(file_name)
        plt.plot(x, f1, '-r')
        plt.subplot(312)
        plt.plot(x, f2, '-r')
        plt.subplot(313)
        plt.plot(x, f3, '-r')

    elif len(b) == 8:
        plt.subplot(711)
        plt.title(file_name)
        plt.plot(x, f1, '-r')
        plt.subplot(712)
        plt.plot(x, f2, '-r')
        plt.subplot(713)
        plt.plot(x, f3, '-r')
        plt.subplot(714)
        plt.plot(x, f4, '-r')
        plt.subplot(715)
        plt.plot(x, f5, '-r')
        plt.subplot(716)
        plt.plot(x, f6, '-r')
        plt.subplot(717)
        plt.plot(x, f7, '-r')
    fin.close()


def read_color_map_array(local_filename):
    """
    basic function for reading dat file containing colormaps both 2d and 3d
    3D files must contain string 3d in the name
    returns y - array-like object,  number of x y (z) grid-points and min-max coordinates of grid
    """
    if '3d' in local_filename:
        fin = open(local_filename, 'r')

        b = fin.readline()

        index = 0
        b = b.split()
        for i in b:
            if i == '!nx':
                nx = int(b[index + 1])
                index += 2
            elif i == 'ny':
                ny = int(b[index + 1])
                index += 2
            elif i == 'nz':
                nz = int(b[index + 1])
                index += 2
            elif i == 'xmin':
                xmin = float(b[index + 1])
                index += 2

            elif i == 'xmax':
                xmax = float(b[index + 1])
                index += 2

            elif i == 'ymin':
                ymin = float(b[index + 1])
                index += 2

            elif i == 'ymax':
                ymax = float(b[index + 1])
                index += 2

            elif i == 'zmin':
                zmin = float(b[index + 1])
                index += 2

            elif i == 'zmax':
                zmax = float(b[index + 1])

        y = []
        temp = []

        for line in fin:
            if line == '\n':
                y.append(temp)
                temp = []
            else:
                temp.append(list(map(float, line.split())))

        fin.close()

        return [y, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax]
    else:
        fin = open(local_filename, 'r')
        b = fin.readline()

        index = 0
        b = b.split()
        for i in b:
            if i == '!nx':
                nx = int(b[index + 1])
                index += 2
            elif i == 'ny':
                ny = int(b[index + 1])
                index += 2

            elif i == 'xmin':
                xmin = float(b[index + 1])
                index += 2

            elif i == 'xmax':
                xmax = float(b[index + 1])
                index += 2

            elif i == 'ymin':
                ymin = float(b[index + 1])
                index += 2

            elif i == 'ymax':
                ymax = float(b[index + 1])

        y = []
        for line in fin:
            y.append(list(map(float, line.split())))

        fin.close()

        return [y, nx, ny, xmin, xmax, ymin, ymax]


def make_map(file_name,
             coordinates='di', orientation_plane='yz', orientation_coord=0,
             abs=False, streamlines=False,
             moon_body=False, moon_body_color=matplotlib.colors.ListedColormap(["grey"]),
             block_moon_body_vmax_vmin=False, colorbar=True, colorbar_prop=None, meanx=0,
             **kwargs):
    """
    that's multifunctional procedure that creates colorplotusing read_color_map_array

    :param file_name: filename
    :param coordinates: changes axis units. avilible are normalized (used in modeling like)  di and cells, and 'real'
    dimentional km and selenograghic.
    :param orientation_plane:  purely 3d thing  allows to look at specific plane we want
    :param orientation_coord:  purely 3d thing  allows to look at specific plane we want
    :param abs: if we have a vector value tipically V (B E) from Vx Vy and Vz maps collect module V
    :param streamlines: draw streamlines ( that's somehow slowes everything down a lot maybe modify later)
    :param moon_body: calculate from zero-time density map how moon body is located and hide evetything that is inside
    the moon since it's unphysical
    :param moon_body_color:  colormap for moon_body
    :param block_moon_body_vmax_vmin: not just hide but modify the initial map itself - setting everything under
    the Moon as none (tipically I alsways use it ? so maybe i should unite it with moon_body)
    :param colorbar: plorthe colorbar or not
    :param colorbar_prop: colorbar properties dictionary
    :param meanx: for streamlines, we tipically have someinitial flow of solar wind in x direction, so it's to substruct it
    from the streamlines
    :param kwargs: pcolormesh keyword arguments
    :return: figure and axes of the plot, color map, list of (x y and vaues of colormap) , moonbody
    """

    if colorbar_prop is None:
        colorbar_prop = {}
    if abs == False and streamlines==True:
       streamlines=False
       warn('abs is False and streamlines is True, blocking streamlines')
    if moon_body == False and block_moon_body_vmax_vmin == True:
        raise AttributeError('block_moon_body_vmax_vmin works together with moon_body! Please switch it on!')
    if '3d' in file_name:
        if abs:
            # MODIFY THIS PART
            data = []
            [datax, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax] = read_color_map_array(file_name + 'x.z')
            [datay, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax] = read_color_map_array(file_name + 'y.z')
            [dataz, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax] = read_color_map_array(file_name + 'z.z')
            for zx, zy, zz in zip(datax, datay, dataz):
                data.append([])
                for yx, yy, yz in zip(zx, zy, zz):
                    data[-1].append([])
                    for xx, xy, xz in zip(yx, yy, yz):
                        data[-1][-1].append((xx ** 2 + xy ** 2 + xz ** 2) ** 0.5)
        else:
            [data, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax] = read_color_map_array(file_name)
        new_data = []
        # planes
        # yz
        if 'x' not in orientation_plane:
            for z in data:
                new_data.append([])
                for y in z:
                    new_data[-1].append(y[orientation_coord])
            if orientation_plane[0] == 'y':
                koordinata_x = np.linspace(ymin, ymax, ny + 1)
                koordinata_y = np.linspace(zmin, zmax, nz + 1)
                pp = np.array(new_data)
            else:
                koordinata_x = np.linspace(zmin, zmax, nz + 1)
                koordinata_y = np.linspace(ymin, ymax, ny + 1)
                pp = np.array(new_data).T
        # xz
        elif 'y' not in orientation_plane:
            for z in data:
                new_data.append(z[orientation_coord])
            if orientation_plane[0] == 'x':
                koordinata_x = np.linspace(xmin, xmax, nx + 1)
                koordinata_y = np.linspace(zmin, zmax, nz + 1)
                pp = np.array(new_data)
            else:
                koordinata_x = np.linspace(zmin, zmax, nz + 1)
                koordinata_y = np.linspace(xmin, xmax, nx + 1)
                pp = np.array(new_data).T
        # xy
        elif 'z' not in orientation_plane:
            new_data = data[orientation_coord]
            if orientation_plane[0] == 'x':
                koordinata_x = np.linspace(xmin, xmax, nx + 1)
                koordinata_y = np.linspace(ymin, ymax, ny + 1)
                pp = np.array(new_data)
            else:
                koordinata_x = np.linspace(ymin, ymax, ny + 1)
                koordinata_y = np.linspace(xmin, xmax, nx + 1)
                pp = np.array(new_data).T
        else:
            raise ValueError('orientation_cord should contain only x y or z')
    else:
        if abs:
            data = []
            [datax, nx, ny, xmin, xmax, ymin, ymax] = read_color_map_array(file_name + 'x.z')
            [datay, nx, ny, xmin, xmax, ymin, ymax] = read_color_map_array(file_name + 'y.z')
            [dataz, nx, ny, xmin, xmax, ymin, ymax] = read_color_map_array(file_name + 'z.z')
            for yx, yy, yz in zip(datax, datay, dataz):
                data.append([])
                for xx, xy, xz in zip(yx, yy, yz):
                    data[-1].append((xx ** 2 + xy ** 2 + xz ** 2) ** 0.5)
        else:
            [data, nx, ny, xmin, xmax, ymin, ymax] = read_color_map_array(file_name)
        koordinata_x = np.linspace(xmin, xmax, nx + 1)
        koordinata_y = np.linspace(ymin, ymax, ny + 1)
        pp = np.array(data)
        print(orientation_plane, orientation_coord)
    if streamlines:
        # yz
        if 'x' not in orientation_plane:
            new_datay = []
            new_dataz = []
            for z in datay:
                new_datay.append([])
                for y in z:
                    new_datay[-1].append(y[orientation_coord])

            for z in dataz:
                new_dataz.append([])
                for y in z:
                    new_dataz[-1].append(y[orientation_coord])

            if orientation_plane[0] == 'y':
                slx = np.array(new_datay)
                sly = np.array(new_dataz)
            else:
                slx = np.array(new_dataz).T  # ?
                sly = np.array(new_datay).T  # ?

        elif 'y' not in orientation_plane:
            new_datax = []
            new_dataz = []
            for z in datax:
                new_datax.append(z[orientation_coord])
            for z in dataz:
                new_dataz.append(z[orientation_coord])
            if orientation_plane[0] == 'x':
                slx = np.array(new_datax)
                sly = np.array(new_dataz)
            else:
                slx = np.array(new_dataz).T  # ?
                sly = np.array(new_datax).T  # ?
        # xy
        elif 'z' not in orientation_plane:
            new_datax = datax[orientation_coord]
            new_datay = datay[orientation_coord]
            if orientation_plane[0] == 'x':
                slx = np.array(new_datax)
                sly = np.array(new_datay)
            else:
                slx = np.array(new_datay).T#?
                sly = np.array(new_datax).T#?
    if moon_body:
        if 'field' in file_name:
            dens_file_name = file_name.replace('field', 'dens').replace('_Bx', '').replace('_By', '').replace('_Bz',
                                                                                                              '').replace(
                '_Ex', '').replace('_Ey', '').replace('_Ez', '')
        elif 'vel' in file_name:
            dens_file_name = file_name.replace('vel', 'dens').replace('_Vx', '').replace('_Vy', '').replace('_Vz', '')
        elif 'dens' in file_name:
            dens_file_name = file_name
        else:
            raise NameError("wrong name of file it has to contain 'field', 'vel' or 'dens'")
        if abs: dens_file_name = dens_file_name.replace('_B', '').replace('_E', '').replace('_V', '') + '.z'

        dens_file_name = dens_file_name[:-12] + '000000_000.z'
        if not '3d' in file_name:
            [data, nx, ny, xmin, xmax, ymin, ymax] = read_color_map_array(dens_file_name)
            moon_mask = np.array(data)
        else:
            [data, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax] = read_color_map_array(dens_file_name)
            moon_mask = np.array(data)
            # planes
            # yz
            if 'x' not in orientation_plane:
                if orientation_plane[0] == 'y':
                    moon_mask = moon_mask[:, :, orientation_coord]
                else:
                    moon_mask = moon_mask[:, :, orientation_coord].T
            # xz
            elif 'y' not in orientation_plane:
                if orientation_plane[0] == 'x':
                    moon_mask = moon_mask[:, orientation_coord, :]
                else:
                    moon_mask = moon_mask[:, orientation_coord, :].T
            # xy
            elif 'z' not in orientation_plane:
                if orientation_plane[0] == 'x':
                    moon_mask = moon_mask[orientation_coord, :, :]
                else:
                    moon_mask = moon_mask[orientation_coord, :, :].T

        for i in range(len(moon_mask)):
            for j in range(len(moon_mask[0])):
                if moon_mask[i, j] != 1.00000:
                    moon_mask[i, j] = None
                elif orientation_plane[0] == 'x' and j + 1 < len(moon_mask[0]) and moon_mask[i, j + 1] != 1.00000:
                    moon_mask[i, j] = None
                elif block_moon_body_vmax_vmin:
                    pp[i, j] = None
                    if streamlines:
                        slx[i,j]=None
                        sly[i,j]=None

        cmoon = plt.get_cmap(moon_body_color, 1)


    if coordinates == 'di':
        pc_args = [koordinata_x, koordinata_y, pp, ]
        plt.xlabel(orientation_plane[0] + ',$d_i$', labelpad=0)
        plt.ylabel(orientation_plane[1] + ',$d_i$', labelpad=0)
    elif coordinates == 'km':
        pc_args = [koordinata_x * likm, koordinata_y * likm, pp, ]
        plt.xlabel(orientation_plane[0] + ',$km$', labelpad=0,fontsize=20)
        plt.ylabel(orientation_plane[1] + ',$km$', labelpad=0,fontsize=20)
        if 'field' in file_name and 'B' in file_name:pp*=B0
        elif 'dens' in file_name: pp*=n0
        elif 'vel' in file_name:pp*=Va
        elif 'field' in file_name and 'E' in file_name: pp*=(B0Gs*Va*1e5)/c
    elif coordinates == 'cells':
        pc_args = [pp, ]
        plt.xlabel(orientation_plane[0], labelpad=0)
        plt.ylabel(orientation_plane[1], labelpad=0)
    elif 'selenographic' in coordinates:
        if 'field' in file_name and 'B' in file_name: pp *= B0
        if 'dens' in file_name: pp *= n0
        if 'vel' in file_name: pp *= Va
        elif 'field' in file_name and 'E' in file_name: pp*=(B0Gs*Va*1e5)/c

        if coordinates[1] == 'Gerasimovich' or coordinates == 'selenographic':
            theta_center, phi_center = 20, 123.5
        else:
            theta_center, phi_center = coordinates[1], coordinates[2]
        # now its only S and W, add else later
        if orientation_plane[0] == 'x':
            koordinata_x = koordinata_x * likm
            plt.xlabel('r,$km$', labelpad=0)
        elif orientation_plane[0] == 'y':
            # koordinata_x = 20 - (np.linspace(ymin, ymax, ny + 1) - (ymax - ymin) / 2) * likm / R_MOON * 180 / np.pi
            koordinata_x = theta_center - (
                        koordinata_x - (koordinata_x[-1] - koordinata_x[0]) / 2) * likm / R_MOON * 180 / np.pi
            plt.xlabel('° S', labelpad=0)
        elif orientation_plane[0] == 'z':
            koordinata_x = phi_center - (
                        np.linspace(zmin, zmax, nz + 1) - (zmax - zmin) / 2) * likm / R_MOON * 180 / np.pi
            plt.xlabel('° W', labelpad=0)
        if orientation_plane[1] == 'x':
            koordinata_y = koordinata_y * likm
            plt.ylabel('r,$km$', labelpad=0)
        elif orientation_plane[1] == 'y':
            koordinata_y = theta_center - (
                        np.linspace(ymin, ymax, ny + 1) - (ymax - ymin) / 2) * likm / R_MOON * 180 / np.pi
            plt.ylabel('° S', labelpad=0)
        elif orientation_plane[1] == 'z':
            # koordinata_y = 123.5 - (np.linspace(zmin, zmax, nz + 1) - (zmax - zmin) / 2) * likm / R_MOON * 180 / np.pi
            koordinata_y = phi_center - (
                        koordinata_y - (koordinata_y[-1] - koordinata_y[0]) / 2) * likm / R_MOON * 180 / np.pi
            plt.ylabel('° W', labelpad=0)
        pc_args = [koordinata_x, koordinata_y, pp, ]
    else:
        raise ValueError("Unknown coordinates, available are 'di','km','selenographic' and 'cells'")

    pcm = None
    if streamlines:
        if len(pc_args)==3:x,y=pc_args[:-1]
        else: x,y=np.linspace(0, nx, nx + 1),np.linspace(0, ny, ny + 1)
        ps=plt.streamplot(x[:-1],y[:-1],slx-meanx,sly,color='r',density=1.5)
    pc = plt.pcolormesh(*pc_args, **kwargs)
    plt.gca().set_xlim(min(pc_args[0]),max(pc_args[0]))
    plt.gca().set_ylim(min(pc_args[1]),max(pc_args[1]))
    if colorbar:
        cb=plt.colorbar(pc,**colorbar_prop)
        cb.ax.tick_params(labelsize=20)
        # cb.ax.tick_params(labelsize=20)

    if moon_body:
        # pc_args[-1] = moon_mask
        pcm = plt.pcolormesh(pc_args[0],pc_args[1],moon_mask, cmap=cmoon)
        plt.scatter([], [], c=[cmoon(1)], s=100, label='Moon body')
        plt.legend(loc='upper right')
        plt.sci(pc)
    return plt.gcf(), plt.gca(), pc,pc_args, pcm,


def make_dens_field_vel_maps(dir_names,file_name_cores,*time,
                             result_dir_name=None,types='all', coordinates='di',meanx=0,
                             orientation=zip((54,-1,-2,-3,-4,-5,),('xz','yz','yz','yz','yz','yz',)),
                             moon=zip((True,),(True,),('_moon_body_',)),sub=(1,1),
                             adj=None,**kwargs):
    """
    creates a bunch of png pictures using make_map. If we want to compare results of different simulations we can use
    several files and place them in subplots. We can make different types of pictures making different combinations
    of times, orientation planes and moon regimes. Example time=[0,1] orientation planes are xy1 and xz10, and we want
    both types of pictures with moon body and without. We will have8 compinations of these parameters and 8 pictures.


    :param dir_names: list of full names of directories where files are located. If we have 1 dictionary string with name
    instead of list can be used
    :param file_name_cores: list with filecores of files. It corresponds with dirnames positionally, if we want a file
    from dir1 with core1 - we need to have dir1 with core1 on the same positions in dir_names and file_name_cores lists
    if we have several cores in same dir add that dir in the dir_names several times cotispondig it with cores you need
    If we have 1 core string with name instead of list can be used
    :param time: time moments we make pictures of it can be 1 arguement =1 moment of time, 2 args(start, final) = times
    from start to final (both includet) with step 1 hyrofrequncy, and 3 args (start, final, step) = times
    from start to final (both includet) with step = step
    :param result_dir_name: full name of directory where results wil be saved
    :param types:  list with lists of filename prefixes suffixes abss and streamlines
    (see abs and streamlinesin make_map) lists are positionally corelated
    :param coordinates:  changes axis units. avilible are normalized (used in modeling like)  di and cells, and 'real'
    dimentional km. Not only axes are affected by this parameter, values that are ploted will have
    normilised units when di or cells is used and SGC/SI units when km or selenograghic is used.
    :param meanx: for streamlines, we tipically have someinitial flow of solar wind in x direction, so it's to substruct it
    from the streamlines
    :param orientation: zip or list with lists of same sence that simbolize plane of picture with orientation_plane and
    orientation_coord parametrs for make_map.
    :param moon:  zip or list with lists of same sence that simbolize different moon_regimes: moon_body and
    block_moon_body_vmax_vmin paramenter for make_map, and moon_suffix that will be added to resulted file
    :param sub: amount of rows and cols of subplots. Note that amount of subplots should be able to contain all files
    :param adj: adjust dictionary (see plt.subplots_adjust)
    :param kwargs: make_map keyword arguments (make_map will use them as pcolormesh keyword arguments)
    :return: None
    """
    if type(dir_names)==str :dir_names=[dir_names]
    if type(file_name_cores) == str: file_name_cores=[file_name_cores]
    if len(dir_names) != len(file_name_cores):raise ValueError('dir_names should have same length as file_name_cores')
    if sub[0] * sub[1] < len(dir_names): raise ValueError('subs should be able to contain all files')

    if sub==(1,1): makemap_colorbar=True
    else:makemap_colorbar=False
    pc_to_cb_h=15
    grid = plt.GridSpec(sub[0]*pc_to_cb_h+(not makemap_colorbar)*2,sub[1])

    if adj is None:
        adj = dict(left=0.115, bottom=0.08, right=0.985, top=0.98)
    for dir_name in dir_names:
        if not os.path.isdir(dir_name): raise FileNotFoundError('No such file or directory: '+dir_name)

    if result_dir_name is None:
        result_dir_name=dir_names[0]+'/results/'
        if not os.path.isdir(result_dir_name ): os.mkdir(result_dir_name)

    if not os.path.isdir(result_dir_name): raise FileNotFoundError('No such file or directory: '+result_dir_name)
    if len(dir_names)==1:result_dir_name=result_dir_name+'/dens_field_vel/'
    elif len(dir_names)>1:result_dir_name=result_dir_name+'/dens_field_vel_subplots/'
    if not os.path.isdir(result_dir_name) and len(dir_names): os.mkdir(result_dir_name)

    ORIENTATION = list(orientation)[:]
    MOON=list(moon)[:]
    if types=='all':
        prefs=['field3d']*2+['vel3d']+['dens3d']+['field3d']*6+['vel3d']*3
        sufs= ['_B','_E','_V','.z','_Bx.z', '_By.z', '_Bz.z','_Ex.z', '_Ey.z', '_Ez.z','_Vx.z', '_Vy.z', '_Vz.z']
        absss=[True]*3+[False]*10
        streamlines=[True]*3+[False]*10
    elif type(types)==str and len(types)==1:
        if types=='E' or types=='B': prefs = ['field3d'] * 4
        elif types=='V': prefs =['vel3d']* 4
        sufs = [ '_'+types+ _ for _ in['',  'x.z', 'y.z', 'z.z',] ]
        absss = [True]  + [False] * 3
        streamlines = [True]  + [False] * 3
    else:
        prefs,sufs,absss,streamlines=types
    if len(time) == 1:
        if type(*time)==np.ndarray:
            if len(time[0].shape)==1: TIMES=time[0]
            else:raise ValueError('only 1d np.ndarray!')
        elif type(*time)==int:TIMES=np.linspace(*time,*time,1)
        else: raise ValueError('1 positional time argument only takes np.ndarray or int')
    elif len(time) == 2: TIMES=np.linspace(*time,time[1]-time[0]+1)
    elif len(time) == 3:
        TIMES = np.linspace(time[0],time[1],round(1+(time[1]-time[0])/time[2]))
    else: raise ValueError('Only 1,2 or 3 positional arguments!')

    for pref,suf,abss,sl in zip(prefs,sufs ,absss,streamlines):
        suff = suf.replace('.z','')
        if suff == '': suff = '_' + pref[:-2]
        print(suff)
        if  not os.path.isdir(result_dir_name +suff[1:]) : os.mkdir(result_dir_name + suff[1:])
        for time in TIMES:
            for orientation_coord, orientation_plane in ORIENTATION: #54,-1,-2,  'xz','yz',
                for moon_body,block_moon_body_vmax_vmin,moon_name in MOON:
                    fig=plt.figure(figsize=(18, 12))
                    for row in range(sub[0]):
                        for col in range(sub[1]):
                            fig.add_subplot(grid[row*pc_to_cb_h:(row+1)*pc_to_cb_h,col])#sub[0],sub[1],row*sub[0]+col+1
                    if not makemap_colorbar:fig.add_subplot(grid[-1, :])  # colorbar ax
                    pcs=[]
                    for ax,dir_name,file_name_core in zip(fig.axes,dir_names,file_name_cores):
                        file_name = dir_name + pref + file_name_core + 't{:010.3F}'.format(time).replace('.', '_',1) + suf
                        plt.sca(ax)

                        dt=make_map(file_name, orientation_coord=orientation_coord, orientation_plane=orientation_plane, abs=abss,
                                 moon_body=moon_body, block_moon_body_vmax_vmin=block_moon_body_vmax_vmin,
                                 streamlines=sl,colorbar=makemap_colorbar,coordinates=coordinates,meanx=meanx,**kwargs)
                        pcs.append((dt[2]))
                        plt.tick_params(labelsize=20)
                        if makemap_colorbar:
                            if coordinates=='km':
                                if suff[1]=='B':plt.gci().colorbar.set_label(suff[1:]+"$,nT$",fontsize=20)
                                elif suff[1]=='V': plt.gci().colorbar.set_label(suff[1:]+"$,km/s$",fontsize=20)
                                elif suff[1]=='E':plt.gci().colorbar.set_label(suff[1:]+"$,Gs$",fontsize=20)
                                elif suff[1:]=='dens':plt.gci().colorbar.set_label("$n, cm^{-3}$",fontsize=20)
                            else:
                                if suff[1]=='B':plt.gci().colorbar.set_label(suff[1:]+"$/B_0$",fontsize=20)
                                elif suff[1]=='V': plt.gci().colorbar.set_label(suff[1:]+"$/V_a$",fontsize=20)
                                elif suff[1]=='E':plt.gci().colorbar.set_label(suff[1:]+"$/ (B_0*V_a/c)$",fontsize=20)
                                elif suff[1:]=='dens':plt.gci().colorbar.set_label("$n/n_0$",fontsize=20)
                    if not makemap_colorbar:
                        MI, MA = [], []
                        for p in pcs:
                            mi, ma = p.get_clim()
                            MI.append(mi)
                            MA.append(ma)
                        for p in pcs:
                            p.set_clim(min(MI), max(MA))
                            if min(MI) == max(MA): p.set_clim(min(MI) - 0.1, min(MI) + 0.1)
                        cb = fig.colorbar(p, cax=fig.axes[-1], orientation='horizontal', format=matplotlib.ticker.ScalarFormatter() )  # ,cax=,
                        cb.ax.tick_params(labelsize=20)
                        if coordinates == 'km':
                            if suff[1] == 'B':
                                plt.gci().colorbar.set_label(suff[1:] + "$,nT$", fontsize=20)
                                # plt.gci().set_clim()
                            elif suff[1] == 'V':
                                plt.gci().colorbar.set_label(suff[1:] + "$,km/s$", fontsize=20)
                            elif suff[1] == 'E':
                                plt.gci().colorbar.set_label(suff[1:] + "$,Gs$", fontsize=20)
                            elif suff[1] == 'dens':
                                plt.gci().colorbar.set_label("$n, cm^{-3}$", fontsize=20)
                        else:
                            if suff[1] == 'B':
                                plt.gci().colorbar.set_label(suff[1:] + "$/B_0$", fontsize=20)
                            elif suff[1] == 'V':
                                plt.gci().colorbar.set_label(suff[1:] + "$/V_a$", fontsize=20)
                            elif suff[1] == 'E':
                                plt.gci().colorbar.set_label(suff[1:] + "$/ (B_0*V_a/c)$", fontsize=20)
                            elif suff[1] == 'dens':
                                plt.gci().colorbar.set_label("$n/n_0$", fontsize=20)


                    plt.subplots_adjust(**adj)
                    if abss:
                        abs_pref = 'abs_'
                    else:
                        abs_pref = ''
                    if sl: sl_suff='_sl'
                    else: sl_suff=''
                    save_name = result_dir_name + suff[1:] + '/' + abs_pref + pref + file_name_core + str(
                                orientation_coord) + orientation_plane + moon_name + 't{:010.3F}'.format(
                                time).replace('.', '_', 1) +sl_suff+ suff+ '.png'
                    print(save_name)
                    plt.savefig(save_name)
                    plt.close('all')

def make_n_subplots(dir_names,file_name_cores,orientation_coords, *time,
                    result_dir_name=None, values=None, adj=None, coordinates='di', coordplane='yz',sub=(1,1), **kwargs):


    """
    !
    This function will be REMADE when we improve code in Maximus. Now Maximus only can give 2d files ( we need xy-planes),
    but we plane to change it so it gives us 3d files too, aftee we finish runing tests for new model.
    I plane to make remaded version to look more like make_dens_field_vel_maps or maybe combine them into one function.
    !

    make maps for concentrations: total upgoing and downgoing.

    :param dir_names: list of full names of directories where files are located. If we have 1 dictionary string with name
    instead of list can be used
    :param file_name_cores: list with filecores of files. It corresponds with dirnames positionally, if we want a file
    from dir1 with core1 - we need to have dir1 with core1 on the same positions in dir_names and file_name_cores lists
    if we have several cores in same dir add that dir in the dir_names several times cotispondig it with cores you need
    If we have 1 core string with name instead of list can be used
    :param orientation_coords: coord of yz plane
    :param time: time moments we make pictures of it can be 1 arguement =1 moment of time, 2 args(start, final) = times
    from start to final (both includet) with step 1 hyrofrequncy, and 3 args (start, final, step) = times
    from start to final (both includet) with step = step
    :param result_dir_name:  full name of directory where results wil be saved
    :param values: consentrations we want picture for (total upgoing and downgoing). If we want all ['n', 'nup', "nотр"]
    :param adj: adjust dictionary (see plt.subplots_adjust)
    :param coordinates: changes axis units. avilible are normalized (used in modeling like)  di and cells, and 'real'
    dimentional km.
    :param coordplane: it can be 'yz' or 'zy' basically first simbol is xaxis and second one is yaxis.
    :param sub:amount of rows and cols of subplots. Note that amount of subplots should be able to contain all files
    :param kwargs: pcolormesh keyword arguments
    :return: None
    """
    if values is None:
        values = ['n', 'nup', "nотр"]
    if adj is None and sub!=(1,1):
        if coordinates=='di':
            adj = dict(left=0.045, bottom=0.085, right=0.98, top=0.96, wspace=0.15, hspace=0.15)
        elif coordinates=='km':
            adj=dict(left=0.065, bottom=0.085, right=0.98, top=0.96,wspace=0.25,hspace=0.15)
    elif adj is None and sub==(1,1) :
        adj = dict(left=0.06, bottom=0.075, right=0.955, top=0.96, wspace=0.25, hspace=0.1)

    if type(dir_names)==str :dir_names=[dir_names]
    if type(file_name_cores) == str: file_name_cores=[file_name_cores]
    if len(dir_names) != len(file_name_cores):raise ValueError('dir_names should have same length as file_name_cores')
    if sub[0] * sub[1] < len(dir_names): raise ValueError('subs should be able to contain all files')

    pc_to_cb_h=15
    grid = plt.GridSpec(sub[0]*pc_to_cb_h+2,sub[1])
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            raise FileNotFoundError('No such file or directory: '+dir_name)

    if result_dir_name is None:
        result_dir_name=dir_names[0]+'/results/'
        if not os.path.isdir(result_dir_name ): os.mkdir(result_dir_name)

    if not os.path.isdir(result_dir_name):
        os.mkdir(result_dir_name)
        warn('No such file or directory: ' + result_dir_name + 'creating one')
    if len(dir_names)==1:result_dir_name=result_dir_name+'/n_subplot/'
    elif len(dir_names)>1:result_dir_name=result_dir_name+'/n_subplots/'
    if not os.path.isdir(result_dir_name) and len(dir_names): os.mkdir(result_dir_name)

    if len(time) == 1:
        if type(*time)==np.ndarray:
            if len(time[0].shape)==1: TIMES=time[0]
            else:raise ValueError('only 1d np.ndarray!')
        elif type(*time)==int:TIMES=np.linspace(*time,*time,1)
        else: raise ValueError('1 positional time argument only takes np.ndarray or int')
    elif len(time) == 2: TIMES=np.linspace(*time,time[1]-time[0]+1)
    elif len(time) == 3:
        TIMES = np.linspace(time[0],time[1],round(1+(time[1]-time[0])/time[2]))
    else: raise ValueError('Only 1,2 or 3 positional arguments!')

    for time in TIMES:
        for orientation_coord in orientation_coords:
            figs=[]
            for v in values:
                figs.append(plt.figure(figsize=(16, 9)))
                if sub!=(1,1):
                    for row in range(sub[0]):
                        for col in range(sub[1]):
                            figs[-1].add_subplot(grid[row * pc_to_cb_h:(row + 1) * pc_to_cb_h, col])  # sub[0],sub[1],row*sub[0]+col+1
                    figs[-1].add_subplot(grid[-1, :])  # colorbar ax
                else:
                    figs[-1].add_subplot(plt.GridSpec(1,100)[:,:-15])
                    figs[-1].add_subplot(plt.GridSpec(1,100)[:,-10:-5])

            for fig, v in zip(figs, values):
                pcs = []
                for dn, core, a in zip(dir_names,file_name_cores, fig.axes):#
                    if v=='n' or v== 'nup' or v== "nотр":
                        if '_He_' in core:
                            sort = 'He'
                            temp = 0.25 / 4
                        elif '_H_' in core:
                            sort = 'H'
                            temp = 0.75
                        else:
                            sort = 'test'
                            temp = 1

                    if v== "n":
                        filename = dn + 'n' + core + str(orientation_coord)+'_' + 't{:010.3F}'.format(time).replace('.', '_', 1) + '.z'
                        data, nx, ny, xmin, xmax, ymin, ymax,  = read_color_map_array(filename)
                        data=np.array(data)/temp
                    elif v== "nup":
                        filename = dn + 'nup' + core + str(orientation_coord)+'_' + 't{:010.3F}'.format(time).replace('.', '_', 1) + '.z'
                        data, nx, ny, xmin, xmax, ymin, ymax,  = read_color_map_array(filename)
                        data=np.array(data)/temp
                    else:
                        filename1 = dn + 'n' + core + str(orientation_coord)+'_' + 't{:010.3F}'.format(time).replace('.', '_', 1) + '.z'
                        filename2 = dn + 'nup' + core + str(orientation_coord)+'_' + 't{:010.3F}'.format(time).replace('.', '_', 1) + '.z'

                        data1, nx, ny, xmin, xmax, ymin, ymax, = read_color_map_array(filename1)
                        data2, nx, ny, xmin, xmax, ymin, ymax, = read_color_map_array(filename2)
                        data = (np.array(data1) -np.array(data2))
                    if coordinates == 'di':
                        x1 = np.linspace(xmin, xmax, nx + 1)
                        y1 = np.linspace(ymin, ymax, ny + 1)
                    elif coordinates == 'km':
                        x1 = np.linspace(xmin, xmax, nx + 1) * likm
                        y1 = np.linspace(ymin, ymax, ny + 1) * likm
                    else:
                        raise ValueError('not supported')
                    xy=[x1,y1]
                    if coordplane=='zy':
                        data[0]=data[0].T
                        xy=[y1,x1,]

                    pc = a.pcolormesh(xy[0],xy[1], data, **kwargs)
                    pcs.append(pc)

                    if v == 'n' or v == 'nup' or v == "nотр":
                        a.set_title('$' + v[0] + '^{' + v[1:] + '}_{' + sort + '}$', fontsize=20)
                        if sort == 'test': a.set_title('$' + v[0] + '^{' + v[1:] + '}_{H}$' + '(Только H)',fontsize=20)
                    if coordinates == 'di':
                        a.set_xlabel('z,$d_i$', labelpad=-7, fontsize=20)
                        a.set_ylabel('y,$d_i$', labelpad=-7, fontsize=20)
                    if coordinates == 'km':
                        a.set_xlabel('z,km', labelpad=-7, fontsize=20)
                        a.set_ylabel('y,km', labelpad=-7, fontsize=20)
                    a.tick_params(labelsize=20)

                MI, MA = [], []
                for p in pcs:
                    mi, ma = p.get_clim()
                    MI.append(mi)
                    MA.append(ma)
                for p in pcs:
                    # p.set_clim(min(MI), max(MA))
                    # p.set_clim(0, max(MA))
                    if v=='nотр':p.set_clim(0, 1)
                    else:p.set_clim(0, 2.5)



                    if min(MI) == max(MA): p.set_clim(min(MI) - 0.1, min(MI) + 0.1)
                if sub==(1,1): o='vertical'
                else:o='horizontal'
                cb = fig.colorbar(p, cax=fig.axes[-1], orientation=o, format=matplotlib.ticker.ScalarFormatter() )  # ,cax=,
                cb.ax.tick_params(labelsize=20)
                if v == 'n' or v == 'nup' or v == "nотр":
                    cb.set_label('$' + v[0] + '^{' + v[1:] + '}_{' + 'i' + '}' + '/n_{' + 'i' + '0}' + '$', fontsize=20)

                fig.subplots_adjust(**adj)
                save_name=result_dir_name + v + '_'+str(orientation_coord)+'_' + 't{:010.3F}'.format(time).replace('.', '_', 1) + '.png'
                fig.savefig(save_name)
                print(save_name)
                plt.close('all')