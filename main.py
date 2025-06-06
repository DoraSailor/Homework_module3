import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt, pi #, fabs, copysign
import os
from func import make_map, read_color_map_array,read_and_paint_2d_plots,make_dens_field_vel_maps,make_n_subplots
from func import pe,pp,p_mag,betae,betap,likm,Omegainv,Va,mp,R_MOON


# при использовании в латехе \ перед " использовать модификатор r


print(f'{pe * 0.1} Pa,{pp * 0.1} Pa ,{p_mag * 0.1} Pa, betae={betae},betap={betap}')
print(
    f'di= {likm} km, Omega = {1 / Omegainv} 1/s, Va= {Va} km/s, u_swx/Va ={308.2 / Va},u_swy/Va ={-44.3 / Va},u_swz/Va ={-133.8 / Va},, Te/(mp*Va^2)={9.8 *  1.6e-12 / mp / Va / Va / 1e10}, Ti/(mp*Va^2)={2.4 *1.6e-12 / mp / Va / Va / 1e10}')
# print(mp * Va * Va * 1e10,)
time=0
dir_name='/home/dasha/Документы/НИР/data/dip/'
file_name_core='_lunar_test_'
TIMES=np.linspace(0, 1, 2)
adj=dict(left=0.095, bottom=0.085, right=0.985, top=0.94)
print(f'Bx={-2.5/3.85} ,By={2.9/3.85} ,Bz={-0.3/3.85}')
# result_dir_name=dir_name+'results/'+'color_scale/'
# if not os.path.isdir(dir_name + 'results'): os.mkdir(dir_name + '/results/')
# if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# num = 2
# rangee=1
# for i in range(rangee):
#     if not os.path.isdir(result_dir_name + str(i)):
#         os.mkdir(result_dir_name + str(i))
# for pref,suf,abss in zip(['field3d']*8,['_B', '_E','_Bx.z', '_By.z', '_Bz.z','_Ex.z', '_Ey.z', '_Ez.z'],[True]*2+[False]*6):
#     for time in TIMES:
#         print('field_scale: ',  time)
#         for orientation_coord, orientation_plane in zip((54,-5,-4,-3,-2,-1),('xz','yz','yz','yz','yz','yz',)):  # 54,-1,-2,  'xz','yz',
#             for moon_body, block_moon_body_vmax_vmin, moon_name in zip((True,), (True,), ('_moon_body',)):
#
#                 file_name = dir_name + pref + file_name_core + 't{:010.3F}'.format(time).replace('.', '_', 1) + suf
#                 print(file_name)
#                 plt.figure()
#                 make_map(file_name,abs=abss, coordinates='cells', orientation_coord=orientation_coord, orientation_plane=orientation_plane, moon_body=moon_body, block_moon_body_vmax_vmin=block_moon_body_vmax_vmin)
#                 Min, Max = plt.gci().get_clim()
#                 print(Min, Max)
#                 suff=suf.replace('.z', '')
#                 plt.title(suff[1:])
#                 if abss:
#                     abs_pref='abs_'
#                 else:
#                     abs_pref = ''
#
#                 for i in range(rangee):
#
#                     mini= Min + (Max - Min) / num * i
#                     maxi= Min + (Max - Min) / num * (i + 1)
#                     plt.gci().set_clim(mini,maxi)
#                     save_name=result_dir_name+str(i)+'/'+abs_pref+pref+file_name_core+str(orientation_coord)+orientation_plane+moon_name+suff+('_t{:010.3F}'.format(time)+'_vmin'+str(mini)+'_vmax'+str(maxi)).replace('.', '_', 3)+'.png'
#                     print('save_name', save_name)
#                     plt.savefig(save_name)
#                 plt.close('all')


# result_dir_name=dir_name+'results/'+'dens_field_vel/'
# if not os.path.isdir(dir_name + '/results'): os.mkdir(dir_name + 'results/')
# if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# for pref,suf,abss in zip(['field3d']*2+['vel3d']+['dens3d']+['field3d']*6+['vel3d']*3, ['_B','_E','_V','.z','_Bx.z', '_By.z', '_Bz.z','_Ex.z', '_Ey.z', '_Ez.z','_Vx.z', '_Vy.z', '_Vz.z'],[True]*3+[False]*10):
#     suff = suf.replace('.z','')
#     if suff == '': suff = '_' + pref[:-2]
#     print(suff)
#     if  not os.path.isdir(result_dir_name +suff[1:]) : os.mkdir(result_dir_name + suff[1:])
#     for time in TIMES:
#         for orientation_coord, orientation_plane in zip((54,-1,-2,-3,-4,-5,),('xz','yz','yz','yz','yz','yz',)): #54,-1,-2,  'xz','yz',
#             for moon_body,block_moon_body_vmax_vmin,moon_name in zip((True,),(True,),('_moon_body_',)):
#                 print(pref[:-2]+': ', time)
#                 file_name = dir_name+pref+file_name_core+'t{:010.3F}'.format(time).replace('.', '_', 1)+suf
#                 print(file_name)
#                 plt.figure()
#                 make_map(file_name, orientation_coord=orientation_coord, orientation_plane=orientation_plane, abs=abss,
#                          moon_body=moon_body, block_moon_body_vmax_vmin=block_moon_body_vmax_vmin)
#                 plt.title(suff[1:])
#                 if suff[1]=='B':plt.gci().colorbar.set_label(suff[1:]+"$/B_0$")
#                 elif suff[1]=='V': plt.gci().colorbar.set_label(suff[1:]+"$/V_a$")
#                 elif suff[1]=='E':plt.gci().colorbar.set_label(suff[1:]+"$/ (B_0*V_a/c)$")
#                 elif suff[1]=='dens':plt.gci().colorbar.set_label("$n/n_0$")
#                 plt.subplots_adjust(**adj)
#                 if abss:
#                     abs_pref = 'abs_'
#                 else:
#                     abs_pref = ''
#                 save_name = result_dir_name + suff[1:] + '/' + abs_pref + pref + file_name_core + str(
#                             orientation_coord) + orientation_plane + moon_name + 't{:010.3F}'.format(
#                             time).replace('.', '_', 1) + suff + '.png'
#
#                 plt.savefig(save_name)
#                 plt.close('all')
# #
#
#
result_dir_name=dir_name+'results/'+'n/'
if not os.path.isdir(dir_name + 'results'): os.mkdir(dir_name + '/results/')
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# for pref,suf in zip(['n','nup'],['.z']*2):
#     for time in TIMES:
#         file_name = dir_name+pref+file_name_core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+suf
#         plt.figure()
#         make_map(file_name)
#         if pref=='n':plt.gci().colorbar.set_label("$n/n_0$")
#         elif pref=='nup':plt.gci().colorbar.set_label("$n_{up}/n_0$")
#
#
#         plt.subplots_adjust(**adj)
#         save_name = result_dir_name  + pref + file_name_core +'27_'+ 't{:010.3F}'.format(time).replace('.', '_', 1) + '.png'
#         plt.savefig(save_name)
#     plt.close('all')
adj = dict(left=0.115, bottom=0.08, right=0.985, top=0.965)
# for time in TIMES:
# for time in [0,]:
#     # plt.figure()
#     filename1=dir_name+'n'+file_name_core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#     filename2=dir_name+'nup'+file_name_core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#     data1=read_color_map_array(filename1)
#     data2=read_color_map_array(filename2)
#     x1 = 20 - (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm/R_MOON*180/np.pi
#
#     # y1 = 123.5 - (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm/R_MOON*180/np.pi
#     # x11 = 20 +(np.linspace(data1[3], data1[4], data1[1] + 1) - (data1[4] - data1[3]) / 2) * likm / R_MOON * 180 / np.pi
#     # y11 = -123.5 + (np.linspace(data1[5], data1[6], data1[2] + 1) - (data1[6] - data1[5]) / 2) * likm / R_MOON * 180 / np.pi
#     x1 = np.linspace(data1[3], data1[4], data1[1] + 1) * likm
#     y1 = np.linspace(data1[5], data1[6], data1[2] + 1) * likm
#
#     # print(x1,y1)
#     # x1 = (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm
#     # y1 = (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm
#     # x1 = np.linspace(data1[3], data1[4], data1[1] + 1)
#     # y1 = np.linspace(data1[5], data1[6], data1[2] + 1)
#     # x2 = np.linspace(data2[3], data2[4], data2[1] + 1)*likm/R_MOON
#     # y2 = np.linspace(data2[5], data2[6], data2[2] + 1)*likm/R_MOON
#     print(data1[0]==data2[0])
#     # print(read_color_map_array(filename1))
#     for x in [x1,]:
#         for y in [y1]:
#             plt.figure(figsize=(10,9))
#             pc = plt.pcolormesh( y,x,(np.array(data1[0])-np.array(data2[0])).T)
#             # pc = plt.pcolormesh( np.array(data1[0]) - np.array(data2[0]))
#             plt.title("$n_{отр}$", fontsize=20)
#             plt.subplots_adjust(**adj)
#             # if y[0]>0:
#             #     plt.gca().add_patch(plt.Circle((22.9, 122.6), 86/2/R_MOON*180/np.pi, color='g',alpha=0.5))
#             #     plt.gca().add_patch(plt.Circle((22.3, 121.6), 26/2/R_MOON*180/np.pi, color='b',alpha=0.5))
#             #     plt.gca().add_patch(plt.Circle((24.1, 125.9), 55/2/R_MOON*180/np.pi, color='r',alpha=0.5))
#             # else:
#             #     plt.gca().add_patch(plt.Circle((22.9, -122.6), 86 / 2 / R_MOON * 180 / np.pi, color='g', alpha=0.5))
#             #     plt.gca().add_patch(plt.Circle((22.3, -121.6), 26 / 2 / R_MOON * 180 / np.pi, color='b', alpha=0.5))
#             #     plt.gca().add_patch(plt.Circle((24.1, -125.9), 55 / 2 / R_MOON * 180 / np.pi, color='r', alpha=0.5))
#             # plt.gca().add_patch(plt.Circle((122.6, 22.9, ), 86 / 2 / R_MOON * 180 / np.pi, color='orange', alpha=0.5))
#             # plt.gca().add_patch(plt.Circle((121.6, 22.3, ), 26/2/R_MOON*180/np.pi, color='m',alpha=0.5))
#             # plt.gca().add_patch(plt.Circle((125.9, 24.1, ), 55/2/R_MOON*180/np.pi, color='r',alpha=0.5))
#             #     (20 - x1)*R_MOON/180*np.pi +(data1[4]- data1[3])/2*likm=  np.linspace(data1[3], data1[4], data1[1] + 1)*likm
#             a= plt.gca().add_patch(plt.Circle(((123.5-122.6)/180*np.pi*R_MOON+(data1[6]- data1[5])/2*likm,
#                                             (20-22.9)/180*np.pi*R_MOON+(data1[4]- data1[3])/2*likm),
#                                            86 / 2 , color='white', alpha=0.5))
#             plt.gca().add_patch(plt.Circle(((123.5-121.6)/180*np.pi*R_MOON+(data1[6]- data1[5])/2*likm,(20-22.3)/180*np.pi*R_MOON+(data1[4]- data1[3])/2*likm, ),
#                 26/2, color='r',alpha=0.5)) # (121.6, 22.3, ),
#             plt.gca().add_patch(plt.Circle(((123.5-125.9)/180*np.pi*R_MOON+(data1[6]- data1[5])/2*likm,(20-24.1)/180*np.pi*R_MOON+(data1[4]- data1[3])/2*likm, ),
#                                            55/2, color='orange',alpha=0.5))#(125.9, 24.1, )
#             print(data1)
#
#             print(a)
#             plt.scatter([], [], c=['white'], s=100, label='Gerasimovich')
#             plt.scatter([], [], c=['r'], s=100, label='D',)
#             plt.scatter([], [], c=['orange'], s=100, label='R')
#             # plt.gca().text(121.6, 22.3,'D', alpha=0.5)
#             # plt.gca().text(125.9, 24.1,'R', alpha=0.5)
#             plt.legend(prop={'size': 20})
#
#     # plt.ylabel( '° S', labelpad=0)
#     # plt.xlabel('° W', labelpad=0)
#     plt.gca().tick_params(labelsize=20)
#     plt.xlabel('z' + ',$km$', labelpad=0, fontsize=20)
#     plt.ylabel('y' + ',$km$',labelpad=0, fontsize=20)
#     cb=plt.colorbar(pc,)
#     cb.set_label(label="$n_{отр}/n_0$",fontsize=20)
#     cb.ax.tick_params(labelsize=20)
#     save_name = result_dir_name + "ndown" +'_km' +'zy'+ file_name_core + '27_' + 't{:010.3F}'.format(time).replace('.', '_',1)+ '.png'
#     # plt.savefig(save_name)
# # plt.close()
# plt.show()
# #
# file_name = dir_name+'field3d'+file_name_core+'t{:010.3F}'.format(time).replace('.', '_', 1)+'_B'
#
# make_map(file_name,coordinates='selenographic', orientation_coord=-5, orientation_plane='zy', abs=True,
#                          moon_body=True, block_moon_body_vmax_vmin=True)
# plt.title('B')
#
# plt.gca().add_patch(plt.Circle((122.6, 22.9, ), 86 / 2 / R_MOON * 180 / np.pi, color='orange', alpha=0.5))
# plt.gca().add_patch(plt.Circle((121.6, 22.3, ), 26/2/R_MOON*180/np.pi, color='m',alpha=0.5))
# plt.gca().add_patch(plt.Circle((125.9, 24.1, ), 55/2/R_MOON*180/np.pi, color='r',alpha=0.5))
# plt.scatter([], [], c=['orange'], s=100, label='Gerasimovich')
# plt.scatter([], [], c=['m'], s=100, label='D')
# plt.scatter([], [], c=['r'], s=100, label='R')
# plt.legend()
# plt.show()

dir_name='/home/dasha/Документы/НИР/data/dip_He/'
file_name_core='_lunar_He_'
file_name_core_sorts=['_lunar_He_','_lunar_H_']
TIMES=np.linspace(0, 5, 6)
adj=dict(left=0.095, bottom=0.085, right=0.985, top=0.94)

result_dir_name=dir_name+'results/'+'dens_field_vel/'
# if not os.path.isdir(dir_name + '/results'): os.mkdir(dir_name + 'results/')
# if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# for pref,suf,abss in zip(['field3d']*2+['vel3d']+['dens3d']+['field3d']*6+['vel3d']*3, ['_B','_E','_V','.z','_Bx.z', '_By.z', '_Bz.z','_Ex.z', '_Ey.z', '_Ez.z','_Vx.z', '_Vy.z', '_Vz.z'],[True]*3+[False]*10):
#     suff = suf.replace('.z','')
#     if suff == '': suff = '_' + pref[:-2]
#     print(suff)
#     if  not os.path.isdir(result_dir_name +suff[1:]) : os.mkdir(result_dir_name + suff[1:])
#     for time in TIMES:
#         for orientation_coord, orientation_plane in zip((54,-1,-2,-3,-4,-5,),('xz','yz','yz','yz','yz','yz',)): #54,-1,-2,  'xz','yz',
#             for moon_body,block_moon_body_vmax_vmin,moon_name in zip((True,),(True,),('_moon_body_',)):
#                 print(pref[:-2]+': ', time)
#                 file_name = dir_name+pref+file_name_core+'t{:010.3F}'.format(time).replace('.', '_', 1)+suf
#                 print(file_name)
#                 plt.figure()
#                 make_map(file_name, orientation_coord=orientation_coord, orientation_plane=orientation_plane, abs=abss,
#                          moon_body=moon_body, block_moon_body_vmax_vmin=block_moon_body_vmax_vmin)
#                 plt.title(suff[1:])
#                 if suff[1]=='B':plt.gci().colorbar.set_label(suff[1:]+"$/B_0$")
#                 elif suff[1]=='V': plt.gci().colorbar.set_label(suff[1:]+"$/V_a$")
#                 elif suff[1]=='E':plt.gci().colorbar.set_label(suff[1:]+"$/ (B_0*V_a/c)$")
#                 elif suff[1]=='dens':plt.gci().colorbar.set_label("$n/n_0$")
#                 plt.subplots_adjust(**adj)
#                 if abss:
#                     abs_pref = 'abs_'
#                 else:
#                     abs_pref = ''
#                 save_name = result_dir_name + suff[1:] + '/' + abs_pref + pref + file_name_core + str(
#                             orientation_coord) + orientation_plane + moon_name + 't{:010.3F}'.format(
#                             time).replace('.', '_', 1) + suff + '.png'
#
#                 plt.savefig(save_name)
#                 plt.close('all')
result_dir_name=dir_name+'results/'+'n/'
# if not os.path.isdir(dir_name + 'results'): os.mkdir(dir_name + '/results/')
# if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# for core in file_name_core_sorts:
#     for time in TIMES:
#         # plt.figure()
#         filename1=dir_name+'n'+core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         filename2=dir_name+'nup'+core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         data1=read_color_map_array(filename1)
#         data2=read_color_map_array(filename2)
#         # x1 = 20 - (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm/R_MOON*180/np.pi
#         # y1 = 123.5 - (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm/R_MOON*180/np.pi
#         # x1 = (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm
#         # y1 = (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm
#         x1 = np.linspace(data1[3], data1[4], data1[1] + 1)
#         y1 = np.linspace(data1[5], data1[6], data1[2] + 1)
#         print(data1[0]==data2[0])
#         # print(read_color_map_array(filename1))
#
#         # plt.figure()
#         if '_He_' in core:
#             sort='He'
#             temp=0.25/4
#         elif '_H_' in core:
#             sort='H'
#             temp=0.75
#         else: raise ValueError('bad sort')
#         pcs,save_names=[],[]
#         for data,n in zip([np.array(data1[0])/temp, np.array(data2[0])/temp, (np.array(data1[0])-np.array(data2[0]))/temp],['n','nup',"ndown"]):
#             fig, ax= plt.subplots()
#             pc = plt.pcolormesh(x1,y1,data)
#             pcs.append(pc)
#             save_names.append(result_dir_name + n + core + '27_' + 't{:010.3F}'.format(time).replace('.', '_',1) + '.png')
#             ax.set_title('$'+n[0]+'^{'+n[1:]+'}_{'+sort+'}$')#"$n_{down}$"
#             plt.subplots_adjust(**adj)
#             plt.colorbar(pc,  format="{x:.2F}", ax=ax,label='$'+n[0]+'^{'+n[1:]+'}_{'+sort+'}'+'/n_{'+sort+'0}'+'$')#,
#         MI,MA=[],[]
#         for pc in pcs:
#             mi,ma=pc.get_clim()
#             MI.append(mi)
#             MA.append(ma)
#
#         for pc,save_name in zip(pcs,save_names):
#             pc.set_clim(min(MI),max(MA))
#             plt.savefig(save_name)
#         # plt.show()
#         plt.close('all')
#         # plt.show()
#         # fig_up, ax_up = plt.subplots()
#         # pc_up = plt.pcolormesh( x1,y1,(np.array(data2[0]))/temp)
#         #
#         #
#         # fig_down, ax_down = plt.subplots()
#         # pc_down = plt.pcolormesh( x1,y1,(np.array(data1[0])-np.array(data2[0]))/temp)
#         # pc = plt.pcolormesh( np.array(data1[0]) - np.array(data2[0]))
#
#
#         # plt.gca().add_patch(plt.Circle((122.6, 22.9, ), 86 / 2 / R_MOON * 180 / np.pi, color='orange', alpha=0.5))
#         # plt.gca().add_patch(plt.Circle((121.6, 22.3, ), 26/2/R_MOON*180/np.pi, color='m',alpha=0.5))
#         # plt.gca().add_patch(plt.Circle((125.9, 24.1, ), 55/2/R_MOON*180/np.pi, color='r',alpha=0.5))
#         # plt.scatter([], [], c=['orange'], s=100, label='Gerasimovich')
#         # plt.scatter([], [], c=['m'], s=100, label='D')
#         # plt.scatter([], [], c=['r'], s=100, label='R')
#         # plt.legend()
#         # plt.ylabel( '° S', labelpad=0)
#         # plt.xlabel('° W', labelpad=0) #+'_selenografic'



dir_name1='/home/dasha/Документы/НИР/data/dip/'
dir_name2='/home/dasha/Документы/НИР/data/dip_He/'

file_name_core1='_lunar_test_'
file_name_core2='_lunar_He_'
file_name_core1_sorts=['_lunar_test_']
file_name_core2_sorts=['_lunar_H_','_lunar_He_',]
TIMES=np.linspace(9, 9, 1)
adj=dict(left=0.095, bottom=0.085, right=0.985, top=0.94)

result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_He/'+'results/'
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
result_dir_name=result_dir_name+'n на одной фигуре/'
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
# for time in TIMES:
#     pcs, save_names = [], []
#     for dn,core in zip([dir_name1,dir_name2,dir_name2],file_name_core1_sorts+file_name_core2_sorts):
#         # plt.figure()
#         filename1=dn+'n'+core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         filename2=dn+'nup'+core+'27_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         data1=read_color_map_array(filename1)
#         data2=read_color_map_array(filename2)
#         # x1 = 20 - (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm/R_MOON*180/np.pi
#         # y1 = 123.5 - (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm/R_MOON*180/np.pi
#         # x1 = (np.linspace(data1[3], data1[4], data1[1] + 1)-(data1[4]- data1[3])/2)*likm
#         # y1 = (np.linspace(data1[5], data1[6], data1[2] + 1)-(data1[6]- data1[5])/2)*likm
#         x1 = np.linspace(data1[3], data1[4], data1[1] + 1)
#         y1 = np.linspace(data1[5], data1[6], data1[2] + 1)
#         print(data1[0]==data2[0])
#         # print(read_color_map_array(filename1))
#         # plt.figure()
#         if '_He_' in core:
#             sort='He'
#             temp=0.25/4
#         elif '_H_' in core:
#             sort='H'
#             temp=0.75
#         elif '_test_' in core:
#             sort='test'
#             temp=1
#         else: raise ValueError('bad sort')
#         pcs.append([])
#         save_names.append([])
#         for data,n in zip([np.array(data1[0])/temp, np.array(data2[0])/temp, (np.array(data1[0])-np.array(data2[0]))/temp],['n','nup',"ndown"]):
#             fig, ax= plt.subplots(1,3)
#             pc = plt.pcolormesh(x1,y1,data)
#             pcs[-1].append(pc)
#             save_names[-1].append(result_dir_name + n + core + '27_' + 't{:010.3F}'.format(time).replace('.', '_',1) + '.png')
#             ax.set_title('$'+n[0]+'^{'+n[1:]+'}_{'+sort+'}$')#"$n_{down}$"
#             plt.subplots_adjust(**adj)
#             plt.colorbar(pc,  format="{x:.2F}", ax=ax,label='$'+n[0]+'^{'+n[1:]+'}_{'+sort+'}'+'/n_{'+sort+'0}'+'$')#,
#     pcs=np.array(pcs)
#     save_names=np.array(save_names)
#     for pp,ss in zip(pcs.T, save_names.T):
#         MI, MA = [], []
#         for p,s in zip(pp,ss):
#             mi, ma = pc.get_clim()
#             MI.append(mi)
#             MA.append(ma)
#             print(p,s)
#         for p, s in zip(pp, ss):
#             p.set_clim(min(MI), max(MA))
#             p.figure.savefig(s)
#
#         print('')
#     plt.close('all')
adj=dict(left=0.025, bottom=0.065, right=0.985, top=0.96,wspace=0.1,hspace=0.1)
#

adj=dict(left=0.085, bottom=0.095, right=0.985, top=0.94)
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip/','_lunar_test_',14,15,1,result_dir_name='/home/dasha/Документы/НИР/data/dip/results')
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_He/','_lunar_He_',9,result_dir_name='/home/dasha/Документы/НИР/data/dip_He/results',adj=adj)

# make_n_subplots('/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_He/',
#                 ['_lunar_test_'],['_lunar_H_','_lunar_He_',],
#                 0,9,result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_He/results1',
#                 vmax=0.3,vmin=0,coordinates='km')

# make_map(dir_name+'field3d'+file_name_core+'t{:010.3F}'.format(time).replace('.', '_', 1)+'_B',abs=True,orientation_plane='xy',orientation_coord=54,moon_body=True,block_moon_body_vmax_vmin=True,streamlines=True)
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_He/','_lunar_He_',0,8,1,result_dir_name='/home/dasha/Документы/НИР/data/dip_He/results',adj=adj)

# plt.show()

adj=dict(left=0.035, bottom=0.065, right=0.985, top=0.96,wspace=0.1,hspace=0.1)
#
result_dir='/home/dasha/Документы/НИР/data/dip and dip_He/'+'results/'
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
#
#
prefs,sufs ,absss,streamlines=['field3d']*2+['vel3d'],['_B','_E','_V'],[True]*3,[True]*3
# prefs,sufs ,absss,streamlines=['field3d'],['_B',],[True],[True]
prefs,sufs ,absss,streamlines=['vel3d'],['_V',],[True],[True]

TIMES=np.linspace(0, 14, 15)

sl_suff='_sl'
orientation_coord , orientation_plane ,moon_name=54,'xy','_moon_body_'
figs=[]
save_names=[]
#
# for pref,suf,abss,sl in zip(prefs,sufs ,absss,streamlines):
#     for time in TIMES:
#         result_dir_name = result_dir +'/'+ suf[1:]+'/'
#         if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
#         fig=plt.figure(figsize=(16, 9))
#         if suf=="_V":mx=0 #9.68
#         else:mx=0
#         #
#         print(suf)
#         grid = plt.GridSpec(30, 2)
#         axs=[fig.add_subplot(grid[:-3, 0]), fig.add_subplot(grid[:-3, 1]), ]
#         plt.sca(axs[0])
#         file_name1 = dir_name1 + pref + file_name_core1 + 't{:010.3F}'.format(time).replace('.', '_', 1) + suf
#         p1=make_map(file_name1, orientation_coord=orientation_coord, orientation_plane=orientation_plane, abs=abss,
#                  moon_body=True, block_moon_body_vmax_vmin=True, streamlines=sl,colorbar=False,meanx=mx)
#
#         plt.title(suf[1:]+"(H only)")
#         plt.sca(axs[1])
#         file_name2 = dir_name2 + pref + file_name_core2 + 't{:010.3F}'.format(time).replace('.', '_', 1) + suf
#
#         p2=make_map(file_name2, orientation_coord=orientation_coord, orientation_plane=orientation_plane, abs=abss,
#                  moon_body=True, block_moon_body_vmax_vmin=True, streamlines=sl,colorbar=False,meanx=mx)
#         # ar =p2[3][-1]-p1[3][-1]
#
#         plt.title(suf[1:]+"(H+He)")
#         MI, MA = [], []
#         for p in [p1[2],p2[2]]:
#             mi, ma = p.get_clim()
#             MI.append(mi)
#             MA.append(ma)
#         for p in [p1[2],p2[2]]:
#             # print(n, min(MI), max(MA))
#             p.set_clim(min(MI), max(MA))
#             if min(MI) == max(MA): p.set_clim(min(MI) - 0.1, min(MI) + 0.1)
#             # p.set_clim(0, 15)
#         fig.colorbar(p, cax=fig.add_subplot(grid[-1, :]), orientation='horizontal', format="{x:.2F}",)  # ,
#         if suf[1] == 'B':
#             plt.gci().colorbar.set_label(suf[1:] + "$/B_0$")
#         elif suf[1] == 'V':
#             plt.gci().colorbar.set_label(suf[1:] + "$/V_a$")
#         elif suf[1] == 'E':
#             plt.gci().colorbar.set_label(suf[1:] + "$/ (B_0*V_a/c)$")
#         elif suf[1] == 'dens':
#             plt.gci().colorbar.set_label("$n/n_0$")
#
#         fig.subplots_adjust(**adj)
#         # plt.figure()
#         # p3=plt.gca().pcolormesh(ar)
#         # plt.colorbar(p3)
#
#         # plt.show()
#         save_name=result_dir_name + pref  + str(orientation_coord) + orientation_plane + moon_name+'t{:010.3F}'.format(time).replace('.', '_', 1) +sl_suff+ suf +suf +'x-'+str(mx)+ '.png'
#         print('!',save_name)
#         plt.savefig(save_name)
#         plt.close()

# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip/','_lunar_test_',0,result_dir_name='/home/dasha/Документы/НИР/data/dip/results',orientation=zip((54,),('xy',)),types=[['field3d'],['_B'],[True],[True]],coordinates='km',vmax=150)
# t,f=110,-123.5
# plt.plot([0,np.cos(t/180*np.pi)*np.sin(f/180*np.pi)],[0,-np.sin(f/180*np.pi)])
# plt.plot([0,np.cos(f/180*np.pi)],[0,0])
# print(((np.cos(t/180*np.pi)*np.sin(f/180*np.pi)**2+np.sin(f/180*np.pi)**2)**(1/2),np.cos(f/180*np.pi)))
# plt.show()
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip/','_lunar_test_',0,result_dir_name='/home/dasha/Документы/НИР/data/test',orientation=zip((-7,-8,),('zy','zy',)),types=[['field3d'],['_B'],[True],[False]],coordinates='km')

# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_He/','_lunar_He_',0,14,result_dir_name='/home/dasha/Документы/НИР/data/dip_He/results',orientation=zip((54,-1,-2,-3,-4,-5,-6,-7,-8),('xy','yz','yz','yz','yz','yz','yz','yz','yz',)),types='all',coordinates='km')

# make_n_subplots('/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_He/',
#                 ['_lunar_test_'],['_lunar_H_','_lunar_He_',],
#                 0,14,result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_He/results',
#                 coordinates='km')#vmax=0.3,vmin=0

# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_hydro/','_lunar_test_',7,15,orientation=zip((54,-1,-2,-3,-4,-5,-6),('xy','yz','yz','yz','yz','yz','yz')),types='all',coordinates='km',)
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_hydro/','_lunar_test_',7,15,orientation=zip((54,-1,-2,-3,-4,-5,-6),('xy','yz','yz','yz','yz','yz','yz')),types=[['vel3d'],['_V'],[True],[True]],coordinates='km',meanx=9.68)

# make_n_subplots('/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_He/',
#                 ['_lunar_test_'],['_lunar_H_','_lunar_He_',],
#                 0,14,result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_He/results_test',
#                 coordinates='km')#vmax=0.3,vmin=0
#
# make_dens_field_vel_maps(['/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_hydro/'],['_lunar_test_','_lunar_test_'],
#                          0,6,orientation=zip((54,),('xy',)),
#                          types=[['vel3d'],['_V'],[True],[True]],coordinates='km',meanx=9.68,
#                          result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_He/results_test',sub=(1,2))

adj=dict(left=0.065, bottom=0.065, right=0.975, top=0.96,wspace=0.25,hspace=0.1)
# make_dens_field_vel_maps(['/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_hydro/'],['_lunar_test_','_lunar_test_'],
#                          7,15,orientation=zip((54,-1,-2,-3,-4,-5,-6,-7,-8),('xy','yz','yz','yz','yz','yz','yz','yz','yz')),
#                          types='all',coordinates='km',meanx=0,
#                          result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_hydro',sub=(1,2),adj=adj)

# make_n_subplots('/home/dasha/Документы/НИР/data/dip/','/home/dasha/Документы/НИР/data/dip_hydro/',
#                 ['_lunar_test_'],['_lunar_test_',], [27],
#                 7,15,result_dir_name='/home/dasha/Документы/НИР/data/dip and dip_hydro',
#                 coordinates='km')#vmax=0.3,vmin=0
file_name_core='_lunar_test_'
result_dir_name='/home/dasha/Документы/НИР/data/dip_hydro/n/'
dir_name='/home/dasha/Документы/НИР/data/dip_hydro/'
TIMES=np.linspace(0, 15, 16)
# for pref,suf in zip(['n','nup'],['.z']*2):
#     for time in TIMES:
#         for co in[25,26,27]:
#             file_name = dir_name+pref+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+suf
#             plt.figure()
#             make_map(file_name)
#             if pref=='n':plt.gci().colorbar.set_label("$n/n_0$",fontsize=12)
#             elif pref=='nup':plt.gci().colorbar.set_label("$n_{up}/n_0$",fontsize=20)
#
#
#             # plt.subplots_adjust(**adj)
#             save_name = result_dir_name  + pref + file_name_core +str(co)+'_'+ 't{:010.3F}'.format(time).replace('.', '_', 1) + '.png'
#             print(save_name)
#             plt.savefig(save_name)
#             plt.close('all')

# for time in TIMES:
#     for co in[25,26,27]:
#         filename1=dir_name+'n'+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         filename2=dir_name+'nup'+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         data1=read_color_map_array(filename1)
#         data2=read_color_map_array(filename2)
#         x1 = np.linspace(data1[3], data1[4], data1[1] + 1) * likm
#         y1 = np.linspace(data1[5], data1[6], data1[2] + 1) * likm
#         print(data1[0]==data2[0])
#         for x in [x1,]:
#             for y in [y1]:
#                 plt.figure(figsize=(10,9))
#                 pc = plt.pcolormesh( x,y,(np.array(data1[0])-np.array(data2[0])))
#                 plt.title("$n_{отр}$", fontsize=20)
#                 plt.subplots_adjust(**adj)
#
#         plt.gca().tick_params(labelsize=20)
#         plt.xlabel('z' + ',$km$', labelpad=0, fontsize=20)
#         plt.ylabel('y' + ',$km$',labelpad=0, fontsize=20)
#         cb=plt.colorbar(pc,)
#         cb.set_label(label="$n_{отр}/n_0$",fontsize=20)
#         cb.ax.tick_params(labelsize=20)
#         save_name = result_dir_name + "ndown"  + file_name_core +str(co)+'_' + 't{:010.3F}'.format(time).replace('.', '_',1)+ '.png'
#         plt.savefig(save_name)
#         plt.close()

adj=dict(left=0.12, bottom=0.075, right=0.975, top=0.98,wspace=0.25,hspace=0.1)

dir_name,file_name_core='/home/dasha/Документы/НИР/data/dip_hydro_nonzero/','_lunar_test_'
TIMES=np.linspace(0,32,33)
result_dir_name=dir_name+'/results'
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)
result_dir_name=result_dir_name+'/n/'
if not os.path.isdir(result_dir_name): os.mkdir(result_dir_name)

# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/dip_hydro_nonzero/','_lunar_test_',0,32,orientation=zip((54,-1,-2,-3,-4,-5,-6),('xy','yz','yz','yz','yz','yz','yz')),types='E',coordinates='km',)

# for pref,suf in zip(['n','nup'],['.z']*2):
#     for time in TIMES:
#         for co in[25,26,27]:
#             file_name = dir_name+pref+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+suf
#             plt.figure(figsize=(10, 9))
#             make_map(file_name,coordinates='km', colorbar_prop={'format': "{x:.2F}"})
#             if pref=='n':plt.gci().colorbar.set_label("$n/n_0$",fontsize=20)
#             elif pref=='nup':plt.gci().colorbar.set_label("$n_{up}/n_0$",fontsize=20)
#             plt.gca().tick_params(labelsize=20)
#             plt.xlabel('z' + ',$km$', labelpad=0, fontsize=20)
#             plt.ylabel('y' + ',$km$', labelpad=0, fontsize=20)
#             plt.gci().colorbar.ax.tick_params(labelsize=20)
#
#             plt.subplots_adjust(**adj)
#             save_name = result_dir_name  + pref + file_name_core +str(co)+'_'+ 't{:010.3F}'.format(time).replace('.', '_', 1) + '.png'
#             print(save_name)
#             plt.savefig(save_name)
#             plt.close('all')
# adj=dict(left=0.12, bottom=0.075, right=0.955, top=0.98,wspace=0.25,hspace=0.1)
#
# for time in TIMES:
#     for co in[25,26,27]:
#         filename1=dir_name+'n'+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         filename2=dir_name+'nup'+file_name_core+str(co)+'_'+'t{:010.3F}'.format(time).replace('.', '_', 1)+'.z'
#         data1=read_color_map_array(filename1)
#         data2=read_color_map_array(filename2)
#         x1 = np.linspace(data1[3], data1[4], data1[1] + 1) * likm
#         y1 = np.linspace(data1[5], data1[6], data1[2] + 1) * likm
#         print(data1[0]==data2[0])
#         plt.figure(figsize=(10,9))
#         pc = plt.pcolormesh( x1,y1,(np.array(data1[0])-np.array(data2[0])))
#         # plt.title("$n_{отр}$", fontsize=20)
#         plt.subplots_adjust(**adj)
#
#         plt.gca().tick_params(labelsize=20)
#         plt.xlabel('z' + ',$km$', labelpad=0, fontsize=20)
#         plt.ylabel('y' + ',$km$',labelpad=0, fontsize=20)
#         cb=plt.colorbar(pc,format="{x:.2F}")
#         cb.set_label(label="$n_{отр}/n_0$",fontsize=20)
#         cb.ax.tick_params(labelsize=20)
#         save_name = result_dir_name + "ndown"  + file_name_core +str(co)+'_' + 't{:010.3F}'.format(time).replace('.', '_',1)+ '.png'
#         plt.savefig(save_name)
#         plt.close()
adj=dict(left=0.07, bottom=0.075, right=0.975, top=0.98,wspace=0.2,hspace=0.1)
# make_dens_field_vel_maps(['/home/dasha/Документы/НИР/data/dip_hydro/','/home/dasha/Документы/НИР/data/dip_hydro_nonzero/'],['_lunar_test_','_lunar_test_'],
#                         0,15,orientation=zip((54,-1,-2,-3,-4,-5,-6,-7,-8),('xy','yz','yz','yz','yz','yz','yz','yz','yz')),
#                          types='E',coordinates='km',meanx=0,
#                          result_dir_name='/home/dasha/Документы/НИР/data/dip_hydro_nonzero/results',sub=(1,2),adj=adj)

# make_n_subplots('/home/dasha/Документы/НИР/data/dip_hydro/','/home/dasha/Документы/НИР/data/dip_hydro_nonzero/',
#                 ['_lunar_test_'],['_lunar_test_',], [25,26,27],
#                 0,15,result_dir_name='/home/dasha/Документы/НИР/data/dip_hydro_nonzero/results',
#                 coordinates='km')#vmax=0.3,vmin=0
# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/fatemi_1/','_lunar_fatemi_',0,40,orientation=zip((54,-1,-2,-3,-4,-5,-6),('xy','yz','yz','yz','yz','yz','yz')),types='all',coordinates='km',)

# make_dens_field_vel_maps('/home/dasha/Документы/НИР/data/fatemi_2/','_lunar_fatemi_',0,40,orientation=zip((54,-1,-2,-3,-4,-5,-6),('xy','yz','yz','yz','yz','yz','yz')),types='all',coordinates='km',)
nipy_spectral = matplotlib.colormaps['nipy_spectral'].resampled(256)
newcolors = nipy_spectral(np.linspace(0.95, 0.97, 10))
# newcolors=[[_**2,(_-1)**2,0,1]for _ in np.linspace(0,1,10)]#[1,0,0,1],[0.5,0.5,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1]

# print(newcolors)
# make_map('/home/dasha/Документы/НИР/data/fatemi_2/field3d_lunar_fatemi_t000006_000_Bx.z',cmap=matplotlib.colors.ListedColormap(newcolors))
# plt.figure()
# make_map('/home/dasha/Документы/НИР/data/fatemi_2/field3d_lunar_fatemi_t000006_000_Bx.z',cmap=matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", newcolors))

# make_n_subplots('/home/dasha/Документы/НИР/data/fatemi_2/',
#                 ['_lunar_fatemi_',], [24,25,26,27],
#                 0,result_dir_name='/home/dasha/Документы/НИР/data/fatemi_2/results/test/',
#                 coordinates='km',cmap= matplotlib.colors.ListedColormap(newcolors))#vmax=0.3,vmin=0


plt.show()

# os.system('spd-say "done";echo -e "\a"; notify-send -u critical -w "pycharm finished working"')
