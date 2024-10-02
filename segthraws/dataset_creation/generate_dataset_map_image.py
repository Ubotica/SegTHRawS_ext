"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from matplotlib import font_manager
from mpl_toolkits.basemap import Basemap


from .constants import thraws_data_path,DATASET_PATH


font_dirs = [os.path.join(os.path.dirname(os.path.dirname(__file__)),'fonts','charter')]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
#Set Charter as default font
mpl.rc('font',family='Charter')


def extract_lat_long(input_string):

    # Define the regex pattern to match the latitude and longitude
    # pattern = r'lat(-?\d+_\d+)_long(-?\d+_\d+)'
    # pattern_2 = r'lat(-?\d+_\d+)_lon(-?\d+_\d+)'
    # pattern = r'lat(-?\d+_\d+)_long(-?\d+_\d+)'
    pattern = r'(\d{4})_lat(-?\d+_\d+)_(?:long|lon)(-?\d+_\d+)'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    # match_2 = re.search(pattern_2, input_string)
    
    if match:
        # Extract latitude and longitude from the matched groups
        year = int(match.group(1))
        lat =  float(match.group(2).replace('_', '.'))
        long = float(match.group(3).replace('_', '.'))
        return year,lat, long
    
    # elif match_2:
    #     lat = match_2.group(1).replace('_', '.')
    #     long = match_2.group(2).replace('_', '.')
    else:
        return None,None, None



def create_dataset_map_image(projection: str = 'merc',
                             data_path: str = thraws_data_path,
                             save_path: str = None,
                             new_images: bool = False,
                             satellite_view: bool = False,
                             save: bool = True,
                             show: bool = False,
                             ) -> None:

    map = Basemap(projection=projection,llcrnrlat=-70,urcrnrlat=70,\
                llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')


    map.drawcoastlines()
    map.drawcountries()

    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    # map.drawmapboundary(fill_color='aqua')
    # map.fillcontinents(color='coral',lake_color='aqua')
    if satellite_view:
        map.bluemarble() #Gets the satellite view 

    year_colors = {
        '2016': 'red',
        '2017': 'orange',
        '2018': 'yellow',
        '2019': 'lightgreen',
        '2020': 'green',
        '2021': 'cyan',
        '2022': 'blue',
        '2023': 'indigo',
        '2024': 'violet'
     }

    granule_year_list = [] # This will avoid repetitios in the legend

    scene_groups_paths = [
                os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
            ]
    for scene_group_path in scene_groups_paths:

        scenes_paths = [
                os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
            ]
        for scene_path in scenes_paths:
            granules_paths = [
                os.path.join(scene_path,granule_name) for granule_name in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, granule_name))
            ]

            for granule_path in granules_paths:
                granule_year = re.match(r'.*_(\d{4})+[0-9]+T[0-9]+_', os.path.basename(granule_path)).group(1)
                color = year_colors[granule_year]
                for file in os.listdir(granule_path):
                    if file[:2]=='S2' and file[-4:]=='.xml':
                        xml_name = file
                        break
                if not xml_name:
                    print(f'Error detected in: {granule_path}')
                else:
                    tree = ET.parse(os.path.join(granule_path,file))
                    coordinates = (tree.getroot().find(".//EXT_POS_LIST").text.split())
                    

                    # Extract individual coordinates
                    latitude = float(coordinates[0])
                    longitude = float(coordinates[1])
                    # altitude = float(coordinates[2])

                    # print("Latitude:", latitude)
                    # print("Longitude:", longitude)

                    x,y = map(longitude,latitude)
                    
                    map.scatter(x, y,color=color,label = granule_year if granule_year not in granule_year_list else '')
                    
                    granule_year_list.append(granule_year)
                    # plt.annotate(f'{scene_name}', xy=(x, y),  xycoords='data',
                    # xytext=(x2, y2), textcoords='offset points',
                    # color='r')



    ## New events included in the dataset 
    if new_images:

        images_path = os.listdir('/home/cristopher/Downloads/THRawS_ampliation')
        for image_path in images_path:
            year,latitude, longitude = extract_lat_long(image_path)
            color = year_colors[str(year)]
            
            x,y = map(longitude,latitude)

            map.scatter(x,y ,color=color,label = str(year) if str(year) not in granule_year_list else '')
                    
            granule_year_list.append(str(year))

        new_images = [(2023,19.8563, 102.4955), (2023,20.5720451,102.6088393),(2023,20.5208984,104.7266651), (2023, 0.9619, 114.5548), (2020, 28.7134, 17.9058), (2023 ,-12.637485, 132.269679), (2023 ,-15.429932, 125.870101), (2023 ,-17.902960, 122.875211), (2023 ,-19.045223, 134.035440)]

    

    for image in new_images:
        color = year_colors[str(image[0])]
        x,y = map(image[2],image[1])

        map.scatter(x,y ,color=color,label = '')


    


    #Order the legend to be chronologically correct
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = labels.copy()
    new_labels.sort()
    order = [labels.index(value) for value in new_labels]
    handles = [handles[i] for i in order]
    

    plt.legend(handles,new_labels,loc = 'lower left',fontsize=7)
    plt.tight_layout()
    plt.title( "Current THRawS dataset coverage")
    if save:
        if not save_path:
            save_path = os.path.join(DATASET_PATH,'dataset_map_distribution.png')
        plt.savefig(save_path,dpi=300)            
    if show:
        plt.show()


if __name__ == '__main__':

    # thraws_data_path = '/home/cristopher/datasets/THRawS'

    create_dataset_map_image(projection='merc',data_path = thraws_data_path,save=True,show=True)



    # data_path = thraws_data_path
    # scene_groups_paths = [
    #             os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    #         ]
    # print(scene_groups_paths)

#     import re
#     test = 'S2A_OPER_MSI_L0__GR_VGS1_20160205T054020_S20210205T022636_D12_N02.09'

#     year = re.match(r'.*_(\d{4})+[0-9]+T[0-9]+_', test).group(1)

#     year_colors = {
#         '2016': 'red',
#         '2017': 'orange',
#         '2018': 'yellow',
#         '2019': 'green',
#         '2020': 'blue',
#         '2021': 'indigo',
#         '2022': 'violet',

#      }

#     map = Basemap(projection='merc',llcrnrlat=-20,urcrnrlat=20,\
#                 llcrnrlon=-50,urcrnrlon=50,lat_ts=0,resolution='c')


#     map.drawcoastlines()
#     map.drawcountries()

#     # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
#     # map.drawmapboundary(fill_color='aqua')
#     # map.fillcontinents(color='coral',lake_color='aqua')
    
#     map.bluemarble() #Gets the satellite view

#     for i, year in enumerate(year_colors.keys()):

#         x,y = map(10+2*i,10)
        
#         map.scatter(x, y,color=year_colors[year],label = f'{year}')

#     plt.legend(loc = 'lower left')

#     plt.show()

   