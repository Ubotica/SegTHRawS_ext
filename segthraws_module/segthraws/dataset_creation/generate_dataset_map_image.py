import os
import re
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from mpl_toolkits.basemap import Basemap

from .constants import thraws_data_path,DATASET_PATH

def create_dataset_map_image(projection: str = 'merc',
                             data_path: str = thraws_data_path,
                             save_path: str = None,
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
        '2019': 'green',
        '2020': 'blue',
        '2021': 'indigo',
        '2022': 'violet',

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


    #Order the legend to be chronologically correct
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = labels.copy()
    new_labels.sort()
    order = [labels.index(value) for value in new_labels]
    handles = [handles[i] for i in order]
    

    plt.legend(handles,new_labels,loc = 'lower left')
    plt.tight_layout()
    if save:
        if not save_path:
            save_path = os.path.join(DATASET_PATH,'dataset_map_distribution.png')
        plt.savefig(save_path)            
    if show:
        plt.show()


if __name__ == '__main__':

    # thraws_data_path = '/home/cristopher/datasets/THRawS'

    create_dataset_map_image(projection='merc',data_path = thraws_data_path,save=True,show=True)
