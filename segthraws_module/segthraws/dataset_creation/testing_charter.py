from constants import SEGTHRAWS_DIRECTORY, MAIN_DIRECTORY, paths_dict, masks_events_dirs, masks_potential_events_dirs, bands_list, plot_names
import os
# MAIN_DIRECTORY = os.path.dirname(__file__)


from matplotlib import font_manager


font_dirs = [os.path.join(SEGTHRAWS_DIRECTORY,'fonts','charter')]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# font_manager._get_fontconfig_fonts.cache_clear()



import matplotlib as mpl
mpl.rc('font',family='Charter')
# mpl.rc('font',family='Charter-Regular')
import matplotlib.pyplot as plt

plt.plot([1,2,3])

plt.title('hilarious')

plt.show()