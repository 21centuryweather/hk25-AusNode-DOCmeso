'''
# -----------------
#  Make_into_gif
# -----------------

'''

# == imports ==
# -- Packages --
import os
import sys
import glob        
import imageio.v2 as imageio
import subprocess
import re


# == main ==
def main(folder):
    # -- get files --
    png_files = glob.glob(f"{folder}/*.png")
    png_summary_plot = [f for f in png_files if 'model-mean' in f or 'cross-model' in f]
    png_files = [f for f in png_files if f not in png_summary_plot]  # Exclude summary plots from main list

    # -- sort by number at the end, if there is one --
    png_without_number = sorted([f for f in png_files if not re.search(r'_(\d+)\.png$', f)])                # Alphabetical sort
    png_with_number = [f for f in png_files if re.search(r'_(\d+)\.png$', f)]
    png_with_number = sorted(png_with_number, key=lambda f: int(re.search(r'_(\d+)\.png$', f).group(1)))    # Numerical sort
    png_files = png_with_number + png_without_number

    # -- create gif --
    result_path = f'{os.path.splitext(__file__)[0]}_output.gif'
    duration = 0.5
    with imageio.get_writer(result_path, mode='I', duration=duration, loop=2) as writer:
        for path in png_files:
            image = imageio.imread(path)
            writer.append_data(image)
    print(f"Animation saved at: {result_path}")

    # # -- show gif --
    # subprocess.run(["code", result_path], check=True)

    # # -- show summary plot --
    # if len(png_summary_plot) > 0:
    #     for plot_path in png_summary_plot:
    #         subprocess.run(["code", plot_path], check=True)


# == when this script is ran / submitted ==
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 make_into_gif.py <folder_path>")
        exit()
    folder = sys.argv[1]
    main(folder)


