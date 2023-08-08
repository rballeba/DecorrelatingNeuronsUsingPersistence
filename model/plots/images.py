import os
import matplotlib.pyplot as plt
import numpy as np

import imageio as imageio


def build_gif(gif_path, frames_per_image=2, extension='png'):
    # Build GIF
    filenames = list(map(lambda filename: int(filename[:-4]), os.listdir(gif_path)))
    filenames.sort()
    with imageio.get_writer(f'{gif_path}/result.gif', mode='I') as writer:
        for filename in [f'{gif_path}/{gif_filename}.{extension}' for gif_filename in filenames]:
            image = imageio.imread(filename)
            for _ in range(frames_per_image):
                writer.append_data(image)


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, list_titles=None, general_title=None, list_cmaps=None, grid=True, num_cols=2,
                    show_axis=False, figsize=(20, 10), title_fontsize=19, general_title_fontsize=30):
    """
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    :param show_axis: if False the axis is hidden, otherwise the axis is visible
    :param general_title: General title of the plot. Suptitle
    :param list_images: list
        List of the images to be displayed.
    :param list_titles: list or None
        Optional list of titles to be shown for each image.
    :param list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    :param grid: boolean
        If True, show a grid over each image
    :param num_cols: int
        Number of columns to show.
    :param figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    :param title_fontsize: int
        Value to be passed to set_title().
    :return fig, axes of matplotlib plot

    """
    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)
        if not show_axis:
            list_axes[i].axis('off')

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)
    if general_title is not None:
        fig.suptitle(general_title, fontsize=general_title_fontsize)
    fig.tight_layout()
    return fig, axes
