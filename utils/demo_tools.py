import numpy as np
import matplotlib.pyplot as plt


def show_single(image, save):
    # show single image
    image = np.array(image, dtype=np.uint8)
    h, w, c = image.shape

    fig, ax = plt.subplots()
    plt.imshow(image)

    fig.set_size_inches(w / 100.0, h / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis("off")
    if save:
        plt.savefig("../sample_demo/color_label.png", bbox_inches='tight', pad_inches=0)
    plt.show()