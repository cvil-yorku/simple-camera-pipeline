from .pipeline_utils import get_metadata, performInterpolation
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def rawpaths_from_dir(dir_path, rawtype=".raw"):
    """rawpaths_from_dir.
    Returns list of image paths within the parent directory

    Parameters
    ----------
    dir_path :
        Directory containing raw paths
    rawtype :
        '.DNG', '.TIFF', etc.
    """

    print("Dir_path:", dir_path)
    paths = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and rawtype in f
    ]

    return paths


def plot_3dhsv(lut, tag="lut"):
    # unpack 3d mesh
    V, H, S = lut
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(tag)
    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("V")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.scatter(H, S, V)
    print(f"Saving figure: {tag}")
    plt.savefig(f"./{tag}_testing.png")


def print_lut(lut):
    h, s, _, _ = lut.shape
    values = np.linspace(0, 1, 10)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("lut loop")

    for i in range(h):
        for j in range(s):
            print(lut[i, j])
            lut_conversion = lut[i, j][0]
            ax.scatter(
                lut_conversion[0],
                lut_conversion[1],
                lut_conversion[2],
            )

            # for v in values:
            # new_h = i * lut_conversion[0]
            # new_s = j + lut_conversion[1]
            # new_v = v + lut_conversion[2]
            # ax.scatter(
            # new_h,
            # new_s,
            # new_v,
            # )

    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("V")

    print("saving figure")
    plt.savefig(f"./lut_loop.png")


def apply_lut(grid, lut, idx):
    # TODO: clean this up, redifining / re slicing to get the same data
    # multiple times, want to be able to have the same data that is plotted
    # calculate for differences to be sure no errors are arising.

    V, H, S = grid
    converted_vals = np.zeros(grid.shape)

    print("H shape:", H.shape)
    print("S shape:", S.shape)
    print("V shape:", V.shape)
    print("LUT SHAPE:", lut.shape)

    # TODO: add parameter to toggle plotting...
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Corrected HSV values")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    print(grid.shape)

    _, i_max, j_max, k_max = grid.shape

    # Convert values from input HSV to output by applying LUT
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                h, s, v = H[i, j, k], S[i, j, k], V[i, j, k]

                # print(f"HSV in: {h}, {s}, {v}")

                hsv_coordinate = np.asarray(
                    [[[h]], [[s]], [[v]]], dtype=np.float32
                )

                hsv_coordinate = np.reshape(hsv_coordinate, (1, 1, 3))

                # print("HSV COORDINATE", hsv_coordinate)

                # NOTE: performInterpolation maintains positioning of HSV
                corrected = performInterpolation(hsv_coordinate, lut)[0, 0]
                corrected[2] = np.clip(corrected[2], 0, 1)

                # corrected[1] = np.clip(corrected[1], 0, 1)
                # corrected[2] = np.clip(corrected[2], 0, 1)
                # corrected = np.clip(corrected, 0, 1)

                # print(f"HSV corrected: {corrected}")

                converted_vals[0][i, j, k] = corrected[0]  # H
                converted_vals[1][i, j, k] = corrected[1]  # S
                converted_vals[2][i, j, k] = corrected[2]  # V

    return converted_vals


def plot_lut_val_slice(before_lut, after_lut, name="noname_camera"):
    # TODO: Add flag for whether to plot or not
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"Input data and LUT conversion diff for {name}")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    V, H, S = before_lut
    _, i_max, j_max, k_max = before_lut.shape

    # Plot each slice separately
    for plot_v_idx in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                plot_v = V[plot_v_idx, j, k]
                h, s = H[plot_v_idx, j, k], S[plot_v_idx, j, k]
                h_c, s_c = (
                    after_lut[0][plot_v_idx, j, k],
                    after_lut[1][plot_v_idx, j, k],
                )

                # print("BEFORE:", h, s, plot_v)
                # print("AFTER", h_c, s_c, plot_v)

                dx, dy, dz = h_c - h, s_c - s, 0

                # print("dx,dy,dz", dx, dy, dz)

                # To get around hue wrapping between 0-360
                if h > 10 and h < 350:
                    ax.scatter(h, s, plot_v, color="green", s=3)
                    ax.scatter(h_c, s_c, plot_v, color="red", s=3)

                    ax.arrow3D(
                        h,
                        s,
                        plot_v,
                        dx,
                        dy,
                        dz,
                        arrowstyle="-|>",
                        linestyle="dashed",
                        mutation_scale=5,
                    )

    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("V")

    print("saving figure")
    plt.savefig(f"./{name}_slice_visualization_allvalslices.png")
    plt.clf()


def afterlut_val_slice_diff(after_lut, name="noname_camera"):
    # Get difference between current value slice and all other slices that
    # aren't the current value slice ( if no difference, then no need for 3D
    # data )

    _, v_max, h_max, s_max = after_lut.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # TODO: Move this to separate function
    for val_idx in range(v_max):
        sl_1 = (
            after_lut[0][val_idx, :, :],
            after_lut[1][val_idx, :, :],
        )

        for compare_idx in range(v_max):
            if compare_idx != val_idx:
                sl_2 = (
                    after_lut[0][compare_idx, :, :],
                    after_lut[1][compare_idx, :, :],
                )

                print(
                    f"========== Value idx {val_idx} vs. {compare_idx} - LUT {name} =========="
                )
                print("H diff:", np.abs(sl_1[0] - sl_2[0]))
                print("S diff:", np.abs(sl_1[1] - sl_2[1]))


# Directory containing raw images
raw_dir = "/shared/data/autoexp-stack-dngs/"

rawpaths = rawpaths_from_dir(raw_dir, ".dng")

print(f"Paths of interest: {rawpaths}")

for index, path in enumerate(rawpaths):
    print(f"CURRENT PATH:{path}")

    # NOTE: Assumes image is named after camera. For now...
    camera_name = os.path.basename(path).strip(".dng")

    # Get metadata of image, by passing path name to image to the function
    metadata = get_metadata(path)

    # Get HSV and 3D lut from metadata
    hsv_lut = metadata["hsv_lut"]
    lut_3d = metadata["profile_lut"]

    print("LUT shape:", hsv_lut.shape)

    # Generate dummy data to apply LUTs to
    print("Generating dummy data to apply LUT")
    block_3d = np.mgrid[0:1:5j, 0:360:5j, 0:1:5j]
    print("Block Shape:", block_3d.shape)

    # Plot dummy data
    plot_3dhsv(block_3d, "Dummy HSV Vals")

    # Apply LUTS to dummy data
    hsv_lut_applied = apply_lut(block_3d, hsv_lut, camera_name)
    lut_3d_applied = apply_lut(block_3d, lut_3d, f"3dlut_{camera_name}")

    # Plot before / after of hsv lut
    plot_lut_val_slice(block_3d, hsv_lut_applied, name=f"{camera_name}_hsv")
    afterlut_val_slice_diff(hsv_lut_applied, name=f"{camera_name}_hsv")

    # Plot before / after of 3D lut
    plot_lut_val_slice(block_3d, lut_3d_applied, name=f"{camera_name}_3dlut")
    afterlut_val_slice_diff(lut_3d_applied, name=f"{camera_name}_3dlut")
