from .pipeline_utils import get_metadata, performInterpolation
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import pdb

results_dir = "./results"


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
    H, S, V = lut
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
    save_pth = os.path.join(results_dir, f"{tag}_testing.png")
    plt.savefig(save_pth)


def print_lut(lut, tag="noname_camera"):
    print(f"Lut shape: {lut.shape}")
    h, s, _, _ = lut.shape
    values = np.linspace(0, 1, 10)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"{tag} direct hsv lut plot")

    for i in range(h):
        for j in range(s):
            # print(lut[i, j])
            lut_conversion = lut[i, j][0]
            print(f"Lut at idx1:{i}, idx2:{j} -> {lut_conversion}")
            ax.scatter(
                lut_conversion[0],  # H
                lut_conversion[1],  # S
                lut_conversion[2],  # V
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
    save_pth = os.path.join(results_dir, f"{tag}_direct_hsvlut.png")
    plt.savefig(save_pth)


def print_val_lut(lut, tag="noname_camera"):
    print(f"Lut shape: {lut.shape}")
    h, s, _, _ = lut.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"{tag} direct hsv lut plot")

    plot_matrix = np.zeros((h, s))

    for i in range(h):
        for j in range(s):
            # print(lut[i, j])
            lut_conversion = lut[i, j][0]
            # ax.scatter(
            # i,  # H
            # j,  # S
            # lut_conversion[2],  # V
            # )

            plot_matrix[i, j] = np.clip(lut_conversion[2], 0, 1)

    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("V")
    ax.set_zlim(0, 1.8)

    xx, yy = np.mgrid[0:h, 0:s]
    ax.plot_surface(
        xx,
        yy,
        plot_matrix,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    print("saving figure")
    save_pth = os.path.join(results_dir, f"{tag}_direct_hsvlut_value.png")
    plt.savefig(save_pth)


def apply_lut(grid, lut, name="noname_camera_hsvorprofile_lut", plot=True):
    # TODO: clean this up, redifining / re slicing to get the same data
    # multiple times, want to be able to have the same data that is plotted
    # calculate for differences to be sure no errors are arising.

    V, H, S = grid
    converted_vals = np.zeros(grid.shape)
    hue_sat_plane = np.zeros((100, 100))

    print("H shape:", H.shape)
    print("S shape:", S.shape)
    print("V shape:", V.shape)
    print("LUT SHAPE:", lut.shape)
    print("GRID SHAPE", grid.shape)

    _, val_len, hue_len, sat_len = grid.shape

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_title(f"Input data and LUT conversion diff for {name}")
        # ax.set_xlim(0, 360)
        # ax.set_ylim(0, 1)
        ax.set_zlim(0, 2)

    # Convert values from input HSV to output by applying LUT
    for i in range(val_len):
        for j in range(hue_len):
            for k in range(sat_len):
                # pdb.set_trace()
                h, s, v = H[i, j, k], S[i, j, k], V[i, j, k]

                # print(f"HSV in: {h}, {s}, {v}")

                hsv_coordinate = np.asarray(
                    [[[h]], [[s]], [[v]]], dtype=np.float32
                )

                # print("HSV COORDINATE BEFORE RESHAPE", hsv_coordinate)
                hsv_coordinate = np.reshape(hsv_coordinate, (1, 1, 3))

                # print("HSV COORDINATE INPUT", hsv_coordinate)

                # NOTE: performInterpolation maintains positioning of HSV
                corrected = performInterpolation(hsv_coordinate, lut)[0, 0]
                # print("CORRECTED HSV", corrected)

                if plot:
                    h_in, s_in, v_in = hsv_coordinate[0, 0]
                    h_out, s_out, v_out = corrected

                    dx, dy, dz = h_out - h_in, s_out - s_in, v_out - v_in

                    ax.scatter(h_in, s_in, v_out, color="green", s=3)
                    hue_sat_plane[j, k] = v_out

                    # ax.scatter(h_out, s_out, v_out, color="red", s=3)

                    # ax.arrow3D(
                    # h_in,
                    # s_in,
                    # v_in,
                    # dx,
                    # dy,
                    # dz,
                    # arrowstyle="-|>",
                    # linestyle="dashed",
                    # mutation_scale=5,
                    # )

                converted_vals[0][i, j, k] = corrected[2]  # V
                converted_vals[1][i, j, k] = corrected[0]  # H
                converted_vals[2][i, j, k] = corrected[1]  # S

    if plot:
        print("saving figure")
        save_pth = os.path.join(results_dir, f"{name}_input_vs_output.png")
        plt.savefig(save_pth)
        plt.clf()

    return converted_vals


# WARN: BUG IN THIS FUNCTION
def plot_lut_val_slice(before_lut, after_lut, name="noname_camera"):
    print(f"plotting {name} arrow chart")
    print(f"Before lut shape {before_lut.shape}")
    print(f"After lut shape {after_lut.shape}")

    # TODO: Add flag for whether to plot or not
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"Input data and LUT conversion diff for {name}")
    # ax.set_xlim(0, 360)
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

                print("BEFORE:", h, s, plot_v)
                print("AFTER", h_c, s_c, plot_v)

                # BUG: should not be fixing v here
                dx, dy, dz = h_c - h, s_c - s, 0

                # print("dx,dy,dz", dx, dy, dz)

                # To get around hue wrapping between 0-360
                # if h > 10 and h < 350:
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
    save_pth = os.path.join(
        results_dir, f"{name}_slice_visualization_allvalslices.png"
    )
    plt.savefig(save_pth)
    plt.clf()


def afterlut_val_slice_diff(after_lut, name="noname_camera"):
    # Get difference between current value slice and all other slices that
    # aren't the current value slice ( if no difference, then no need for 3D
    # data )

    _, v_max, h_max, s_max = after_lut.shape

    # TODO: Plot the difference between a slice and all others as a heatmap

    for val_idx in range(v_max):
        sl_1 = (
            after_lut[0][val_idx, :, :],
            after_lut[1][val_idx, :, :],
        )

        for compare_idx in range(v_max):
            # TODO: calculate all comparisons and then plot, so
            # heatmaps have the same scale between all comparisons
            if compare_idx != val_idx:
                sl_2 = (
                    after_lut[0][compare_idx, :, :],
                    after_lut[1][compare_idx, :, :],
                )

                print(
                    f"========== Value idx {val_idx} vs. {compare_idx} - LUT {name} =========="
                )
                h_diff = sl_1[0] - sl_2[0]
                print("Hdiff max", np.abs(h_diff).max())

                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                ax.set_title(f"{name}, {val_idx} vs {compare_idx}")
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 4)
                ax.set_zlim(0, 4)

                for h_idx in range(h_diff.shape[0]):
                    for hval_idx in range(h_diff.shape[1]):
                        # TODO: normalize colors based on maximum across all
                        # differences so colors are standardized and not relative
                        # to each hue slice

                        norm = np.linspace(0, 1, 360)
                        colors = cm.rainbow(norm)

                        h_diff_color_idx = int(np.abs(h_diff[h_idx, hval_idx]))

                        # print("H diff color index:", h_diff_color_idx)
                        # print("COLOR:", colors[h_diff_color_idx])
                        # print("h idx", h_idx)
                        # print("val_idx", val_idx)
                        # print("compare_idx", compare_idx)

                        ax.scatter(
                            hval_idx,
                            compare_idx,
                            h_idx,
                            color=colors[h_diff_color_idx],
                        )

                ax.set_xlabel("V")
                ax.set_ylabel("V slice compare")
                ax.set_zlabel("H")
                norm = plt.Normalize(0, 360)
                fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow))
                # fig.colorbar(cm.ScalarMappable(norm=), ax=ax)
                save_pth = os.path.join(
                    results_dir,
                    f"{name}-validx-{val_idx}-compared-{compare_idx}.png",
                )
                plt.savefig(save_pth)
                plt.clf()

                # s_diff = sl_1[1] - sl_2[1]
                print("H diff:", np.abs(sl_1[0] - sl_2[0]))
                # print("S diff:", np.abs(sl_1[1] - sl_2[1]))


# Directory containing raw images
raw_dir = "/shared/data/autoexp-stack-dngs/"

rawpaths = rawpaths_from_dir(raw_dir, ".dng")

print(f"Paths of interest: {rawpaths}")

for index, path in enumerate(rawpaths[1:]):
    print(f"CURRENT PATH:{path}")

    # NOTE: Assumes image is named after camera. For now...
    camera_name = os.path.basename(path).strip(".dng")

    # Get metadata of image, by passing path name to image to the function
    metadata = get_metadata(path)

    # Get HSV LUT from Metadata
    hsv_lut = metadata["hsv_lut"]
    print(f"hsv_lut for {camera_name}", hsv_lut)
    print("HSV LUT shape:", hsv_lut.shape)

    # Get Profule LUT from Metadata
    # profile_lut = metadata["profile_lut"]
    # print(f"Profile lut for {camera_name}", profile_lut)
    # print(f"Profile lut shape", profile_lut.shape)

    # Generate dummy data to apply LUTs to
    print("Generating dummy data to apply LUT")
    # in dimensions -> v,h,s
    block_3d = np.mgrid[0.5:0.5:1j, 0:360:100j, 0:1:100j]
    print("Block Shape:", block_3d.shape)

    # Plot dummy data
    # plot_3dhsv(block_3d, "Dummy HSV Vals")

    # Apply LUTS to dummy data
    # profile_lut_applied = apply_lut(
    # block_3d_0_5, profile_lut, name=f"{camera_name}_profile_lut_05"
    # )

    print_val_lut(hsv_lut, tag=camera_name)
    hsv_lut_applied = apply_lut(block_3d, hsv_lut, name=f"{camera_name}_hsv_1")

    # profile_lut_applied = apply_lut(
    # block_3d_1, profile_lut, name=f"{camera_name}_profile_lut_1"
    # )
    # Visualize data before and after the LUT is applied
