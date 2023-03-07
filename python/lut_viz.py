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


def interp_hsv_lut(hsv_coord: tuple, lut: np.ndarray):
    hue_p = np.linspace(0, 90, 90)
    sat_p = np.linspace(0, 30, 30)
    points = (hue_p, sat_p)
    hsv_vals = [hsv_coord[0], hsv_coord[1]]
    print("HSV VALS:", hsv_vals[0], hsv_vals[1])
    outInterpolate = scipy.interpolate.interpn(
        points=points, values=lut, xi=hsv_vals
    )

    print("INTERPOLATED Correction:", outInterpolate[:, :])
    print("INTERPOLATED Correction:", outInterpolate[:, :, 0][0])
    print("INTERPOLATED Correction:", outInterpolate[:, :, 1][0])
    print("INTERPOLATED Correction:", outInterpolate[:, :, 2][0])

    new_h = hsv_coord[0] + outInterpolate[:, :, 0][0]
    new_s = hsv_coord[1] * outInterpolate[:, :, 1][0]
    new_v = hsv_coord[2] * outInterpolate[:, :, 2][0]

    return (new_h, new_s, new_v)


def rawpaths_from_dir(dir_path, rawtype=".raw"):
    # NOTE: for cyc7 no bldc motor captures
    # for path, subdirs, files in os.walk(dir_path):
    # for file in files:
    # print(file)
    # if '.raw' in file:
    # paths.append(os.path.join(path, file))
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
    V, H, S = grid
    print("H shape:", H.shape)
    print("S shape:", S.shape)
    print("V shape:", V.shape)
    print("LUT SHAPE:", lut.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Corrected HSV values")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    print(grid.shape)
    _, i_max, j_max, k_max = grid.shape
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                h, s, v = H[i, j, k], S[i, j, k], V[i, j, k]
                # h, s, v = 1.05, 1, 1
                print(f"H: {h}, S: {s}, V: {v}")
                hsv_coordinate = np.asarray(
                    [[[h]], [[s]], [[v]]], dtype=np.float32
                )
                hsv_coordinate = np.reshape(hsv_coordinate, (1, 1, 3))
                print("HSV COORDINATE", hsv_coordinate)
                corrected = performInterpolation(hsv_coordinate, lut)[0, 0]
                corrected[2] = np.clip(corrected[2], 0, 1)
                # corrected[1] = np.clip(corrected[1], 0, 1)
                # corrected[2] = np.clip(corrected[2], 0, 1)
                # corrected = np.clip(corrected, 0, 1)
                # print(f"HSV corrected: {corrected}")
                # ax.arrow3D(
                # h,
                # s,
                # v,
                # int(corrected[0]),
                # int(corrected[1]),
                # int(corrected[2]),
                # mutation_scale=1,
                # ec="green",
                # )
                # ax.quiver(
                # h,
                # s,
                # v,
                # corrected[0],
                # corrected[1],
                # corrected[2],
                # )
                ax.quiver(
                    h,
                    s,
                    v,
                    corrected[0],
                    corrected[1],
                    corrected[2],
                )
                # break
            # break
        # break

    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("V")

    print("saving figure")
    plt.savefig(f"./apply_lut_{idx}.png")


raw_dir = "/shared/data/autoexp-stack-dngs/"

rawpaths = rawpaths_from_dir(raw_dir, ".dng")

for idx, path in enumerate(rawpaths):
    metadata = get_metadata(path)

    # HSV 3D LUT
    hsv_lut = metadata["hsv_lut"]
    lut_3d = metadata["profile_lut"]

    print("LUT shape:", hsv_lut.shape)
    # print_lut(hsv_lut)
    block_3d = np.mgrid[0:1:3j, 0:360:3j, 0:1:3j]
    print("Block Shape:", block_3d.shape)
    plot_3dhsv(block_3d, "Dummy HSV Vals")

    apply_lut(block_3d, hsv_lut, idx)
    apply_lut(block_3d, lut_3d, "3dlut")
    break
    # Generate the image which covers all hsv values to see the transformation
    # 1 channel is hue, 1 channel is saturation, another is value
    # Want 3 channel image, where each channel is a 2D plane between 0,1
    # H = np.mgrid[0:90:100j, 0:90:100j][0]
    # S = np.mgrid[0:30:100j, 0:30:100j][1]
    # V = np.mgrid[0:1:100j, 0:1:100j][0]
    # print(H.shape)
    # print(S.shape)
    # print(V.shape)
    # dummy_hsv = np.dstack([H, S, V])
    # print("Dummy hsv shape:", dummy_hsv.shape)
    # imageio.imsave("dummyhsv.png", dummy_hsv)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.set_xlabel("H")
    # ax.set_ylabel("S")
    # ax.set_zlabel("V")
    # ax.scatter(H, S, V)
    # print("Saving figure")
    # plt.savefig("./mgrid_testing.png")

    # interpolated = performInterpolation(dummy_hsv, hsv_lut)
    # print("Interpolated shape", interpolated.shape)
    # X = interpolated[:, :, 0]
    # Y = interpolated[:, :, 1]
    # Z = interpolated[:, :, 2]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.set_xlabel("H")
    # ax.set_ylabel("S")
    # ax.set_zlabel("V")
    # ax.scatter(Y, Z, X)
    # print("Saving figure")
    # plt.savefig("./lut_testing.png")
    # break

    # print("INTERPOLATED SHAPE", interpolated.shape)

    # plot_3DLUT(dummy_hsv, "hsv")
    # plot_3DLUT(interpolated, "lut conversion")

    # print(profile_lut)
