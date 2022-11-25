#!/usr/bin/env python3
"""
 !  Alaa Akkoush (Fritz Haber Institute)
 !  Minor tweaks by Mariana Rossi (2022)
 !  HISTORY
 !  November 2022
   python3 local.py --name C6H6 -s srun -r path-of-FHIaims-binary -z 3 -m 11 -f 0.0025 -d 0.5 0.5 -n 8 8 -t -0.000030 -1.696604 -4.614046 --plot
"""
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
from ase import Atoms
from ase.io import read


def construct_grid(gridinfo):
    """
    Returns a grid matrix of shape (ngrid, 3)
    """
    orgx, orgy, orgz = gridinfo["org"]
    nx, ny, nz = gridinfo["nstep"]
    stepx, stepy, stepz = gridinfo["stepsize"]
    x = np.linspace(orgx - (nx - 1) * stepx, orgx + (nx - 1) * stepx, nx)
    y = np.linspace(orgy - (ny - 1) * stepy, orgy + (ny - 1) * stepy, ny)
    z = np.linspace(orgz - (nz - 1) * stepz, orgz + (nz - 1) * stepz, nz)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    gx = gx.flatten()
    gy = gy.flatten()
    gz = gz.flatten()
    return np.stack((gx, gy, gz)).transpose()


def split_line(lines):
    """Split input line"""
    line_array = np.array(lines.strip().split(" "))
    line_vals = line_array[line_array != ""]
    return line_vals


def moveCOM(height, col, gridinfo):
    scanxyz = construct_grid(gridinfo)
    molc = read("geometry.in", format="aims")
    n_atoms = len((molc.numbers))
    COM = molc.get_center_of_mass(scaled=False)
    V = Atoms(positions=[COM])
    tp = [scanxyz[col][0], scanxyz[col][1], scanxyz[col][2]]
    # Moving the molecule center of mass to the tip's
    molc.translate(tp - V.positions)
    COM = molc.get_center_of_mass(scaled=False)
    molc.write("geometry.in", format="aims")
    return n_atoms


def fraction(f, norm_mode):  # norm_mode is here the mode in Cartesian coordinates
    frac = f / (np.linalg.norm(norm_mode))
    return frac


# Reading the normal modes from get_vibrations.py
def car_modes(n_atoms, name, num):
    num_line = 1
    norm_mode = np.array([])
    filename = "car_eig_vec." + str(name) + ".dat"
    if os.path.exists(filename):
        temp = open(filename)
        lines = temp.readlines()
        for line in lines:
            if num_line == num:
                norm_mode = np.float64(line.split()[0:])
            if num > (3 * n_atoms):  # 3*n
                print("The mode you are requesting does not exist :)")
            num_line = num_line + 1
    else:
        print("Normal modes not found, run get_vibrations.py (run mode = 1)")
        sys.exit(1)
    norm_mode = norm_mode.reshape(n_atoms, 3)
    norm_mode = np.array(norm_mode)
    return norm_mode


def read_geo(filename):
    """Function to transfer coordinates from atom_frac to atom"""
    fdata = []
    element = []
    with open(filename) as f:
        for line in f:
            t = line.split()
            if len(t) == 0:
                continue
            if t[0] == "#":
                continue
            elif t[0] == "atom":
                fdata += [(float(t[1]), float(t[2]), float(t[3]))]
                element += [(str(t[4]))]
            else:
                continue
    fdata = np.array(fdata)
    element = np.array(element)
    return fdata, element


def shift_geo(f, col, scanxyz, name, direction, num, n_atoms):
    norm_mode = car_modes(n_atoms, name, num)
    frac = fraction(f, norm_mode)
    fdata, element = read_geo("geometry.in")
    pos = fdata + frac * norm_mode
    neg = fdata - frac * norm_mode
    folder = (
        name
        + "_disp_"
        + str(direction)
        + "_{}_{}_{}".format(scanxyz[col][0], scanxyz[col][1], scanxyz[col][2])
    )
    # print("Geometry Files copied successfully.")
    if not os.path.exists(folder):
        os.mkdir(folder)
    new_geo = open(folder + "/geometry.in", "w")
    for i in range(0, len(fdata)):
        if direction == "pos":
            new_geo.write(
                "atom" + ((" %.8f" * 3) % tuple(pos[i, :])) + " " + element[i] + "\n"
            )
        else:
            new_geo.write(
                "atom" + ((" %.8f" * 3) % tuple(neg[i, :])) + " " + element[i] + "\n"
            )


def precontrol(filename, name, scanxyz, col, direction, AIMS_CALL, run_aims):
    """Function to copy and edit control.in"""
    aimsout = "aims.out"
    folder = (
        name
        + "_disp_"
        + str(direction)
        + "_{}_{}_{}".format(scanxyz[col][0], scanxyz[col][1], scanxyz[col][2])
    )
    f = open(filename, "r")  # read control.in template
    template_control = f.read()
    f.close
    # print("Cube Files copied successfully.")
    if not os.path.exists(folder):
        os.mkdir(folder)
    shutil.copy("tipA_05_vh_ft_0049_3221meV_x1000.cube", folder)
    shutil.copy("zeros.cube", folder)
    new_control = open(folder + "/control.in", "w")
    new_control.write(
        template_control
        + "DFPT local_polarizability nearfield \n "
        + "DFPT local_parameters numerical zeros.cube zeros.cube tipA_05_vh_ft_0049_3221meV_x1000.cube  \n"
    )
    new_control.close()
    os.chdir(folder)
    # Change directoy
    if run_aims:
        os.system(
            AIMS_CALL + " > " + aimsout
        )  # Run aims and pipe the output into a file named 'filename'
    os.chdir("..")


def postpro(direction, name, scanxyz, col, AIMS_CALL, run_aims):
    """Function to read outputs"""
    alpha = np.zeros(6)
    folder = (
        name
        + "_disp_"
        + str(direction)
        + "_{}_{}_{}".format(scanxyz[col][0], scanxyz[col][1], scanxyz[col][2])
    )
    aimsout = "aims.out"
    #      # checking existence of aims.out
    if os.path.exists(folder + "/" + aimsout):
        data = open(folder + "/" + aimsout)
        out = data.readlines()
        if "Have a nice day." in out[-2]:
            print(
                "Aims calculation is complete for direction  "
                + str(direction)
                + "_{}_{}_{}".format(scanxyz[col][0], scanxyz[col][1], scanxyz[col][2])
                + "\n"
            )
        else:
            print(
                "Aims calculation isnt complete for direction "
                + str(direction)
                + "_{}_{}_{}".format(scanxyz[col][0], scanxyz[col][1], scanxyz[col][2])
                + "\n"
            )
        # os.chdir(folder)
        # if run_aims:
        #     os.system(
        #         AIMS_CALL + " > " + aimsout
        #     )  # Run aims and pipe the output into a file named 'filename'
        # os.chdir("..")
        #        sys.exit(1)
        for line in out:
            if line.rfind("Polarizability") != -1:
                alpha = np.float64(split_line(line)[-6:])  # alpha_zz
    else:
        os.chdir(folder)
        if run_aims:
            os.system(
                AIMS_CALL + " > " + aimsout
            )  # Run aims and pipe the output into a file named 'filename'
        os.chdir("..")
    return alpha


def main():

    parser = ArgumentParser(description="TERS model calculation with FHI-aims")
    parser.add_argument(
        "--name",
        dest="name",
        action="store",
        help="A string identifying the file where the eigenvectors will be read from. Works with usual outputs of get_vibrations.py. E.g. use C6H6 if the file is car_eig_vec.C6H6.dat ",
        required=True,
    )
    parser.add_argument(
        "-i", "--info", action="store_true", help="Set up/ Calculate vibrations & quit"
    )
    parser.add_argument(
        "-n",
        "--step",
        action="store",
        type=int,
        nargs=2,
        dest="step",
        help="is the number of steps of the 2D grid in which to scan the TERS image (number of steps in each direction)",
    )
    parser.add_argument(
        "-d",
        "--size",
        action="store",
        type=float,
        nargs=2,
        dest="size",
        help="the step size of the 2D grid. The COM of the system will be aligned to the tip apex and the grid is defined around it in each direction as np.linspace(origin-(n-1)*step, origin+(n-1)*step, n)",
    )
    parser.add_argument(
        "-t",
        "--tip",
        action="store",
        type=float,
        nargs=3,
        dest="tip",
        help="the tip apex position that is given in the position.dat file in angstrom (Example: -0.000030 -1.696604 -4.6140)",
    )
    parser.add_argument(
        "-r", "--run", action="store", help="path of FHI-aims binary", default=""
    )
    parser.add_argument(
        "-s",
        "--prefix",
        action="store",
        help="Call prefix for binary e.g. 'mpirun -np 12 '",
        default="",
    )
    parser.add_argument(
        "-m", "--mode", action="store", type=int, help="Mode number", nargs=1
    )
    parser.add_argument(
        "-z",
        "--height",
        action="store",
        type=float,
        help="Distance from tip apex (please check the coordinates of the apex)",
        default=1.0,
    )
    parser.add_argument(
        "-f",
        "--displacement",
        action="store",
        type=float,
        help="The displacement of mode coordinate in Angstroms",
        default=0.002,
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Generate TERS image with pyplot"
    )
    options = parser.parse_args()
    if options.info:
        print(__doc__)
        sys.exit(0)

    AIMS_CALL = options.prefix + " " + options.run
    name = options.name
    num = options.mode
    num = num[0]
    f = options.displacement
    z = options.height
    n = options.step
    d = options.size
    tip = options.tip
    # Construct scanning grid. Store coordinates in scanxyz
    gridinfo = {}
    gridinfo["org"] = [tip[0], tip[1], tip[2] - z]
    gridinfo["stepsize"] = [d[0], d[1], 0]
    gridinfo["nstep"] = [n[0], n[1], 1]
    scanxyz = construct_grid(gridinfo)
    if not options.mode:
        parser.error("Specify the vibrational mode you want by typing -N #")
    run_aims = False
    if options.run != "":
        run_aims = True
    newline_ir = "\n"
    irname = name + ".data"
    print("The calculation will start ...")
    molc = read("geometry.in", format="aims")
    n_atoms = len((molc.numbers))
    norm_mode = car_modes(n_atoms, name, num)
    frac = fraction(f, norm_mode)
    print("fraction of shifting of the normal mode", frac)
    # Moving the molecule below the tip
    molc = read("geometry.in", format="aims")
    COM = molc.get_center_of_mass(scaled=False)
    V = Atoms(positions=[COM])
    # Moving the molecule center of mass to the tip's
    molc.translate(tip - V.positions)
    molc.write("geometry_00.in", format="aims")
    for col in range(n[0] * n[1]):
        moveCOM(z, col, gridinfo)
        shift_geo(f, col, scanxyz, name, "neg", num, n_atoms)
        shift_geo(f, col, scanxyz, name, "pos", num, n_atoms)

        precontrol("control.in", name, scanxyz, col, "pos", AIMS_CALL, run_aims)
        precontrol("control.in", name, scanxyz, col, "neg", AIMS_CALL, run_aims)
    for col in range(n[0] * n[1]):
        alpha_pos = postpro("pos", name, scanxyz, col, AIMS_CALL, run_aims)
        alpha_neg = postpro("neg", name, scanxyz, col, AIMS_CALL, run_aims)
        # Intensity
        alphas = alpha_pos - alpha_neg
        alphas = alphas / (2 * frac)
        alphaszz = alphas[2]
        # Intensity
        # polarizability tensor derivative
        # alphasxx = alphas[0]
        # alphasyy = alphas[1]
        # alphasxy = alphas[3]
        # alphasxz = alphas[4]
        # alphasyz = alphas[5]
        # alpha = (alphasxx + alphasyy + alphaszz) * (1.0 / 3)
        # beta = (
        #     (alphasxx - alphasyy) ** 2
        #     + (alphasxx - alphaszz) ** 2
        #     + (alphasyy - alphaszz) ** 2
        #     + 6 * ((alphasxy) ** 2 + (alphasxz) ** 2 + (alphasyz) ** 2)
        # )
        # man Scattering Intensity:
        I = ((alphaszz) ** 2) * 0.02195865620442408  # bohr^6/ang^2 to ang^4
        # Saving Intensity
        newline_ir = newline_ir + "{0:25.8f} {1:25.8f}{2:25.8f}\n".format(
            scanxyz[col][0], scanxyz[col][1], I
        )

        ir = open(irname, "w")
        ir.writelines("#x   y   Raman_zz")
        ir.writelines(newline_ir)
        ir.close()
    if options.plot:
        import matplotlib.pyplot as plt
        from matplotlib import rc, rcParams

        rc("text", usetex=True)
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20
        plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # plt.rcParams['font.family'] = 'serif'
        # scheiner.stephan@gmail.complt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        params = {
            "axes.labelsize": 22,
            "font.size": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "mathtext.fontset": "stix",
            "axes.linewidth": 2.0,
        }
        rcParams.update(params)
        gridx, gridy, gridz = np.loadtxt(name + ".data", unpack=True)
        gridx = gridx - tip[0]
        gridy = gridy - tip[1]
        N = int(len(gridz) ** 0.5)
        Z = gridz.reshape(N, N)
        Z = Z.T
        fg = plt.imshow(
            Z,
            origin="lower",
            extent=(np.amin(gridx), np.amax(gridx), np.amin(gridy), np.amax(gridy)),
            cmap=plt.cm.jet,
            aspect="equal",
            interpolation="spline36",
        )
        plt.title("TERS Image")
        plt.xlabel("Distance, \AA")
        plt.ylabel("Distance, \AA")
        cbar = plt.colorbar(fg, ticks=[Z.min(), Z.max()])
        cbar.set_label("Intensity", rotation=270, labelpad=20, y=0.2)
        cbar.ax.set_yticklabels(["low", "high"])  # vertically oriented colorbar
        plt.tight_layout()
        plt.savefig(name + ".png", transparent=True, dpi=400)
        plt.show()


if __name__ == "__main__":
    main()
