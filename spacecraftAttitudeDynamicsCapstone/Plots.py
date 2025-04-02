
import matplotlib.pyplot as plt
import numpy as np
from PointingControl import Mode

plt.rcParams.update({'font.size': 20})


def PlotMode(mode, t, title=None):
    
    modes = [str(s.name) for s in Mode]
    y_ticks = np.linspace(-1, 2, 4)

    fig,axs = plt.subplots()
    fig.suptitle(title)


    axs.plot(t, mode, 'b')
    axs.set_yticks(y_ticks, labels=modes)
    axs.set(xlabel='t [s]', ylabel='Mode')
    axs.grid()
    plt.show()

def PlotMrpAndOmegaComponents(mrps, omegas, t, title=None):

    fig,axs = plt.subplots(2)
    fig.suptitle(title)
    axs[0].plot(t, mrps[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t, mrps[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t, mrps[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t, omegas[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t, omegas[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t, omegas[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t [s]', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    plt.show()

def PlotMrpAndOmegaNorms(mrps, omegas, t, title=None):

    mrp_norms  = []
    for mrp in mrps:
        norm = np.linalg.norm(mrp)
        mrp_norms.append(norm)

    omega_norms  = []
    for omega in omegas:
        norm = np.linalg.norm(omega)
        omega_norms.append(norm)

    fig,axs = plt.subplots(2)
    fig.suptitle(title)
    axs[0].plot(t, mrp_norms, 'b')
    axs[0].set(ylabel='$|\sigma_{B/R}|$')
    axs[0].grid()

    axs[1].plot(t, omega_norms, 'b', label='$|\omega_{B/R}|$')
    axs[1].set(xlabel='t [s]', ylabel='$|\omega_{B/R}|$ [rad/s]')
    axs[1].grid()
    plt.show()


def PlotStatesAndReferences(sigma_BN, sigma_RN, omega_BN, omega_RN, t, title=None):
    fig,axs = plt.subplots(3,2)
    fig.suptitle(title)

    # Plot attitude and components and their references
    axs[1,0].plot(t, sigma_BN[:, 0], 'b', label='$\sigma_{BN}$')
    axs[1,0].plot(t, sigma_RN[:, 0], 'b--', label='$\sigma_{RN}$')
    axs[1,0].legend(loc='best')
    axs[1,0].set(ylabel='$\sigma_1$')
    axs[1,0].grid()
    
    axs[2,0].plot(t, sigma_BN[:, 1], 'b', label='$\sigma_{BN}$')
    axs[2,0].plot(t, sigma_RN[:, 1], 'b--', label='$\sigma_{RN}$')
    axs[2,0].legend(loc='best')
    axs[2,0].set(ylabel='$\sigma_3$')
    axs[2,0].grid()
    
    axs[3,0].plot(t, sigma_BN[:, 2], 'b', label='$\sigma_{BN}$')
    axs[3,0].plot(t, sigma_RN[:, 2], 'b--', label='$\sigma_{RN}$')
    axs[3,0].legend(loc='best')
    axs[3,0].set(xlabel='t [s]',ylabel='$\sigma_3$')
    axs[3,0].grid()

    # Plot ang vel components and their references
    axs[1,1].plot(t, omega_BN[:, 0], 'b', label='$\omega_{BN}$')
    axs[1,1].plot(t, omega_RN[:, 0], 'b--', label='$\omega_{RN}$')
    axs[1,1].legend(loc='best')
    axs[1,1].set(ylabel='$\omega_1$')
    axs[1,1].grid()
    
    axs[2,1].plot(t, omega_BN[:, 1], 'b', label='$\omega_{BN}$')
    axs[2,1].plot(t, omega_RN[:, 1], 'b--', label='$\omega_{RN}$')
    axs[2,1].legend(loc='best')
    axs[2,1].set(ylabel='$\omega_3$')
    axs[2,1].grid()
    
    axs[3,1].plot(t, omega_BN[:, 2], 'b', label='$\omega_{BN}$')
    axs[3,1].plot(t, omega_RN[:, 2], 'b--', label='$\omega_{RN}$')
    axs[3,1].legend(loc='best')
    axs[3,1].set(xlabel='t [s]',ylabel='$\omega_3$')
    axs[3,1].grid()
    plt.show()

def PlotTorqueComponents(u, t, title=None):

    fig,axs = plt.subplots()
    fig.suptitle(title)

    axs.plot(t, u[:, 0], 'b', label='$u_1$')
    axs.plot(t, u[:, 1], 'g', label='$u_2$')
    axs.plot(t, u[:, 2], 'r', label='$u_3$')
    axs.legend(loc='best')
    axs.set(xlabel='t [s]', ylabel='Torque $u$ [Nm]')
    axs.grid()

    plt.show()
