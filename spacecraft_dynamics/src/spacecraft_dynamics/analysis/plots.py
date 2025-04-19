import matplotlib.pyplot as plt
import numpy as np

from ..guidance import Mode

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


def PlotWheelSpeeds(wheel_speeds, time_list, title=''):
    """
    Plot wheel speeds for all VSCMGs over time
    
    Args:
        wheel_speeds: Dictionary with VSCMG index as key and list of wheel speeds as value
        time_list: List of time points
        title: Plot title (optional)
    """
    plt.figure(figsize=(10, 6))
    
    for idx, speeds in wheel_speeds.items():
        plt.plot(time_list, speeds, label=f'VSCMG {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Wheel Speed [rad/s]')
    plt.grid(True)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def PlotGimbalAngles(gimbal_angles, time_list, title=''):
    """
    Plot gimbal angles for all VSCMGs over time
    
    Args:
        gimbal_angles: Dictionary with VSCMG index as key and list of gimbal angles as value
        time_list: List of time points
        title: Plot title (optional)
    """
    plt.figure(figsize=(10, 6))
    
    for idx, angles in gimbal_angles.items():
        # Convert angles to degrees for better readability
        angles_deg = np.array(angles) * 180 / np.pi
        plt.plot(time_list, angles_deg, label=f'VSCMG {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Gimbal Angle [deg]')
    plt.grid(True)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def PlotGimbalTorques(gimbal_torque_dict: dict, t_list: list, title: str = None):
    """
    Plot gimbal torques for all VSCMGs
    
    Args:
        gimbal_torque_dict (dict): Dictionary containing gimbal torques for each VSCMG
        t_list (list): List of time points
        title (str, optional): Plot title. Defaults to None.
    """
    plt.figure()
    for idx, torques in gimbal_torque_dict.items():
        plt.plot(t_list, torques, label=f'VSCMG {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Gimbal Torque [Nm]')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def PlotWheelTorques(wheel_torque_dict: dict, t_list: list, title: str = None):
    """
    Plot wheel torques for all VSCMGs
    
    Args:
        wheel_torque_dict (dict): Dictionary containing wheel torques for each VSCMG
        t_list (list): List of time points
        title (str, optional): Plot title. Defaults to None.
    """
    plt.figure()
    for idx, torques in wheel_torque_dict.items():
        plt.plot(t_list, torques, label=f'VSCMG {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Wheel Torque [Nm]')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()