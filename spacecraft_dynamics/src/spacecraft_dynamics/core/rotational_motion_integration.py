import numpy as np
import matplotlib.pyplot as plt

from ..utils import MRP
from ..utils import rigid_body_kinematics as RBK 
from ..utils import initial_conditions as IC 
from ..guidance import Mode
from ..analysis import plots as plots


def TorqueFreeRotationalEquationsOfMotion(state, t, u, args=()):
    """
    Differential equations of motion of a rigid body experiencing torque free rotational motion
    Here, the state is of form [[s1, s2, s3], [w1, w2, w3]]

    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the body
    :param t: time (unused in this equation)
    :param u: control torque (unused in this equation)
    :param args: Tuple containing only the moment of inertia matrix of the body (assumed in principal frame)

    :return state_dot: 2x3 array containing the time derivative of the state 
    """
    I = args[0]
    I1 = I[0,0]
    I2 = I[1,1]
    I3 = I[2,2]

    # TODO: create "from_array" method in MRP class
    sigma = MRP(state[0,0], state[0,1], state[0,2])

    # Convert to array to use with numpy
    sigma = sigma.as_array()
    Bmat = RBK.BmatMRP(sigma) 

    B_omega_BN = np.array([state[1,0], state[1,1], state[1,2]])

    w1 = B_omega_BN[0]
    w2 = B_omega_BN[1]
    w3 = B_omega_BN[2]

    # Calculate MRP rates from omega and sigma
    sigma_dot = 0.25 * np.matmul(Bmat, B_omega_BN)
    # print("sigma_dot: {}".format(sigma_dot.shape))

    omega1_dot = (-(I3-I2)*w2*w3)/I1
    omega2_dot = (-(I1-I3)*w3*w1)/I2
    omega3_dot = (-(I2-I1)*w1*w2)/I3

    omega_dot = np.array([omega1_dot, omega2_dot, omega3_dot])
    
    state_dot = np.array([sigma_dot, omega_dot])
    # print("state_dot: {}".format(state_dot))

    return state_dot

def RotationalEquationsOfMotion(state, t, u, args=()):
    """
    Differential equations of motion of a rigid body experiencing torque free rotational motion
    Here, the state is of form [[s1, s2, s3], [w1, w2, w3]]

    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the body
    :param t: time (unused in this equation)
    :param u: control torque
    :param args: Tuple containing only the moment of inertia matrix of the body (assumed in principal frame)

    :return state_dot: 2x3 array containing the time derivative of the state 
    """
    I = args[0]
    I1 = I[0,0]
    I2 = I[1,1]
    I3 = I[2,2]
    
    B_u1 = u[0]
    B_u2 = u[1]
    B_u3 = u[2]

    # TODO: create "from_array" method in MRP class
    sigma = MRP(state[0,0], state[0,1], state[0,2])

    # Convert to array to use with numpy
    sigma = sigma.as_array()
    Bmat = RBK.BmatMRP(sigma) 

    B_omega_BN = np.array([state[1,0], state[1,1], state[1,2]])

    w1 = B_omega_BN[0]
    w2 = B_omega_BN[1]
    w3 = B_omega_BN[2]

    # Calculate MRP rates from omega and sigma
    sigma_dot = 0.25 * np.matmul(Bmat, B_omega_BN)
    # print("sigma_dot: {}".format(sigma_dot.shape))

    omega1_dot = (-(I3-I2)*w2*w3 + B_u1)/I1
    omega2_dot = (-(I1-I3)*w3*w1 + B_u2)/I2
    omega3_dot = (-(I2-I1)*w1*w2 + B_u3)/I3

    omega_dot = np.array([omega1_dot, omega2_dot, omega3_dot])
    
    state_dot = np.array([sigma_dot, omega_dot])
    # print("state_dot: {}".format(state_dot))

    return state_dot

def ZeroControl(t, state, gains=(None, None) ):
    mode = Mode.INVALID
    sigma_BR = MRP(0,0,0)
    B_omega_BR = np.zeros(3)
    return (np.zeros(3)), mode, sigma_BR, B_omega_BR 

def ConstantControl(t, state, gains=(None, None)):
    mode = Mode.INVALID
    sigma_BR = MRP(0,0,0)
    B_omega_BR = np.zeros(3)
    return np.array([0.01, -0.01, 0.02]), mode, sigma_BR, B_omega_BR 

def RungeKutta(init_state = np.zeros((2,3)),  
                t_max = 100, 
                t_step = 0.1, 
                diff_eq = TorqueFreeRotationalEquationsOfMotion,
                args = (None, ),
                ctrl = ZeroControl,
                ctrl_args = (None, None),
                inertia = None):
    """
    4th order RK4 integrator 
    Note this integrator does slightly more than just integrate equations of motion, it also
    calculates the necessary control torque to apply to the equations of motion

    :param init_state: initial state array of [sigma_BN, omega_BN] at time 0
    :param t_max: End time of integrator
    :param t_step: time step for integration
    :param diff_eq: Function handle for the equations of motion to be integrated, must be of 
            form f(state, t, u, args)
    :param args: Tuple of arguments passed into equations of motion (used for inertia matrix)
    :param ctrl: Function handle for generating the control torque to be applied to the differential
            equation, must be of form g(t, state, gains)
    :param ctrl_args: Tuple of arguments passed into control function (used for gains)

    :return solution_dict: Dictionary mapping variables to lists of the values they had during integration
    :rtype: dict
    """
    
    # Initialize state and time
    state = init_state
    t = 0.0

    # Initial control torque and tracking errors
    # TODO: make this cleaner
    if inertia is not None:
        B_u, mode, sigma_BR, B_omega_BR = ctrl(t, state, ctrl_args, inertia=inertia)
    else:
         B_u, mode, sigma_BR, B_omega_BR = ctrl(t, state, ctrl_args)
    
    # convert sigma_BR to array # TODO make MRP class indexible
    sigma_BR = sigma_BR.as_array()

    # Initialize containers for storing data
    solution_dict = {}
    solution_dict["MRP"] = [state[0]]   # sigma_BN
    solution_dict["omega_B_N"] = [state[1]] # B_omega_BN
    solution_dict["control"] = [B_u]
    solution_dict["mode_value"] = [mode.value]
    solution_dict["time"] = [t]
    solution_dict["sigma_BR"] = [sigma_BR]
    solution_dict["B_omega_BR"] = [B_omega_BR]

    I_b = args[0]

    while t < t_max:



        # Calculate intermediate values
        k1 = t_step*diff_eq(state, t, B_u, args=(I_b,))
        k2 = t_step*diff_eq(state + k1/2, t + t_step/2, B_u, args=(I_b,))
        k3 = t_step*diff_eq(state + k2/2, t + t_step/2, B_u, args=(I_b,))
        k4 = t_step*diff_eq(state + k3, t + t_step, B_u, args=(I_b,))

        # Update state
        state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

        # Check MRP magnitude and covert to shadow set if necessary
        sigma = MRP(state[0,0], state[0,1], state[0,2])
        if sigma.norm() > 1.0:
            sigma = sigma.convert_to_shadow_set()
            
            # Update state with new MRP set
            state[0] = sigma.as_array()

        # Increment the time
        t = t + t_step

        # Calculate the desired control torque for the next step
        # TODO: make this cleaner
        if inertia is not None:
            B_u, mode, sigma_BR, B_omega_BR = ctrl(t, state, ctrl_args, inertia=inertia)
        else:
            B_u, mode, sigma_BR, B_omega_BR = ctrl(t, state, ctrl_args)
        sigma_BR = sigma_BR.as_array()
        
        # Save states and controls
        solution_dict["MRP"].append(state[0])
        solution_dict["omega_B_N"].append(state[1])
        solution_dict["control"].append(B_u)
        solution_dict["mode_value"].append(mode.value)
        solution_dict["time"].append(t)
        solution_dict["sigma_BR"].append(sigma_BR)
        solution_dict["B_omega_BR"].append(B_omega_BR)
        # solution_state.append(state)
        # solution_controls.append(u)

    # Convert lists to arrays so they're easier to work with later
    for key in solution_dict.keys():
        solution_dict[key] = np.array(solution_dict[key])

    return solution_dict

def Homework4Problem2():
    """
    Homework 4 Problem 2
    """
    
    # Define the pure rotation angular velocity vectors [rad/s]
    # Each of these will be analyzed independently
    w_B_N_1 = np.array([1.,0.,0.]) # Rotation about b1 
    w_B_N_2 = np.array([0.,1.,0.]) # Rotation about b2
    w_B_N_3 = np.array([0.,0.,1.]) # Rotation about b3

    # Define the perturbations to be applied to the non-primary rotation axes [rad/s]
    w_perturb = 0.1   

    # Define initial MRP
    sigma0 = MRP(0.,0.,0.)
    sigma0 = sigma0.as_array()


    ### First Case: Pure spin about b1
    # Define the initial state
    # x0_case_1 = [sigma0[0], sigma0[1], sigma0[2], sigma_dot0_case_1[0], sigma_dot0_case_1[1], sigma_dot0_case_1[2], w_B_N_1[0], w_B_N_1[1], w_B_N_1[2], w_B_N_dot0[0], w_B_N_dot0[1], w_B_N_dot0[2]]
    x0_case_1 = np.array([sigma0, w_B_N_1])
    print("x0_case_1: \n{}".format(x0_case_1))

    # Simulate case 1
    solution_state_1 = RungeKutta(init_state=x0_case_1)

    # Extract the log data
    mrps_case_1 = solution_state_1["MRP"]
    omegas_case_1 = solution_state_1["omega_B_N"]
    t_vals = solution_state_1["time"]

    # Plot case 1
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Pure Spin About $\hat b_1 $')
    axs[0].plot(t_vals, mrps_case_1[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_1[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_1[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_1[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_1[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_1[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    # plt.show()


    ### Second Case: Pure spin about b2
    # Define the initial state
    x0_case_2 = np.array([sigma0, w_B_N_2])
    print("x0_case_2: \n{}".format(x0_case_2))

    # Simulate case 2
    solution_state_2 = RungeKutta(init_state=x0_case_2)

    # Extract the log data
    mrps_case_2 = solution_state_2["MRP"]
    omegas_case_2 = solution_state_2["omega_B_N"]
    t_vals = solution_state_2["time"]

    # Plot case 2
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Pure Spin About $\hat b_2 $')
    axs[0].plot(t_vals, mrps_case_2[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_2[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_2[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_2[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_2[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_2[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    # plt.show()

    ### Third Case: Pure spin about b3
    # Define the initial state
    x0_case_3 = np.array([sigma0, w_B_N_3])
    print("x0_case_3: \n{}".format(x0_case_3))

    # Simulate case 3
    solution_state_3 = RungeKutta(init_state=x0_case_3)

    # Extract the log data
    mrps_case_3 = solution_state_3["MRP"]
    omegas_case_3 = solution_state_3["omega_B_N"]
    t_vals = solution_state_3["time"]

    # Plot case 3
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Pure Spin About $\hat b_3 $')
    axs[0].plot(t_vals, mrps_case_3[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_3[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_3[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_3[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_3[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_3[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    # plt.show()

    ### Fourth Case: Spin about b1 with small perturbations in b3
    # Define the initial state
    w_B_N_4_pert = w_B_N_1 + np.array([0, 0, w_perturb])
    x0_case_4 = np.array([sigma0, w_B_N_4_pert])
    print("x0_case_4: {}".format(x0_case_4))

    # Simulate case 4
    solution_state_4 = RungeKutta(init_state=x0_case_4)

    # Extract the log data
    mrps_case_4 = solution_state_4["MRP"]
    omegas_case_4 = solution_state_4["omega_B_N"]
    t_vals = solution_state_4["time"]

    # Plot case 4
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Spin About $\hat b_1 $ with Perturbations in $\omega_3 \hat b_3$')
    axs[0].plot(t_vals, mrps_case_4[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_4[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_4[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_4[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_4[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_4[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    # plt.show()

    ### Fifth Case: Spin about b2 with small perturbations in b1
    # Define the initial state
    w_B_N_5_pert = w_B_N_2 + np.array([w_perturb, 0, 0])
    x0_case_5 = np.array([sigma0, w_B_N_5_pert])
    print("x0_case_5: {}".format(x0_case_5))

    # Simulate case 5
    solution_state_5 = RungeKutta(init_state=x0_case_5)

    # Extract the log data
    mrps_case_5 = solution_state_5["MRP"]
    omegas_case_5 = solution_state_5["omega_B_N"]
    t_vals = solution_state_5["time"]

    # Plot case 5
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Spin About $\hat b_2 $ with Perturbations in $\omega_1 \hat b_1$')
    axs[0].plot(t_vals, mrps_case_5[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_5[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_5[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_5[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_5[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_5[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    # plt.show()

    ### Sixth Case: Spin about b3 with small perturbations in b2
    # Define the initial state
    w_B_N_6_pert = w_B_N_3 + np.array([0, w_perturb, 0])
    x0_case_6 = np.array([sigma0, w_B_N_6_pert])
    print("x0_case_6: {}".format(x0_case_6))

    # Simulate case 6
    solution_state_6 = RungeKutta(init_state=x0_case_6)

    # Extract the log data
    mrps_case_6 = solution_state_6["MRP"]
    omegas_case_6 = solution_state_6["omega_B_N"]
    t_vals = solution_state_6["time"]
    # Plot case 6
    fig,axs = plt.subplots(2)
    fig.suptitle('Angular Velocity and MRPs During Spin About $\hat b_3 $ with Perturbations in $\omega_2 \hat b_2$')
    axs[0].plot(t_vals, mrps_case_6[:, 0], 'b', label='$\sigma_1$')
    axs[0].plot(t_vals, mrps_case_6[:, 1], 'g', label='$\sigma_2$')
    axs[0].plot(t_vals, mrps_case_6[:, 2], 'r', label='$\sigma_3$')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='t', ylabel='$\sigma$')
    axs[0].grid()

    axs[1].plot(t_vals, omegas_case_6[:, 0], 'b', label='$\omega_1$')
    axs[1].plot(t_vals, omegas_case_6[:, 1], 'g', label='$\omega_2$')
    axs[1].plot(t_vals, omegas_case_6[:, 2], 'r', label='$\omega_3$')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='t', ylabel='$\omega$ [rad/s]')
    axs[1].grid()
    
    plt.show()
    ##### End Homework 4 Problem 2

def CapstoneTask7():
    """
    Function for task 7 of the captsone project
    """

    # Define the initial state
    sigma_0 = IC.LMO_SIGMA_BN_0.as_array()
    B_omega_BN_0 = IC.LMO_B_OMEGA_BN_0
    state_0 = np.array([sigma_0, B_omega_BN_0])
    # u_0 = np.zeros(3)

    # Simulate first 500s
    solution_state = RungeKutta(init_state=state_0, t_max=500, t_step=1.0, diff_eq=RotationalEquationsOfMotion, args=(IC.LMO_INERTIA, ), ctrl=ZeroControl)

    # Extract the log data
    sigma_BN_list = solution_state["MRP"]
    B_omega_BN_list = solution_state["omega_B_N"]
    t_list = solution_state["time"]

    # At 500s
    sigma_BN = sigma_BN_list[-1]
    B_omega_BN = B_omega_BN_list[-1]
    t = t_list[-1]

    # Calculate angular momentum in body frame components at 500s [kg*m^2/s]
    B_ang_mom = np.matmul(IC.LMO_INERTIA, B_omega_BN)

    # Calculate kinetic energy at 500s [J]
    kinetic_energy = 1/2*np.matmul(B_omega_BN, np.matmul(IC.LMO_INERTIA, B_omega_BN))

    # Calculate the angular momentum in inertial frame components at 500s [kg*m^2/s]
    dcm_B_N = RBK.MRP2C(sigma_BN)
    dcm_N_B = np.transpose(dcm_B_N)

    N_ang_mom = np.matmul(dcm_N_B, B_ang_mom)

    # Print values
    print("{Torque free values}")
    print("> Time: {} \n> sigma_BN: {} \n> B_omega_BN: {} \n> B_ang_mom: {} \n> kinetic_energy: {} \n> N_ang_mom: {}".format(t, sigma_BN, B_omega_BN, B_ang_mom, kinetic_energy, N_ang_mom))   
    
    # Plot values
    plots.PlotMrpAndOmegaComponents(sigma_BN_list, B_omega_BN_list, t_list, title='Evolution of $\sigma_{B/N}$ and $\omega_{B/N}$ With 0 Control')

    # Define a constant control vector and simulate for 100s
    # u_0 = np.array([0.01, -0.01, 0.02]) # [Nm]

    solution_state_torque = RungeKutta(init_state=state_0, t_max=500, t_step=1.0, diff_eq=RotationalEquationsOfMotion, args=(IC.LMO_INERTIA, ), ctrl=ConstantControl)

    # Extract the log data
    sigma_BN_list = solution_state_torque["MRP"]
    B_omega_BN_list = solution_state_torque["omega_B_N"]
    t_list = solution_state_torque["time"]

    # At 100s
    sigma_BN = sigma_BN_list[-1]
    B_omega_BN = B_omega_BN_list[-1]
    t = t_list[-1]

    # Print values
    print("{Constant Torque Values}")
    print("> Time: {} \n> sigma_BN: {}".format(t, sigma_BN))

    # Plot values
    plots.PlotMrpAndOmegaComponents(sigma_BN_list, B_omega_BN_list, t_list, title='Evolution of $\sigma_{B/N}$ and $\omega_{B/N}$ With Constant Control')


if __name__ == "__main__":
    
    # Homework4Problem2()
    CapstoneTask7()
