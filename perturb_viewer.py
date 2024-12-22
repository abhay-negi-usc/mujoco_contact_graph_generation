# from dm_control import mjcf
# from dm_control.mujoco.wrapper.mjbindings import mjlib
# from dm_control.utils.inverse_kinematics import qpos_from_site_pose

# import numpy as np
# from scipy.spatial.transform import Rotation
# import mujoco 
# import time 

# model = mujoco.MjModel.from_xml_path("./env/cross_env.xml") 
# data = mujoco.MjData(model) 

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # Close the viewer automatically after 30 wall-seconds.
#     start = time.time()
#     x0 = np.random.uniform(-3, +3) * (1e-3) 
#     y0 = np.random.uniform(-3, +3) * (1e-3) 
#     z0 = np.random.uniform(+25, +30) * (1e-3) 
#     a0 = np.random.uniform(-5.0, +5.0) * (np.pi/180)  
#     b0 = np.random.uniform(-5.0, +5.0) * (np.pi/180)  
#     c0 = np.random.uniform(-5.0, +5.0) * (np.pi/180) 

#     mujoco.mj_resetData(model, data)
#     data.qpos = np.array([x0, y0, z0, a0, b0, c0]) 
#     data.qvel = np.zeros(6) 
#     mujoco.mj_forward(model, data)
#     z_step = 1e-6
#     i = 0 

#     while viewer.is_running() and time.time() - start < 3000:
#         step_start = time.time()

#         # controller update 
#         peg_x_axis = data.xmat[2].reshape(3,3)[:,0] 
#         peg_y_axis = data.xmat[2].reshape(3,3)[:,1] 
#         peg_z_axis = data.xmat[2].reshape(3,3)[:,2]  
#         angle_step = 1.0 * np.pi/180 
#         ii = i * angle_step 
#         theta = 3 * ii 
#         phi = 5 * ii 
#         psi = 7 * ii 
#         tau = ii 
#         r = 1.0e-9 
#         delta_x = r * np.cos(tau) * peg_x_axis 
#         delta_y = r * np.sin(tau) * peg_y_axis 
#         delta_z = -i * z_step * peg_z_axis 
#         delta_pos = delta_x + delta_y + delta_z 
#         delta_a = 5.0 * np.sin(theta)
#         delta_b = 5.0 * np.sin(phi)
#         delta_c = 5.0 * np.sin(psi) 
#         delta_angle = np.array([delta_a, delta_b, delta_c]) * np.pi/180 
#         delta_pose_tool = np.concatenate([delta_pos, delta_angle]) 
#         data.ctrl = data.qpos + delta_pose_tool 
    
#         mujoco.mj_forward(model, data)
#         i += 1 

#         # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         viewer.sync()

#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)

import mujoco
import numpy as np
import time

def main():
    # Load the XML file
    model = mujoco.MjModel.from_xml_path("./env/cross_env.xml")
    data = mujoco.MjData(model)

    # Create a window and camera configuration
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    # Initialize GLFW
    mujoco.glfw.init()
    window = mujoco.glfw.create_window(1200, 900, "MuJoCo Scene", None, None)
    mujoco.glfw.make_context_current(window)

    # Initialize OpenGL context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Set camera configuration
    cam.azimuth = 90
    cam.distance = 4.0
    cam.elevation = -45
    cam.lookat = np.array([0.0, 0.0, 0.0])

    # Rendering loop
    while not mujoco.glfw.window_should_close(window):
        time_prev = time.time()

        # Step simulation
        mujoco.mj_step(model, data)

        # Get framebuffer viewport
        viewport = mujoco.MjrRect(0, 0, 1200, 900)

        # Update scene and render
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, 
            mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        mujoco.mjr_render(viewport, scene, context)

        # Swap buffers
        mujoco.glfw.swap_buffers(window)
        mujoco.glfw.poll_events()

        # Control rendering rate
        time_until_next = time_prev + 1/60 - time.time()
        if time_until_next > 0:
            time.sleep(time_until_next)

    # Clean up
    mujoco.glfw.terminate()

if __name__ == "__main__":
    main()