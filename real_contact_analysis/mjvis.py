import mujoco
import numpy as np
import time

def load_and_simulate(xml_path, simulation_time=100000.0, render=True):
    """
    Load and simulate a MuJoCo XML file
    
    Args:
        xml_path (str): Path to the XML file
        simulation_time (float): Duration of simulation in seconds
        render (bool): Whether to render the simulation
    """
    # Load the model from XML file
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Create visualization objects
    if render:
        renderer = mujoco.Renderer(model)
        scene_option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10000)
        cam = mujoco.MjvCamera()
        
        # Set camera configuration
        cam.distance = 4.0
        cam.azimuth = 90.0
        cam.elevation = -45.0
        cam.lookat[0] = 0
        cam.lookat[1] = 0
        cam.lookat[2] = 0

    # Simulation parameters
    timestep = model.opt.timestep
    num_steps = int(simulation_time / timestep)

    # Simulation loop
    for step in range(num_steps):
        mujoco.mj_step(model, data)
        
        if render:
            mujoco.mjv_updateScene(
                model,                # const mjModel* m
                data,                 # mjData* d
                scene_option,         # const mjvOption* opt
                None,                 # const mjvPerturb* pert
                cam,                  # mjvCamera* cam
                mujoco.mjtCatBit.mjCAT_ALL.value,  # int catmask
                scene                 # mjvScene* scn
            )
            renderer.update_scene(data)
            renderer.render()
            # time.sleep(timestep)

if __name__ == "__main__":
    xml_path = "/home/rp/abhay_ws/mujoco_contact_graph_generation/env/cross_env_v2.xml"  # Replace with your XML file path
    load_and_simulate(xml_path)

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.