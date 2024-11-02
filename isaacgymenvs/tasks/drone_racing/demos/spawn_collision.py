from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.drone_racing.assets import (
    create_drone_quadcopter,
    DroneQuadcopterOptions,
)

sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP
sim_params.physx.max_depenetration_velocity = 0.001  # THIS PARAM IS IMPORTANT
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1 / 60
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)

gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
gym.add_ground(sim, plane_params)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 0)

spawn_tf = gymapi.Transform()

spawn_tf.p = gymapi.Vec3(0.1, 0.1, 2.5)
drone_asset = create_drone_quadcopter(gym, sim, DroneQuadcopterOptions())
drone_actor = gym.create_actor(env, drone_asset, spawn_tf, "drone")

spawn_tf.p = gymapi.Vec3(0.0, 0.0, 1.0)
box_asset_opts = gymapi.AssetOptions()
box_asset_opts.fix_base_link = True
box_asset = gym.create_box(sim, 0.2, 0.2, 0.2, box_asset_opts)
box_actor_0 = gym.create_actor(env, box_asset, spawn_tf, "box_0")

spawn_tf.p = gymapi.Vec3(0.0, 0.0, 2.5)
box_actor_1 = gym.create_actor(env, box_asset, spawn_tf, "box_1")

spawn_tf.p = gymapi.Vec3(0.0, 0.9, 1.75)
box_actor_2 = gym.create_actor(env, box_asset, spawn_tf, "box_2")

contact_force = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))

gym.prepare_sim(sim)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "step")
gym.refresh_net_contact_force_tensor(sim)
print(contact_force[0])

while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "step" and evt.value > 0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_net_contact_force_tensor(sim)
            print(contact_force[0])
            gym.step_graphics(sim)

    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
