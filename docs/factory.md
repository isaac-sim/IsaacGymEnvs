Factory
=======

Here we provide additional documentation on the Factory assets, environments, controllers, and simulation methods. This documentation will be regularly updated and extended.

Before starting to use Factory, we would **highly** recommend familiarizing yourself with Isaac Gym, including the simpler RL examples.

Assets
------

Meshes for our assets are located in the [mesh subdirectory](../../assets/factory/mesh). The meshes for the electrical connectors are currently unavailable, as the raw designs were originally sourced from manufacturers and are undergoing approval for release.

URDF files for our assets are located in the [urdf subdirectory](../../assets/factory/urdf/).

There are also YAML files located in the [yaml subdirectory](../../assets/factory/yaml/). These files contain asset-related constants that are used by the Factory RL examples.

Onshape CAD files for all the assets will also be made available shortly.

Classes, Modules, and Abstract Base Classes
-------------------------------------------

The class hierarchy for the Factory examples is as follows:

[FactoryBase](../isaacgymenvs/tasks/factory/factory_base.py): assigns physics simulation parameters; imports Franka and table assets; assigns asset options for the Franka and table; translates higher-level controller selection into lower-level controller parameters; sets targets for controller

Each of the environment classes inherits the base class:
* [FactoryEnvNutBolt](../isaacgymenvs/tasks/factory/factory_env_nut_bolt.py): imports nut and bolt assets; assigns asset options for the nuts and bolts; creates Franka, table, nut, and bolt actors
* [FactoryEnvInsertion](../isaacgymenvs/tasks/factory/factory_env_insertion.py): imports plug and socket assets (including pegs and holes); assigns asset options for the plugs and sockets; creates Franka, table, plug, and socket actors
* [FactoryEnvGears](../isaacgymenvs/tasks/factory/factory_env_gears.py): imports gear and gear base assets; assigns asset options for the gears and gear base; creates Franka, table, gears, and gear base actors

Each of the task classes inherits the corresponding environment class:
* [FactoryTaskNutBoltPick](../isaacgymenvs/tasks/factory/factory_task_nut_bolt_pick.py): contains higher-level RL code for the Pick subpolicy (e.g., applying actions, defining observations, defining rewards, resetting environments), which is used by the lower-level [rl-games](https://github.com/Denys88/rl_games) library
* [FactoryTaskNutBoltPlace](../isaacgymenvs/tasks/factory/factory_task_nut_bolt_place.py): contains higher-level RL code for the Place subpolicy
* [FactoryTaskNutBoltScrew](../isaacgymenvs/tasks/factory/factory_task_nut_bolt_screw.py): contains higher-level RL code for the Screw subpolicy
* [FactoryTaskInsertion](../isaacgymenvs/tasks/factory/factory_task_insertion.py): contains template for Insertion policy
* [FactoryTaskGears](../isaacgymenvs/tasks/factory/factory_task_gears.py): contains template for Gears policy

There is also a control module ([factory_control.py](../isaacgymenvs/tasks/factory/factory_control.py)) that is imported by [factory_base.py](../isaacgymenvs/tasks/factory/factory_base.py) and contains the lower-level controller code that converts controller targets to joint torques.

Finally, there are abstract base classes that define the necessary methods for base, environment, and task classes ([factory_schema_class_base.py](../isaacgymenvs/tasks/factory/factory_schema_class_base.py), [factory_schema_class_env.py](../isaacgymenvs/tasks/factory/factory_schema_class_env.py), and [factory_schema_class_task.py](../isaacgymenvs/tasks/factory/factory_schema_class_task.py)). These are useful to review in order to better understand the structure of the code, but you will probably not need to modify them. They are also recommended to inherit if you would like to quickly add your own environments and tasks.

Configuration Files and Schema
------------------------------

In addition to the task-level configuration files described in the primary readme, there is also a base-level configuration file ([FactoryBase.yaml](../isaacgymenvs/cfg/task/FactoryBase.yaml)) and environment-level configuration files ([FactoryEnvNutBolt.yaml](../isaacgymenvs/cfg/task/FactoryEnvNutBolt.yaml), [FactoryEnvInsertion.yaml](../isaacgymenvs/cfg/task/FactoryEnvInsertion.yaml), and [FactoryEnvGears.yaml](../isaacgymenvs/cfg/task/FactoryEnvGears.yaml)).

There are schema for these configuration files ([factory_schema_config_base.py](../isaacgymenvs/tasks/factory/factory_schema_config_base.py), [factory_schema_config_env.py](../isaacgymenvs/tasks/factory/factory_schema_config_env.py), and [factory_schema_config_task.py](../isaacgymenvs/tasks/factory/factory_schema_config_tasks.py)). These schema are enforced for the base and environment configuration files, but not enforced for the task configuration files. These are useful to review in order to better understand the structure of the configuration files and see descriptions of common parameters, but you will probably not need to modify them.

Controllers
-----------

Controller types and gains can be specified in the task-level configuration files. In addition to the 7 controllers described in the Factory paper, there is also the option of using Gym's built-in joint-space PD controller. This controller is generally quite stable, but uses a symplectic integrator that may introduce some artifical damping.

The controllers are implemented as follows:
* When launching a task, the higher-level controller type is parsed into lower-level controller options (e.g., joint space or task space, inertial compensation or no inertial compensation)
* At each time step (e.g., see [factory_task_nut_bolt_pick.py](../isaacgymenvs/tasks/factory/factory_task_nut_bolt_pick.py)), the actions are applied as controller targets, the appropriate Jacobians are computed in [factory_base.py](../isaacgymenvs/tasks/factory/factory_base.py), and the lower-level controller options, targets, and Jacobians are used by the lower-level controller code ([factory_control.py](../isaacgymenvs/tasks/factory/factory_control.py)) to generate corresponding joint torques.

This controller implementation will be made simpler and more developer-friendly in future updates.

Collisions and Contacts
-----------------------

**URDF Configuration:**

Different pairs of interacting objects can use different geometric representations (e.g., convex decompositions, triangular meshes, SDFs) to generate contacts and resolve collisions. If you would like any asset (or link of an asset) to engage in SDF collisions, you simply need to edit its URDF description and add an `<sdf>` element to its `<collision>` element. For example:

```
<?xml version="1.0"?>
<robot name="nut">
    <link name="nut">
        <visual>
            <geometry>
                <mesh filename="nut.obj"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="nut.obj"/>
            </geometry>
            <sdf resolution="256"/>
        </collision>
    </link>
</robot>
```

SDFs are computed from the mesh file along a discrete voxel grid. The resolution attribute specifies the number of voxels along the longest dimension of the object.

**Collision Logic:**

For a pair of colliding objects, by including or not including the `<sdf>` field in the corresponding URDFs, the collision scheme used for that pair of objects can be controlled. Specifically, consider 2 colliding objects, Object A and Object B.

* If A and B both have an `<sdf>` field, SDF-mesh collision will be applied. The object with the larger number of features (i.e., triangles) will be represented as an SDF, and the triangular mesh of the other object will be queried against the SDF to check for collisions and generate contacts. At any timestep, if too few contacts are generated between the objects, the SDF-mesh identities of the objects will be flipped, and contacts will be regenerated.

* If A has an `<sdf>` field and B does not, convex-mesh collision will be applied. Object A will be represented as a triangular mesh, and object B will be represented as a convex.

* If neither A nor B has an `<sdf>` tag, PhysX’s default convex-convex collision will be applied.

**Best Practices and Debugging:**

For small, complex parts (e.g., nuts and bolts), use an SDF resolution between 256 and 512.

If you are observing **minor penetration issues**, try the following:

* Increase `sim_params.physx.contact_offset` (global setting) or `asset_options.contact_offset` (asset-specific setting), which is the minimum distance between 2 objects at which contacts are generated. The default value in Factory is 0.005. As a rule of thumb, keep this value at least 1 order-of-magnitude greater than `v * dt / n`, where `v` is the maximum characteristic velocity of the object, `dt` is the timestep size, and `n` is the number of substeps.

* Increase the density of your meshes (i.e., number of triangles). In particular, when exporting OBJ files from some CAD programs, large flat surfaces can be meshed with very few triangles. Currently, PhysX generates a maximum of 1 contact per triangle; thus, very few contacts are generated on such surfaces. Software like Blender can be used to quickly increase the number of triangles on regions of a mesh using methods like edge subdivision.

* Increase `sim_params.physx.rest_offset` (global setting) or `asset_options.rest_offset` (asset-specific setting), which is the minimum separation distance between 2 objects in contact. The default value in Factory is 0.0. As a rule of thumb, for physically-accurate results, keep this value at least 1 order-of-magnitude less than the minimum characteristic length of your object (e.g., the thickness of your mug or bowl).

If you are observing **severe penetration issues** (e.g., objects passing freely through other objects), PhysX's contact buffer is likely overflowing. You may not see explicit warnings in the terminal output. Try the following:

* Reduce the number of environments. As a reference, we tested most of the Factory tasks with 128 environments. You can also try reducing them further.

* Increase `sim_params.physx.max_gpu_contact_pairs`, which is the size of your GPU contact buffer. The default value in Factory is 1024^2. You will likely not be able to exceed a factor of 50 beyond this value due to GPU memory limits.

* Increase `sim_params.physx.default_buffer_size_multiplier`, which will scale additional buffers used by PhysX. The default value in Factory is 8.

If you are experiencing any **stability issues** (e.g., jitter), try the following:

* Decrease `sim_params.dt`, increase `sim_params.substeps`, and/or increase `sim_params.physx.num_position_iterations`, which control the size of timesteps, substeps, and solver iterations. In general, increasing the number of iterations will slow down performance less than modifying the other parameters.

* Increase `sim_params.physx.contact_offset` and/or `sim_params.physx.friction_offset_threshold`, which are the distances at which contacts and frictional constraints are generated.

* Increase the SDF resolution in the asset URDFs.

* Increase the coefficient of friction and/or decrease the coefficient of restitution between the actors in the scene. However, be careful not to violate physically-reasonable ranges (e.g., friction values in excess of 2.0).

* Tune the gains of your controllers. Instability during robot-object contact may also be a result of poorly-tuned controllers, rather than underlying physics simulation issues. As in the real world, some controllers can be notoriously hard to tune.
