import numpy as np
import hppfcl
import pinocchio as pin

import meshcat
import meshcat.geometry as g


RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.0])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.0])


def get_transform(T_: hppfcl.Transform3f):
    """Returns a np.ndarray instead of a pin.SE3 or a hppfcl.Transform3f

    Args:
        T_ (hppfcl.Transform3f): transformation to change into a np.ndarray. Can be a pin.SE3 as well

    Raises:
        NotADirectoryError: _description_

    Returns:
        _type_: _description_
    """
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T


class MeshcatWrapper:
    """Wrapper displaying a robot and a target in a meshcat server."""

    def __init__(self, grid=False, axes=False):
        """Wrapper displaying a robot and a target in a meshcat server.

        Args:
            grid (bool, optional): Boolean describing whether the grid will be displayed or not. Defaults to False.
            axes (bool, optional): Boolean describing whether the axes will be displayed or not. Defaults to False.
        """

        self._grid = grid
        self._axes = axes

    def visualize(
        self,
        TARGET=None,
        robot_model=None,
        robot_collision_model=None,
        robot_visual_model=None,
    ):
        """ Returns the visualiser, displaying the robot and the target if they are in input.

        Args:
            TARGET (pin.SE3, optional): pin.SE3 describing the position of the target. Defaults to None.
            robot_model (pin.Model, optional): pinocchio model of the robot. Defaults to None.
            robot_collision_model (pin.GeometryModel, optional): pinocchio collision model of the robot. Defaults to None.
            robot_visual_model (pin.GeometryModel, optional): pinocchio visual model of the robot. Defaults to None.

        Returns:
            tuple: viewer pinocchio and viewer meshcat. 
        """
        # Creation of the visualizer,
        self.viewer = self.create_visualizer()

        if TARGET is not None:
            self._renderSphere("target", dim=5e-2, pose=TARGET)

        self._rmodel = robot_model
        self._cmodel = robot_collision_model
        self._vmodel = robot_visual_model

        Viewer = pin.visualize.MeshcatVisualizer

        self.viewer_pin = Viewer(
            self._rmodel,self._cmodel,self._vmodel
            )
        self.viewer_pin.initViewer(
            viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        )

        self.viewer_pin.loadViewerModel()
        self.viewer_pin.displayCollisions(True)
        # self.viewer_pin.displayVisuals(True)


        return self.viewer_pin, self.viewer

    def create_visualizer(self):
        """Creation of an empty visualizer.

        Returns
        -------
        vis : Meshcat.Visualizer
            visualizer from meshcat
        """
        self.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.viewer.delete()
        if not self._grid:
            self.viewer["/Grid"].set_property("visible", False)
        if not self._axes:
            self.viewer["/Axes"].set_property("visible", False)
        return self.viewer

    def _renderSphere(self, e_name: str, dim: np.ndarray, pose: pin.SE3, color=GREEN):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(g.Sphere(dim), self._meshcat_material(*color))
        T = get_transform(pose)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

    def _meshcat_material(self, r, g, b, a):
        """Converting RGBA color to meshcat material.

        Args:
            r (int): color red
            g (int): color green
            b (int): color blue
            a (int): opacity

        Returns:
            material : meshcat.geometry.MeshPhongMaterial(). Material for meshcat
        """
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        return material


import numpy as np

# Define a function to convert HPPFCL shape to URDF geometry element
def hppfcl_shape_to_urdf_geometry(shape):
    shape_type = shape.get_type()
    
    if shape_type == 'Box':
        size = shape.get_size()
        return f'<box size="{size[0]} {size[1]} {size[2]}" />'
    elif shape_type == 'Sphere':
        radius = shape.get_radius()
        return f'<sphere radius="{radius}" />'
    elif shape_type == 'Cylinder':
        radius = shape.get_radius()
        height = shape.get_height()
        return f'<cylinder radius="{radius}" length="{height}" />'
    else:
        raise ValueError("Unsupported shape type")

# Define a function to convert SE3 transform to URDF origin tag
def se3_to_urdf_origin(transform):
    translation = transform[:3, 3]
    rotation_matrix = transform[:3, :3]
    
    # Convert rotation matrix to roll, pitch, yaw (Tait-Bryan angles)
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return f'<origin xyz="{translation[0]} {translation[1]} {translation[2]}" rpy="{roll} {pitch} {yaw}" />'

import hppfcl
import pinocchio as pin
import numpy as np

def generate_urdf_string(shape, placement, link_name="link"):
    # Convert SE3 placement to translation and RPY (roll, pitch, yaw) format
    translation = placement.translation
    rotation = placement.rotation
    rpy = pin.rpy.matrixToRpy(rotation)

    urdf = f"""
    <collision>
      <origin xyz="{translation[0]} {translation[1]} {translation[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
      <geometry>"""

    if isinstance(shape, hppfcl.Box):
        size = f"{shape.halfSide[0] * 2} {shape.halfSide[1]} {shape.halfSide[2]}"
        urdf += f"""
        <box size="{size}"/>"""
    elif isinstance(shape, hppfcl.Sphere):
        radius = shape.radius
        urdf += f"""
        <sphere radius="{radius}"/>"""
    elif isinstance(shape, hppfcl.Cylinder):
        radius = shape.radius
        length = shape.halfLength * 2
        urdf += f"""
        <cylinder radius="{radius}" length="{length}"/>"""
    elif isinstance(shape, hppfcl.Capsule):
        radius = shape.radius
        length = shape.halfLength * 2

        # Add cylinder for the middle part of the capsule
        urdf += f"""
        <cylinder radius="{radius}" length="{length}"/>"""

        # Calculate the positions for the sphere caps
        cap1_translation = translation + rotation @ np.array([0, 0, length / 2])
        cap2_translation = translation - rotation @ np.array([0, 0, length / 2])

        urdf += f"""
      </geometry>
    </collision>
    <collision>
      <origin xyz="{cap1_translation[0]} {cap1_translation[1]} {cap1_translation[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
      <geometry>
        <sphere radius="{radius}"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="{cap2_translation[0]} {cap2_translation[1]} {cap2_translation[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
      <geometry>
        <sphere radius="{radius}"/>
      </geometry>
    </collision>"""
    else:
        raise NotImplementedError(f"Shape type {type(shape)} not supported")
    return urdf

if __name__ == "__main__":
    
    urdf_model_path = "/home/arthur/Desktop/Code/convexify-urdf/kuka/urdf/iiwa_convex.urdf"
    meshes = "/home/arthur/Desktop/Code/convexify-urdf/kuka"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, meshes)
    MeshcatVis = MeshcatWrapper()

    collision_model_copy = collision_model.copy()
    for obj in collision_model_copy.geometryObjects:
        print(obj.name)
        name = "L7_0"
        
        if name ==  obj.name:
            # print(f"parentJoint : {obj.parentJoint}")
            # print(f"parentFrame : {obj.parentFrame}")

            placement = pin.SE3.Identity()
            placement.translation = np.array([0.0,0.0,0])
            placement.rotation = pin.rpy.rpyToMatrix(0,0,0)
            radius = 0.045
            halfLength = 0.05
            hpp_geom = hppfcl.Capsule(radius, halfLength)
            geom = pin.GeometryObject(
            name + "0",
            obj.parentFrame,
            obj.parentJoint,
            hpp_geom,
            placement,
        )      
            geom.meshColor = np.array([0.9, 0.9,0.9, 0.9])
            collision_model.addGeometryObject(geom)
            collision_model.removeGeometryObject(name)

    
    # Generating the meshcat visualizer
    # print(generate_urdf_string(hpp_geom, placement))
    vis = MeshcatVis.visualize(
        robot_model=model, robot_visual_model=visual_model, robot_collision_model=collision_model
    )
    q = pin.neutral(model)

    vis[0].display(q)
    
    # input()
    # q = pin.randomConfiguration(model)

    # vis[0].display(q)
    