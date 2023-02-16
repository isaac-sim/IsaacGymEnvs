# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for 3D pose conversion."""
import math
import numpy as np

from pybullet_utils import transformations

VECTOR3_0 = np.zeros(3, dtype=np.float64)
VECTOR3_1 = np.ones(3, dtype=np.float64)
VECTOR3_X = np.array([1, 0, 0], dtype=np.float64)
VECTOR3_Y = np.array([0, 1, 0], dtype=np.float64)
VECTOR3_Z = np.array([0, 0, 1], dtype=np.float64)

# QUATERNION_IDENTITY is the multiplicative identity 1.0 + 0i + 0j + 0k.
# When interpreted as a rotation, it is the identity rotation.
QUATERNION_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def Vector3RandomNormal(sigma, mu=VECTOR3_0):
  """Returns a random 3D vector from a normal distribution.

  Each component is selected independently from a normal distribution.

  Args:
    sigma: Scale (or stddev) of distribution for all variables.
    mu: Mean of distribution for each variable.

  Returns:
    A 3D vector in a numpy array.
  """

  random_v3 = np.random.normal(scale=sigma, size=3) + mu
  return random_v3


def Vector3RandomUniform(low=VECTOR3_0, high=VECTOR3_1):
  """Returns a 3D vector selected uniformly from the input box.

  Args:
    low: The min-value corner of the box.
    high: The max-value corner of the box.

  Returns:
    A 3D vector in a numpy array.
  """

  random_x = np.random.uniform(low=low[0], high=high[0])
  random_y = np.random.uniform(low=low[1], high=high[1])
  random_z = np.random.uniform(low=low[2], high=high[2])
  return np.array([random_x, random_y, random_z])


def Vector3RandomUnit():
  """Returns a random 3D vector with unit length.

  Generates a 3D vector selected uniformly from the unit sphere.

  Returns:
    A normalized 3D vector in a numpy array.
  """
  longitude = np.random.uniform(low=-math.pi, high=math.pi)
  sin_latitude = np.random.uniform(low=-1.0, high=1.0)
  cos_latitude = math.sqrt(1.0 - sin_latitude * sin_latitude)
  x = math.cos(longitude) * cos_latitude
  y = math.sin(longitude) * cos_latitude
  z = sin_latitude
  return np.array([x, y, z], dtype=np.float64)


def QuaternionNormalize(q):
  """Normalizes the quaternion to length 1.

  Divides the quaternion by its magnitude.  If the magnitude is too
  small, returns the quaternion identity value (1.0).

  Args:
    q: A quaternion to be normalized.

  Raises:
    ValueError: If input quaternion has length near zero.

  Returns:
    A quaternion with magnitude 1 in a numpy array [x, y, z, w].

  """
  q_norm = np.linalg.norm(q)
  if np.isclose(q_norm, 0.0):
    raise ValueError(
        'Quaternion may not be zero in QuaternionNormalize: |q| = %f, q = %s' %
        (q_norm, q))
  return q / q_norm


def QuaternionFromAxisAngle(axis, angle):
  """Returns a quaternion that generates the given axis-angle rotation.

  Returns the quaternion: sin(angle/2) * axis + cos(angle/2).

  Args:
    axis: Axis of rotation, a 3D vector in a numpy array.
    angle: The angle of rotation (radians).

  Raises:
    ValueError: If input axis is not a normalizable 3D vector.

  Returns:
    A unit quaternion in a numpy array.

  """
  if len(axis) != 3:
    raise ValueError('Axis vector should have three components: %s' % axis)
  axis_norm = np.linalg.norm(axis)
  if np.isclose(axis_norm, 0.0):
    raise ValueError('Axis vector may not have zero length: |v| = %f, v = %s' %
                     (axis_norm, axis))
  half_angle = angle * 0.5
  q = np.zeros(4, dtype=np.float64)
  q[0:3] = axis
  q[0:3] *= math.sin(half_angle) / axis_norm
  q[3] = math.cos(half_angle)
  return q


def QuaternionToAxisAngle(quat, default_axis=VECTOR3_Z, direction_axis=None):
  """Calculates axis and angle of rotation performed by a quaternion.

  Calculates the axis and angle of the rotation performed by the quaternion.
  The quaternion should have four values and be normalized.

  Args:
    quat: Unit quaternion in a numpy array.
    default_axis: 3D vector axis used if the rotation is near to zero. Without
      this default, small rotations would result in an exception.  It is
      reasonable to use a default axis for tiny rotations, because zero angle
      rotations about any axis are equivalent.
    direction_axis: Used to disambiguate rotation directions.  If the
      direction_axis is specified, the axis of the rotation will be chosen such
      that its inner product with the direction_axis is non-negative.

  Raises:
    ValueError: If quat is not a normalized quaternion.

  Returns:
    axis: Axis of rotation.
    angle: Angle in radians.
  """
  if len(quat) != 4:
    raise ValueError(
        'Quaternion should have four components [x, y, z, w]: %s' % quat)
  if not np.isclose(1.0, np.linalg.norm(quat)):
    raise ValueError('Quaternion should have unit length: |q| = %f, q = %s' %
                     (np.linalg.norm(quat), quat))
  axis = quat[:3].copy()
  axis_norm = np.linalg.norm(axis)
  min_axis_norm = 1e-8
  if axis_norm < min_axis_norm:
    axis = default_axis
    if len(default_axis) != 3:
      raise ValueError('Axis vector should have three components: %s' % axis)
    if not np.isclose(np.linalg.norm(axis), 1.0):
      raise ValueError('Axis vector should have unit length: |v| = %f, v = %s' %
                       (np.linalg.norm(axis), axis))
  else:
    axis /= axis_norm
  sin_half_angle = axis_norm
  if direction_axis is not None and np.inner(axis, direction_axis) < 0:
    sin_half_angle = -sin_half_angle
    axis = -axis
  cos_half_angle = quat[3]
  half_angle = math.atan2(sin_half_angle, cos_half_angle)
  angle = half_angle * 2
  return axis, angle


def QuaternionRandomRotation(max_angle=math.pi):
  """Creates a random small rotation around a random axis.

  Generates a small rotation with the axis vector selected uniformly
  from the unit sphere and an angle selected from a uniform
  distribution over [0, max_angle].

  If the max_angle is not specified, the rotation should be selected
  uniformly over all possible rotation angles.

  Args:
    max_angle: The maximum angle of rotation (radians).

  Returns:
    A unit quaternion in a numpy array.

  """

  angle = np.random.uniform(low=0, high=max_angle)
  axis = Vector3RandomUnit()
  return QuaternionFromAxisAngle(axis, angle)


def QuaternionRotatePoint(point, quat):
  """Performs a rotation by quaternion.

  Rotate the point by the quaternion using quaternion multiplication,
  (q * p * q^-1), without constructing the rotation matrix.

  Args:
    point: The point to be rotated.
    quat: The rotation represented as a quaternion [x, y, z, w].

  Returns:
    A 3D vector in a numpy array.
  """

  q_point = np.array([point[0], point[1], point[2], 0.0])
  quat_inverse = transformations.quaternion_inverse(quat)
  q_point_rotated = transformations.quaternion_multiply(
      transformations.quaternion_multiply(quat, q_point), quat_inverse)
  return q_point_rotated[:3]


def IsRotationMatrix(m):
  """Returns true if the 3x3 submatrix represents a rotation.

  Args:
    m: A transformation matrix.

  Raises:
    ValueError: If input is not a matrix of size at least 3x3.

  Returns:
    True if the 3x3 submatrix is a rotation (orthogonal).
  """
  if len(m.shape) != 2 or m.shape[0] < 3 or m.shape[1] < 3:
    raise ValueError('Matrix should be 3x3 or 4x4: %s\n %s' % (m.shape, m))
  rot = m[:3, :3]
  eye = np.matmul(rot, np.transpose(rot))
  return np.isclose(eye, np.identity(3), atol=1e-4).all()

# def ZAxisAlignedRobotPoseTool(robot_pose_tool):
#   """Returns the current gripper pose rotated for alignment with the z-axis.

#   Args:
#     robot_pose_tool: a pose3d.Pose3d() instance.

#   Returns:
#     An instance of pose.Transform representing the current gripper pose
#     rotated for alignment with the z-axis.
#   """
#   # Align the current pose to the z-axis.
#   robot_pose_tool.quaternion = transformations.quaternion_multiply(
#       RotationBetween(
#           robot_pose_tool.matrix4x4[0:3, 0:3].dot(np.array([0, 0, 1])),
#           np.array([0.0, 0.0, -1.0])), robot_pose_tool.quaternion)
#   return robot_pose_tool

# def RotationBetween(a_translation_b, a_translation_c):
#   """Computes the rotation from one vector to another.

#   The computed rotation has the property that:

#     a_translation_c = a_rotation_b_to_c * a_translation_b

#   Args:
#     a_translation_b: vec3, vector to rotate from
#     a_translation_c: vec3, vector to rotate to

#   Returns:
#     a_rotation_b_to_c: new Orientation
#   """
#   rotation = rotation3.Rotation3.rotation_between(
#       a_translation_b, a_translation_c, err_msg='RotationBetween')
#   return rotation.quaternion.xyzw
