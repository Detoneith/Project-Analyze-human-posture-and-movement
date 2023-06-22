import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import mpl_toolkits.mplot3d as mp3d

world_origin = (1, 4, 1)


class Camera:
    def __init__(self, P):
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None
    def translate(self, translation_vector):
        T = np.eye(4)
        T[:3, 3] = translation_vector
        self.P = np.dot(self.P, T)
    def rotate(self, rotation_vector):
        R = rotation_matrix(rotation_vector)
        self.P = np.dot(self.P, R)
    def position(self):
        return self.P[:3, 3]






def rotation_matrix(a):
    R = np.eye(4)
    R[:3, :3] = expm(np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]))
    return R

def draw_axis(ax, position, label, color):
    x_arrow = np.array([1, 0, 0])
    y_arrow = np.array([0, 1, 0])
    z_arrow = np.array([0, 0, 1])
    ax.quiver(*position, *x_arrow, color=color, arrow_length_ratio=0.1)
    ax.quiver(*position, *y_arrow, color=color, arrow_length_ratio=0.1)
    ax.quiver(*position, *z_arrow, color=color, arrow_length_ratio=0.1)
    ax.text(*(position + x_arrow), f"{label}_x")
    ax.text(*(position + y_arrow), f"{label}_y")
    ax.text(*(position + z_arrow), f"{label}_z")

def drawRep(ax):
    x, y, z = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
    u, v, w = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")

def drawPlan(ax, cam):
    print('pos cam = ', cam)
    w = 3
    h = 2
    dx = w*0.5
    dy = 3
    dz = h*0.5
    top = [(cam[0]-2, cam[1]+3, cam[2]-2),
           (cam[0]+2, cam[1]+3, cam[2]-2),
           (cam[0]+2, cam[1]+3, cam[2]+2),
           (cam[0]-2, cam[1]+3,  cam[2]+2),
           ]
    alpha = 0.5
    face = mp3d.art3d.Poly3DCollection([top], alpha=0.5, linewidth=1)
    face.set_facecolor((0, 0, 1, alpha))
    ax.add_collection3d(face)

def drawPoint(ax, p, cam):
    x, y, z = p[0], p[1], p[2]
    ax.scatter(x, y, z)
    vx = np.array([cam[0], x])
    vy = np.array([cam[1], y])
    vz = np.array([cam[2], z])
    ax.plot3D(vx, vy, vz)

def _get_roll_matrix(theta_x: float = 0.0) -> np.ndarray:
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x)],
            [0.0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    return Rx


def _get_pitch_matrix(theta_y: float = 0.0) -> np.ndarray:
    Ry = np.array(
        [
            [np.cos(theta_y), 0.0, np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_y), 0.0, np.cos(theta_y)],
        ]
    )
    return Ry


def _get_yaw_matrix(theta_z: float = 0.0) -> np.ndarray:
    Rz = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0.0],
            [np.sin(theta_z), np.cos(theta_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Rz


def get_rotation_matrix(
    theta_x: float = 0.0, theta_y: float = 0.0, theta_z: float = 0.0
) -> np.ndarray:
    # Roll
    Rx = _get_roll_matrix(theta_x)
    # Pitch
    Ry = _get_pitch_matrix(theta_y)
    # Yaw
    Rz = _get_yaw_matrix(theta_z)
    return Rz @ Ry @ Rx

def to_inhomogeneus(X: np.ndarray) -> np.ndarray:
    if X.ndim > 1:
        raise ValueError("x must be one-dimensional.")

    return (X / X[-1])[:-1]


def to_homogeneus(X: np.ndarray) -> np.ndarray:
    if X.ndim > 1:
        raise ValueError("X must be one-dimensional.")

    return np.hstack([X, 1])

def get_calibration_matrix(
    f: float,
    px: float = 0.0,
    py: float = 0.0,
    mx: float = 1.0,
    my: float = 1.0,
) -> np.ndarray:
    K = np.diag([mx, my, 1]) @ np.array([[f, 0.0, px], [0.0, f, py], [0.0, 0.0, 1.0]])
    return K

def get_plucker_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = to_homogeneus(A)
    B = to_homogeneus(B)
    L = A.reshape(-1, 1) * B.reshape(1, -1) - B.reshape(-1, 1) * A.reshape(1, -1)
    return L

def get_plane_from_three_points(
    X1: np.ndarray, X2: np.ndarray, X3: np.ndarray
) -> np.ndarray:
    pi = np.hstack([np.cross(X1 - X3, X2 - X3), -X3 @ np.cross(X1, X2)])
    return pi

def plot_world_and_camera(world_coordinates, camera, X=[2, 4, 2], x=[0, 0, 0]):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    drawRep(ax)
    world_x, world_y, world_z = zip(world_coordinates)
    ax.scatter(world_x, world_y, world_z, c='b', marker='o', label='World Coordinates', s=54)
    camera_x, camera_y, camera_z = camera.position()
    drawPlan(ax, camera.position())
    drawPoint(ax, X, camera.position())
    drawPoint(ax, x, camera.position())
    ax.scatter(camera_x, camera_y, camera_z, c='r', marker='^', label='Camera Position', s=54)
    draw_axis(ax, camera.position(), "C", color="red")
    draw_axis(ax, world_origin, "W", color="blue")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 6)
    ax.legend()
    plt.show()


# Set up Camera
P = np.hstack((np.eye(3), np.array([[3], [0], [2]])))
print('Camera ')
print(P)


cam = Camera(P)
translation_vector = np.array([0, -5, 0])
cam.translate(translation_vector)
# Plot the World and Camera Coordinate Systems
plot_world_and_camera(world_origin, cam)

# translation_vector = np.array([-3, 1, 1])
rotation_axis = np.array([0, 0.5, 1])
rotation_angle = np.radians(20)
rotation_vector = rotation_axis * rotation_angle
# cam.translate(translation_vector)
cam.rotate(rotation_vector)
## Plot the World and Camera Coordinate Systems after moving the Camera
#plot_world_and_camera(world_origin, cam)


C = np.array([3, -5, 2])
dx = np.array([5, -2, 1])
dy = np.array([4, -2, 1])
dz = np.array([5, -2, 2])
X = np.array([1, 2, 5])
L = get_plucker_matrix(C, X)
plan = get_plane_from_three_points(dx, dy, dz)
x = to_inhomogeneus(L @ plan)
print(plan)
print('Plucker ')
print(L)
print('x = ', x)
print('plan')


plot_world_and_camera(world_origin, cam, X, x)











FOCAL_LENGTH = 3.0 # focal length
PX= 2.0 # principal point x-coordinate
PY= 1.0 # principal point y-coordinate
MX = 1.0 # number of pixels per unit distance in image coordinates in x direction
MY = 1.0 # number of pixels per unit distance in image coordinates in y direction
THETA_X = np.pi / 2.0 # roll angle
THETA_Y = 0.0 # pitch angle
THETA_Z = np.pi # yaw angle
C = np.array([3, -5, 2]) # camera centre
IMAGE_HEIGTH = 4
IMAGE_WIDTH = 6

#calibration_kwargs = 'f': FOCAL_LENGTH, 'px': PX, 'py': PY, 'mx': MX, 'my': MY}
# rotation_kwargs = {“theta_x”: THETA_X, “theta_y”: THETA_Y, “theta_z”: THETA_Z}
# projection_kwargs = {**calibration_kwargs, **rotation_kwargs, “C”: C}


#Calibration Matrix
K = [[3., 0., 2.],
 [0., 3., 1.],
 [0., 0., 1.]]

#Rotation Matrix
R = [[-1., -0., 0.],
 [0., -0., 1.],
 [0., 1., 0.]]

#Projection Matrix
P = [[-3., 2., 0., 10.],
 [0., 1., 3., -1.],
 [0., 1., 0., 5.]]

dx, dy, dz = np.eye(3)
print('dx =', dx)
print('dy =', dy)
print('dz =', dz)

translation_vector = np.array([0, -1, 0])
cam.translate(translation_vector)
# Plot the World and Camera Coordinate Systems
#plot_world_and_camera(world_origin, cam)
