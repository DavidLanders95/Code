import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util


@dataclass
class Ray:
    matrix: jnp.ndarray  # 1 x 5 matrix
    z: float
    amplitude: float
    pathlength: float
    wavelength: float

    @property
    def x(self):
        return self.matrix[0, 0]

    @property
    def dx(self):
        return self.matrix[0, 1]

    @property
    def y(self):
        return self.matrix[0, 2]

    @property
    def dy(self):
        return self.matrix[0, 3]

def point_source(ps_params, ray: Ray):
    centre_yx = ps_params['centre_yx']
    slope_yx = ps_params['slope_yx']
    z = ps_params['z']
    amplitude = ps_params['amplitude']
    wavelength = ps_params['wavelength']

    matrix = jnp.array([centre_yx[1], slope_yx[1], centre_yx[0], slope_yx[0], 1.]).reshape(1, 5)

    return Ray(
        z=z,
        matrix=matrix,
        amplitude=amplitude,
        pathlength=0.0,
        wavelength=wavelength
    )


def propagate(distance, ray: Ray):

    x = ray.x + ray.dx * distance
    dx = ray.dx
    y = ray.y + ray.dy * distance
    dy = ray.dy

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)
    new_matrix = jnp.array([x, dx, y, dy, 1.]).reshape(1, 5)

    return Ray(
        z=ray.z + distance,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


def lens_step(lens_params, ray: Ray):

    f = lens_params['focal_length']
    x = ray.x
    dx = -x / f + ray.dx
    y = ray.y
    dy = -y / f + ray.dy

    pathlength = ray.pathlength - (ray.x ** 2 + ray.y ** 2) / (2 * jnp.float64(f))
    new_matrix = jnp.array([x, dx, y, dy, 1.]).reshape(1, 5)

    return Ray(
        z=ray.z,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


def def_step(def_params, ray: Ray):

    def_x = def_params['def_x']
    def_y = def_params['def_y']

    x = ray.x
    dx = ray.dx + def_x
    y = ray.y
    dy = ray.dy + def_y
    pathlength = ray.pathlength - (def_x + def_y)

    new_matrix = jnp.array([x, dx, y, dy, 1.]).reshape(1, 5)

    return Ray(
        z=ray.z,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


def sample_step(sample_params, ray: Ray):
    px_size = sample_params['px_size']
    field = sample_params['field']

    x, y = ray.x, ray.y
    dx, dy = ray.dx, ray.dy
    x_idx = jnp.round((x - px_size / 2) / px_size).astype(jnp.int32)
    y_idx = jnp.round((y - px_size / 2) / px_size).astype(jnp.int32)
    amplitude = ray.amplitude
    pathlength = ray.pathlength

    # Check if the indices are within the bounds of the sample
    if (0 <= x_idx < field.shape[1]) and (0 <= y_idx < field.shape[0]):
        amplitude *= jnp.abs(field[y_idx, x_idx])
        pathlength -= jnp.angle(field[y_idx, x_idx]) * ray.wavelength / (2 * jnp.pi)
    else:
        amplitude = 0.0
        pathlength = 0.0

    pathlength = ray.pathlength + pathlength
    new_matrix = jnp.array([x, dx, y, dy, 1]).reshape(1, 5)

    return Ray(
        z=ray.z,
        matrix=new_matrix,
        amplitude=amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


def descan_error_step(descan_error_params, ray: Ray):
    shift_error_yx = descan_error_params['shift_error_yx']
    tilt_error_yx = descan_error_params['tilt_error_yx']

    x, y = ray.x, ray.y
    dx, dy = ray.dx, ray.dy

    x += x * shift_error_yx[1]
    y += y * shift_error_yx[0]
    dx += dx * tilt_error_yx[1]
    dy += dy * tilt_error_yx[0]

    z = ray.z
    pathlength = ray.pathlength

    new_matrix = jnp.array([x, dx, y, dy, 1]).reshape(1, 5)

    return Ray(
        z=z,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


@dataclass
class Detector:
    z: float
    shape: tuple
    px_size: float
    centre_yx: tuple

    def step(ray: Ray):
        return ray

    def xy_grid(self):
        x = jnp.linspace(-self.shape[0] / 2, self.shape[0] / 2, self.shape[0]) + centre_yx[1]
        y = jnp.linspace(-self.shape[1] / 2, self.shape[1] / 2, self.shape[1]) + centre_yx[0]
        return jnp.meshgrid(y, x)

    def get_image(self, ray: Ray):
        x, y = ray.x, ray.y
        x_idx = jnp.round((x - self.px_size / 2) / self.px_size).astype(jnp.int32)
        y_idx = jnp.round((y - self.px_size / 2) / self.px_size).astype(jnp.int32)

        image = jnp.zeros(self.shape, dtype=jnp.complex64)

        # Add the amplitude and phase of each ray to the pixel it lands on
        if (0 <= x_idx < self.shape[1]) and (0 <= y_idx < self.shape[0]):
            image[y_idx, x_idx] += ray.amplitude * jnp.exp(1j * ray.pathlength)
        return image


@dataclass
class Model:
    components: list

    def run_to_end(self, ray: Ray):
        for component in self.components:
            distance = component.z - ray.z
            ray = ray.propagate(distance)
            ray = component.step(ray)
        return ray

# Register the Ray dataclass with JAX
tree_util.register_pytree_node(
    Ray,
    lambda x: ((x.matrix, x.z, x.amplitude, x.pathlength, x.wavelength), None),
    lambda _, xs: Ray(*xs)
)

# Register the Lens dataclass with JAX
# tree_util.register_pytree_node(
#     Lens,
#     lambda x: ((x.z, x.focal_length), None),
#     lambda _, xs: Lens(*xs)
# )

# # Register the Deflector dataclass with JAX
# tree_util.register_pytree_node(
#     Deflector,
#     lambda x: ((x.z, x.def_x, x.def_y), None),
#     lambda _, xs: Deflector(*xs)
# )

# Register the Detector dataclass with JAX
# tree_util.register_pytree_node(
#     Detector,
#     lambda x: ((x.z, x.shape, x.px_size, x.centre_yx), None),
#     lambda _, xs: Detector(*xs)
# )

# Register the Model dataclass with JAX
# tree_util.register_pytree_node(
#     Model,
#     lambda x: ((x.components,), None),
#     lambda _, xs: Model(*xs)
# )

# Register the Sample dataclass with JAX
# tree_util.register_pytree_node(
#     Sample,
#     lambda x: ((x.z, x.field, x.px_size), None),
#     lambda _, xs: Sample(*xs)
# )

# Register the PointSource dataclass with JAX
# tree_util.register_pytree_node(
#     PointSource,
#     lambda x: ((x.z, x.centre_yx), None),
#     lambda _, xs: PointSource(*xs)
# )
