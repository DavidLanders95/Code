import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util


@dataclass
class Ray:
    z: float
    matrix: jnp.ndarray  # 1 x 5 matrix
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

    def propagate(self, distance: float):
        self.matrix = self.matrix.at[0, 0].set(self.x + self.dx * distance)
        self.matrix = self.matrix.at[0, 2].set(self.y + self.dy * distance)
        self.pathlength += distance
        self.z += distance
        return self


@dataclass
class Lens:
    z: float
    focal_length: float

    def step(self, ray: Ray):

        f = self.focal_length

        x = ray.x
        dx = ray.dx
        y = ray.y
        dy = ray.dy

        ray.matrix = ray.matrix.at[0, 1].set(-x / f + dx)
        ray.matrix = ray.matrix.at[0, 3].set(-y / f + dy)

        ray.pathlength -= (ray.x ** 2 + ray.y ** 2) / (2 * jnp.float64(f))

        return ray


@dataclass
class Deflector:
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):

        def_x = self.def_x
        def_y = self.def_y

        dx = ray.dx
        dy = ray.dy

        ray.matrix = ray.matrix.at[0, 1].set(dx + def_x)
        ray.matrix = ray.matrix.at[0, 3].set(dy + def_y)
        ray.pathlength -= (def_x + def_y)

        return ray


@dataclass
class Detector:
    z: float
    shape: tuple
    px_size: float
    centre_yx: tuple

    def step(self, ray: Ray):
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
class PointSource:
    z: float
    centre_yx: tuple

    def step(self, ray: Ray):
        return ray


@dataclass
class Model:
    components: list

    def run_to_end(self, ray: Ray):
        for component in self.components:
            distance = component.z - ray.z
            ray = ray.propagate(distance)
            ray = component.step(ray)
        return ray

@dataclass
class Sample:
    z: float
    field: jnp.ndarray
    px_size: float

    def step(self, ray: Ray):
        x, y = ray.x, ray.y
        x_idx = jnp.round((x - self.px_size / 2) / self.px_size).astype(jnp.int32)
        y_idx = jnp.round((y - self.px_size / 2) / self.px_size).astype(jnp.int32)

        # Check if the indices are within the bounds of the sample
        if (0 <= x_idx < self.field.shape[1]) and (0 <= y_idx < self.field.shape[0]):
            ray.amplitude *= jnp.abs(self.field[y_idx, x_idx])
            ray.path_length -= jnp.angle(self.field[y_idx, x_idx]) * ray.wavelength / (2 * jnp.pi)
        else:
            ray.amplitude = 0.0
        return ray


# Register the Ray dataclass with JAX
tree_util.register_pytree_node(
    Ray,
    lambda x: ((x.z, x.matrix, x.amplitude, x.pathlength, x.wavelength), None),
    lambda _, xs: Ray(*xs)
)

# Register the Lens dataclass with JAX
tree_util.register_pytree_node(
    Lens,
    lambda x: ((x.z, x.focal_length), None),
    lambda _, xs: Lens(*xs)
)

# Register the Deflector dataclass with JAX
tree_util.register_pytree_node(
    Deflector,
    lambda x: ((x.z, x.def_x, x.def_y), None),
    lambda _, xs: Deflector(*xs)
)

# Register the Detector dataclass with JAX
tree_util.register_pytree_node(
    Detector,
    lambda x: ((x.z, x.shape, x.px_size, x.centre_yx), None),
    lambda _, xs: Detector(*xs)
)

# Register the Model dataclass with JAX
tree_util.register_pytree_node(
    Model,
    lambda x: ((x.components,), None),
    lambda _, xs: Model(*xs)
)

# Register the Sample dataclass with JAX
tree_util.register_pytree_node(
    Sample,
    lambda x: ((x.z, x.field, x.px_size), None),
    lambda _, xs: Sample(*xs)
)

# Register the PointSource dataclass with JAX
tree_util.register_pytree_node(
    PointSource,
    lambda x: ((x.z, x.centre_yx), None),
    lambda _, xs: PointSource(*xs)
)
