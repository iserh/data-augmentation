# noqa: D100
from vae.evaluation.distribution import plot_single_distribution
from vae.evaluation.evaluation_setup import eval_setup
from vae.evaluation.feature_space import visualize_feature_space
from vae.evaluation.generate_images import generate_from_normal, generate_from_test
from vae.evaluation.interpolation import (
    interpolate_two_ciffers,
    visualize_grid_interpolation,
)

setup_objects = eval_setup()

visualize_feature_space(*setup_objects)
visualize_grid_interpolation(*setup_objects)
interpolate_two_ciffers(*setup_objects)
generate_from_normal(*setup_objects)
generate_from_test(*setup_objects)
plot_single_distribution(*setup_objects)
