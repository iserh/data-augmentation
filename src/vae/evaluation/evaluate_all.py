# noqa: D100
from data_augmentation.VAE.evaluation.distribution import plot_single_distribution
from data_augmentation.VAE.evaluation.evaluation_setup import eval_setup
from data_augmentation.VAE.evaluation.feature_space import visualize_feature_space
from data_augmentation.VAE.evaluation.generate_images import (
    generate_from_normal,
    generate_from_test,
)
from data_augmentation.VAE.evaluation.interpolation import (
    interpolate_two_ciffers,
    visualize_grid_interpolation,
)

setup_objects = eval_setup(use_cuda=False)

visualize_feature_space(*setup_objects)
visualize_grid_interpolation(*setup_objects)
interpolate_two_ciffers(*setup_objects)
generate_from_normal(*setup_objects)
generate_from_test(*setup_objects)
plot_single_distribution(*setup_objects)
