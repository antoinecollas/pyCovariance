__all__ = [
    'Feature',
    'covariance',
    'covariance_euclidean',
    'covariance_texture',
    'location_covariance_texture_Gaussian',
    'location_covariance_texture_Tyler',
    'location_covariance_texture_RGD',
    'center_euclidean',
    'center_intensity_euclidean',
    'identity_euclidean',
    'intensity_vector_euclidean',
    'mean_vector_euclidean',
    'subspace_SCM',
    'subspace_tau_UUH',
    'subspace_tau_UUH_RGD',
    'tau_UUH',
    'tau_UUH_RGD'
]

from .base import Feature
from .covariance import covariance, covariance_euclidean
from .covariance_texture import covariance_texture
from .location_covariance_texture import\
        location_covariance_texture_Gaussian,\
        location_covariance_texture_Tyler,\
        location_covariance_texture_RGD
from .dummy import\
        center_euclidean,\
        center_intensity_euclidean,\
        identity_euclidean,\
        intensity_vector_euclidean,\
        mean_vector_euclidean
from .low_rank_models import\
        subspace_SCM,\
        subspace_tau_UUH,\
        subspace_tau_UUH_RGD,\
        tau_UUH,\
        tau_UUH_RGD
