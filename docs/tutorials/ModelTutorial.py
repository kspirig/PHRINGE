import astropy.units as u
import matplotlib.pyplot as plt

from phringe.core.instrument import Instrument
from phringe.core.observation import Observation
from phringe.core.perturbations.power_law_psd_perturbation import PowerLawPSDPerturbation
from phringe.core.scene import Scene
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell
from phringe.main import PHRINGE
from phringe.util.baseline import OptimalNullingBaseline


phringe = PHRINGE(gpu_index=5, seed=42, grid_size=60)


"""
Either x_position and y_position (both float, in radians) or semi_major_axis (float, in meters), eccentricity
            (float), inclination (float, in radians), raan (float, in radians), argument_of_periapsis (float, in radians),
            true_anomaly (float, in radians), host_star_distance (float, in meters), host_star_mass (float, in kg) and planet_mass
            (float, in kg).
            """
from astropy.constants import au, M_sun, M_earth
print("Start")

scene = Scene()
phringe.set(scene)

earth_twin = Planet(
    name='Earth-Twin',
    has_orbital_motion=True,  # Whether the planet is propagated in time along its orbit
    mass=1 * u.Mearth,
    radius=1 * u.Rearth,
    temperature=254 * u.K,
    semi_major_axis=1 * u.au,
    eccentricity=0,
    inclination=0 * u.deg,
    raan=0 * u.deg,
    argument_of_periapsis=135 * u.deg,
    true_anomaly=0 * u.deg,
    input_spectrum=None,
    # host_star_distance=10 * u.pc,  # Is only required if no star is added explicitly to the scene
    # host_star_mass=1 * u.Msun,  # Is only required if no star is added explicitly to the scene
)


scene.add_source(earth_twin)

earth_twin_spectrum = phringe.get_source_spectrum('Earth-Twin').cpu().numpy()


model_counts = phringe.get_model_counts(
    kernels=True,
    spectral_energy_distribution=earth_twin_spectrum,
    # x_position=3e-7,  # in radians
    # y_position=0e-7, # in radians
    semi_major_axis=au.value,
    eccentricity=0,
    inclination=0 ,
    raan=0,
    argument_of_periapsis=135,
    true_anomaly=0 ,
    host_star_distance=10 ,
    host_star_mass = M_sun.value,
    planet_mass= M_earth.value,
    # radius=1 * u.Rearth,
    # temperature=254 * u.K,
    # input_spectrum=None,
)

plt.imshow(model_counts[0], cmap='Greys')
plt.title('Model Counts for Earth-Twin')
plt.ylabel('Wavelength Channel')
plt.xlabel('Time Step')
plt.colorbar()
plt.show()

# Or at one wavelength
plt.plot(model_counts[0, 20])
plt.title('Model Counts for Earth-Twin at Wavelength Bin 20')
plt.ylabel('Counts')
plt.xlabel('Time Step')
plt.show()