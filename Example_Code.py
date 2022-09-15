from BBDrift import BBDrift
import numpy as np
from datetime import timedelta

o = BBDrift(loglevel=20)
o2 = BBDrift(loglevel=20)
o3 = BBDrift(loglevel=20)

# Adding readers
from opendrift.readers import reader_netCDF_CF_generic
# Arome atmospheric model
reader_arome = reader_netCDF_CF_generic.Reader(o.test_data_folder() + '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc')
# Norkyst ocean model
reader_norkyst = reader_netCDF_CF_generic.Reader(o.test_data_folder() + '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc')
o.add_reader([reader_norkyst, reader_arome])
o2.add_reader([reader_norkyst, reader_arome])
o3.add_reader([reader_norkyst, reader_arome])
# o3 with different parameters

# Define start time of seeding
start_time = reader_arome.start_time

# Seeding right-handed BBs
# Default value of Orientation parameter is 1, which corresponds to a right-handed BB
o.seed_elements(lon=4.4, lat=59.9, number=25, time=[start_time, start_time + timedelta(hours=5)])

# Seeding left-handed BBs
# Note that the new Orientation value of -1 (corresponding to left-handed BB) is given for each particle by providing an array
o2.seed_elements(lon=4.4, lat=59.9, number=25, time=[start_time, start_time + timedelta(hours=5)], Orientation=-np.ones(25))

# Seeding right-handed BBs with increased sail height and sail width
# Once again, an array is used to change parameter values
o3.seed_elements(lon=4.4, lat=59.9, number=25, time=[start_time, start_time + timedelta(hours=5)], 
                 Sail_height=0.021*np.ones(25), Sail_width=0.024*np.ones(25))

# Define end time of simulation
end_time = reader_arome.end_time

o.run(end_time=end_time, time_step=1800, time_step_output=3600)
o2.run(end_time=end_time, time_step=1800, time_step_output=3600)
o3.run(end_time=end_time, time_step=1800, time_step_output=3600)

o.plot(fast=True, compare=[o2,o3], legend=['Right-handed','Left-handed','Larger sail'])