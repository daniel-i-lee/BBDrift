# OpenDrift module for bluebottle drift.

import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import OceanDrift
from opendrift.elements.passivetracer import PassiveTracer

# We add bluebottle properties to the existing element class PassiveTracer.
PassiveTracer.variables = PassiveTracer.add_variables([
    ('Sail_height', {'dtype': np.float32,
                           'units': 'm',
                     'description': 'Height of the sail on top of float',
                          'default': 0}),
    ('Sail_chord', {'dtype': np.float32,
                           'units': 'm',
                     'description': 'Chord or length of sail',
                          'default': 0.034}),
    ('Sail_width', {'dtype': np.float32,
                           'units': 'm',
                     'description': 'Width of sail, perpendicular to the chord',
                          'default': 0.016}),
    ('Body_height', {'dtype': np.float32,
                           'units': 'm',
                     'description': 'Height of the submerged body',
                          'default': 0.014}),
    ('Body_width', {'dtype': np.float32,
                       'units': 'm',
                 'description': 'Width of the submerged body',
                      'default': 0.027}),
    ('Camber', {'dtype': np.float32,
                           'units': '1',
                'description': 'Ratio that indicates the bending of the sail',
                           'default': 0.01}),
    ('Orientation', {'dtype': np.float32,
                         'units': '1',
                     'description': '1 for right-handed (left-sailing). -1 for left-handed (right sailing)',
                         'default': 1}),
    ('beta_a', {'dtype': np.float32,
                           'units': 'degrees',
                'description': 'Angle of the Bluebottle sail relative to the wind. Always keep positive',
                           'default': 40}),
    ('C_H', {'dtype': np.float32,
                           'units': '1',
             'description': 'Hydrodynamic force coefficient',
                           'default': 1}),
    ('C_Ax', {'dtype': np.float32,
                           'units': '1',
              'description': 'Component of the aerodynamic force coefficient on the axis parallel to the sail',
                           'default': 0.1}),
    ('current_drift_factor', {'dtype': np.float32,
                           'units': '1',
              'description': 'Current drift factor',
                           'default': 1})
    ])

class BBDrift(OceanDrift):
    """Trajectory model based on the OceanDrift framework.
    """

    ElementType = PassiveTracer

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        #'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        #'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        }


    def __init__(self, *args, **kwargs):

        # Call parent constructor.
        super(BBDrift, self).__init__(*args, **kwargs)


    def update(self):
        """Update positions of bluebottles at each timestep."""

        # Simply move particles with ambient current.
        self.advect_ocean_current()
        
        # Calculate wind speed and direction
        Wind_speed = np.sqrt(self.environment.x_wind**2 + self.environment.y_wind**2)
        Wind_angle = np.arctan2(self.environment.y_wind,self.environment.x_wind)        

        # Densities of air and water.
        Rho_A = 1.225
        Rho_H = 1025        
        
        # Sail height and chord (length) calculation. Ratio is used if chord (length) is given but no height.
        Sail_ratio = 0.41 # Ratio of bluebottle sail height to sail chord (length).
        Sail_height_final = np.zeros(len(self.elements.Sail_height))
        for i in range(len(self.elements.Sail_height)):
            if self.elements.Sail_height[i] == 0: 
                Sail_height_final[i] = Sail_ratio * self.elements.Sail_chord[i]
            else:     
                Sail_height_final[i] = self.elements.Sail_height[i]

        # Calculating area of sail (S_y), estimating the shape as a circular segment.
        Radius = Sail_height_final/2 + self.elements.Sail_chord**2/(8*Sail_height_final)
        Theta = 2*np.arcsin(self.elements.Sail_chord/(2*Radius))
        S_y = Radius**2 * (Theta - np.sin(Theta))/2                    

        # Calculating side area of sail (S_x), estimated the shape as a triangle.
        S_x = Sail_height_final * self.elements.Sail_width /2

        # Calculating area of submerged body, estimating the shape as a rectangle.
        S_H = self.elements.Body_width * self.elements.Body_height   
        
        # Change angle of attack to negative if bluebottle is left-handed.
        beta_a_signed = self.elements.beta_a*self.elements.Orientation
        
        # Calculating sail aspect ratio.
        Aspect_ratio = Sail_height_final**2 / S_y
        
        # Calculating aerodynamic force coefficient on axis perpendicular to bluebottle sail.
        C_Ay = np.pi*Aspect_ratio/2*np.radians(self.elements.beta_a) + 4*np.pi*Aspect_ratio/3*self.elements.Camber

        # Shape parameter (Lambda)
        Lambda = np.sqrt(Rho_A*np.sqrt((S_y*C_Ay)**2 + (S_x*self.elements.C_Ax)**2)/(Rho_H*S_H*self.elements.C_H))
        
        # Obtuse solution required for beta
        beta = np.abs(np.pi - np.arctan(S_y*C_Ay/(S_x*self.elements.C_Ax)))

        # Alpha equation depends on orientation of the bluebottle.
        alpha = -self.elements.Orientation*beta - np.radians(beta_a_signed) + Wind_angle

        # Bluebottle final speed and course.
        V_bbx =  - Lambda*Wind_speed*np.cos(alpha)
        V_bby =  - Lambda*Wind_speed*np.sin(alpha)

        # Update particle positions based on final speed and course.
        self.update_positions(V_bbx, V_bby)