import h5py
import numpy as np
from scipy.interpolate import griddata
import os

class HecRasHDF5Reader:
    """Utility class to read 2D flow results from HEC-RAS HDF5 output files."""
    def __init__(self, hdf_path):
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(f"HEC-RAS HDF5 file not found: {hdf_path}")
        self.hdf_path = hdf_path
        print("Warning: Ensure HEC-RAS HDF5 CRS matches target grid CRS.")

    def get_2d_flow_area_name(self):
        with h5py.File(self.hdf_path, 'r') as f:
            try:
                base_path = '/Geometry/2D Flow Areas/'
                keys = list(f[base_path].keys())
                if len(keys) > 0:
                    return keys[0]
                else:
                    raise ValueError("No 2D Flow Areas found in Geometry.")
            except KeyError:
                raise ValueError("Invalid HEC-RAS HDF5 structure: Geometry/2D Flow Areas not found.")

    def get_mesh_coordinates(self, area_name=None):
        if area_name is None:
            area_name = self.get_2d_flow_area_name()
        with h5py.File(self.hdf_path, 'r') as f:
            coords_path = f'/Geometry/2D Flow Areas/{area_name}/Cells Center Coordinate'
            if coords_path in f:
                data = f[coords_path][:]
                return data[:, 0], data[:, 1]
            else:
                raise KeyError(f"Coordinates not found at {coords_path}")

    def get_time_series(self, variable, area_name=None):
        if area_name is None:
            area_name = self.get_2d_flow_area_name()
        with h5py.File(self.hdf_path, 'r') as f:
            base_path = f'/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{area_name}/{variable}'
            if base_path in f:
                return f[base_path][:]
            else:
                raise KeyError(f"Variable {variable} not found.")

    def get_grid_data(self, variable, time_step, target_grid_x, target_grid_y, method='linear', fill_value=np.nan, wet_dry_thresh=0.01):
        src_x, src_y = self.get_mesh_coordinates()
        points = np.column_stack((src_x, src_y))
        try:
            data_series = self.get_time_series(variable)
            if time_step >= data_series.shape[0]:
                raise IndexError(f"Time step {time_step} out of bounds.")
            values = data_series[time_step, :]
        except KeyError:
            if 'Velocity' in variable:
                print(f"Warning: {variable} not found. Returning filled array.")
            return np.full_like(target_grid_x, fill_value)

        grid_z = griddata(points, values, (target_grid_x, target_grid_y), method=method, fill_value=fill_value)
        if variable == 'Depth':
            with np.errstate(invalid='ignore'):
                 mask = grid_z < wet_dry_thresh
            grid_z[mask] = 0.0
        return grid_z
