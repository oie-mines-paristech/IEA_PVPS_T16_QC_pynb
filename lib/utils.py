import os, sys
import pandas as pd
from libinsitu.common import nc2df
from datetime import datetime, timedelta
import sg2
import numpy as np
import json


URL = 'http://tds.webservice-energy.org/thredds/dodsC/surfrad-stations/SURFRAD-BON.nc'
TMP_FILE="./tmpData/data.parquet"
META_FILE="./tmpData/data.meta"

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
    
def save_meta(df, filename):
    # Save meta data
    with open(filename, 'w') as f:
        json.dump(df.attrs, f, cls=npEncoder)
    
def save_df(df) :
    """Save data and meta data"""
    df.to_parquet(TMP_FILE)
    
    save_meta(df, META_FILE)
    
    

def fetch_data() :
    """ Fetch 3 years of data from OpenDAP or from tmp file"""
    if os.path.exists(TMP_FILE) :
        out = pd.read_parquet(TMP_FILE)
        
        # Load meta data
        with open(META_FILE, 'r') as f:
            attrs = json.load(f)
        out.attrs = attrs
        
        return out
    
    
    start = datetime.now() - timedelta(days=365*3)

    # Load data over OpenDAP into a pandas DataFrame
    # 'url' is defined in the section above 
    data = nc2df(URL, vars=["GHI", "DHI", "BNI"], start_time=start, drop_duplicates=True)
    
    lon = float(data.attrs["Station_Longitude"])
    lat = float(data.attrs["Station_Latitude"])
    elev = float(data.attrs["Station_Elevation"])
    
    params = [lon, lat, elev]
    
    # Compute sun position 
    sun_pos = sg2.sun_position(
        [params], 
        data.index.values,
             ["geoc.EOT",
              "topoc.omega",
              "topoc.gamma_S0",
              "topoc.alpha_S",
              "topoc.delta",
              "topoc.r_alpha",
              "topoc.toa_hi",
              "topoc.toa_ni"])
    
    # Compute sunrise / sunset times
    sun_rise = sg2.sun_rise(
        [params], 
        data.index.values)
    
    
    # Combine the two
    out = pd.DataFrame({
       'GHI': data.GHI,
       'BNI': data.BNI,
       'DHI': data.DHI,
       'DELTA':np.squeeze(sun_pos.topoc.delta),
       'OMEGA':np.squeeze(sun_pos.topoc.omega),
       'EOT':np.squeeze(sun_pos.geoc.EOT),
       'THETA_Z':np.squeeze(np.pi/2-sun_pos.topoc.gamma_S0),
       'GAMMA_S0':np.squeeze(sun_pos.topoc.gamma_S0),
       'ALPHA_S':np.squeeze(sun_pos.topoc.alpha_S),
       'R':np.squeeze(sun_pos.topoc.r_alpha),
       'TOA':np.squeeze(sun_pos.topoc.toa_hi),
       'TOANI':np.squeeze(sun_pos.topoc.toa_ni),
       'SR_h':sun_rise[:,0,0],
       'SS_h':sun_rise[:,0,2],
      }, index=data.index.values)
    
    # Copy metadata
    out.attrs = data.attrs
    
    save_df(out)
    
    return out
    
    
        
    