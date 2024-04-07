# %% 
from symbol import parameters
from reretry import retry
import yaml
import config
from pcse.models import  Wofost72_PP
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.util import WOFOST72SiteDataProvider, DummySoilDataProvider
import datetime
import json
def define_agromanagement(run_details):
    """Define the agromanagement for POTATO."""
    agro_yaml = """
    - {campaign_start_date}:
        CropCalendar:
            crop_name: potato
            variety_name: {variety_name}
            crop_start_date: {crop_start_date}
            crop_start_type: sowing
            crop_end_date: {crop_end_date}
            crop_end_type: harvest
            max_duration: 300
        TimedEvents: null
        StateEvents: null
    """.format(**run_details._asdict())
    agro = yaml.safe_load(agro_yaml)
    return agro, agro_yaml


def get_modelparameters(run_details):
    """Get parameter sets for crop, soil and site."""
    cropd = YAMLCropDataProvider(fpath=config.CROP_DATA_PATH,force_reload=True) # Standard crop parameter library
    soild = DummySoilDataProvider() # We don't need soil for potential production, so we use dummy values
    sited = WOFOST72SiteDataProvider(WAV=50)  # Some site parameters

    # Retrieve all parameters in the form of a single object. 
    # In order to see all parameters for the selected crop already, we
    # synchronise data provider cropd with the crop/variety: 
    cropd.set_active_crop(run_details.crop_name, run_details.variety_name)
    params = ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild)
    return params   

def date_to_string(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    return obj

def run_wofost_simulation(id, paramset, run_details, wdp, problem=None, local=True):
    """
    Run a WOFOST simulation.

    This function runs a WOFOST simulation with the given parameters and run details. 
    It uses the Wofost72_PP class to run the simulation, and returns the output of the 
    simulation along with the parameter set. If an error occurs during the simulation, 
    it returns an error message.

    Parameters:
    paramset (list): A list of parameters for the WOFOST model.
    run_details (dict): A dictionary containing details about the run.
    wdp (WofostData): An instance of the WofostData class containing the data for the simulation.
    problem (dict, optional): A dictionary containing problem details. Must be provided when 'local' is False.
    local (bool, optional): A flag indicating whether the simulation is local. Defaults to True.

    Returns:
    tuple: A tuple containing the output of the simulation and the parameter set. If an error occurs, 
    a string with the error message is returned.
    """
    # print(f"Running simulation {id} with paramset {paramset}")
    # print(f"Run details: {run_details}")

    try:
        agro, _ = define_agromanagement(run_details)
        params = get_modelparameters(run_details)
        params.clear_override()
        if local:
            params.set_override(paramset[0], float(paramset[1]))
        else:
            if problem is None:
                raise ValueError("The 'problem' argument must be provided when 'local' is False.")
            for name, value in zip(problem["names"], paramset):
                params.set_override(name, value)

        wofost = Wofost72_PP(params, wdp, agro)
        wofost.run_till_terminate()
        output = wofost.get_output()
        # Handle output within the function
        if local:
            filename = f'{config.p_out_LSAsims}/{id}_{paramset[0]}.json'
            paramset_str = paramset[1]

        else: 
            filename = f'{config.p_out_sims}/{id}.json'
            paramset = [round(i, 5) for i in paramset]
            paramset_str = '_'.join(map(str, paramset))       # Convert paramset to a string
      
        with open(filename, 'w') as file:
            json.dump(output, file, default=date_to_string)
        # Delete unnecessary variables
        # del params, wofost, output
        return paramset_str
    except Exception as e:
        return f"An error occurred: {e}"

    


