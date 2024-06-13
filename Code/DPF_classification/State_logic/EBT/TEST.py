import numpy as np
import yaml
from state_logic_EBT_Prod import soot_burn_quantification,soot_burn_quantification_R2

# Read configuration file
Config_path = "D:/Work/Timeseries_models/Code/DPF_classification/State_logic/EBT/config_EBT.yml"
with open(Config_path, 'r') as file:
    Config = yaml.safe_load(file)


array = np.random.rand(50)
array1 = np.random.rand(50)

Q,A = soot_burn_quantification(12*array,1,Config['Thresholds'])

#Q,A = soot_burn_quantification_R2(12*array,12*array1,1,Config['Thresholds'])
print('Quality', Q)
print('Alert', A)