import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import ChronicsHandler    
from grid2op.Action import BaseAction
from grid2op.VoltageControler import VCFromFileAgentOverrides
from TD3_torch_TFE import set_seed

set_seed(1)
env_name = "l2rpn_2019"

env = grid2op.make(
    env_name,
    backend=PandaPowerBackend(),
    action_class=BaseAction,
    chronics_handler=ChronicsHandler(),
    voltagecontroler_class=VCFromFileAgentOverrides,
   )
N_max_chronics = len(env.chronics_handler.subpaths) 

#Cette commande crée les env de training, validation et test ==> besoin une seule fois au début

nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random( # type: ignore
    pct_val=0.5, 
    pct_test=0.5, 
    add_for_val="val", 
    add_for_test="test"
)