# MJFL-Simulation
An implementation of some FL methods

1. First, run `dataprocessing.py` to get the 
   data of Group-A and Group-B
   
2. Run `utils/gen_client_time.py` to simulate the capabilities of all devices.

   
3. Edit `config.json`, `isIID` is set to 1 to experiment on IID data, and to 0 is to experiment on non-IID; 
   `group` is `A` or `B`; Available scheduling methods for `scheduler`  include Bayesian, DRL, Common (is Random in our article), Fedcs, Greedy, Genetic.

4. Run `simu_main.py` to get the selected devices and time used in per round of each optimization method.

5. Pass in the job number at the end of command line and specify the gpu that assigned to the job. \
   Run `python -m paddle.distributed.launch --gpus=0 main.py 0`  to get the accuracy of the first job. \
   Run `python -m paddle.distributed.launch --gpus=1 main.py 1` to get the accuracy of the second job. \
   Run `python -m paddle.distributed.launch --gpus=2 main.py 2` to get the accuracy of the third job. 

**Note: The deep reinforcement learning scheduling algorithm needs to run `Pretrain_RL.py` before implement step 4.**