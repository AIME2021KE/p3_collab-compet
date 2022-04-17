# Project environment details 
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Introduction to my Udacity Collaboration and Competition Tennis Project
This readme describes what you need to run my solution in ADDITION / SUPPLIMENTAL to the basic Udacity 3nd Project for the Reinforcement Learning class Collaboration and Competition Tennis project P3_collab-compet original readme information.

Briefly the project uses the Unity (and MS Visual Studios) pre-defined environment (Tennis.exe), which is two agents controlling rackets and keep the ball in play. 

The original readme.md details of this project is contained in this directory in the MoreProjectDetails.md file, which I renamed to avoid conflict/confusion. 


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Brief description of setting up the environment
This development was performed in Windows 64bit environment, so if you have a different computer environment you may need (slightly) different instructions, particularly with regards to the Unity Tennis_Windows_x86.zip file and setting the env as the Tennis application (Tennis.exe for Windows).

If you have already set up this Unity environment for the p1_navigation or p2_continuous-control project, you can skip most of the one-time setup of the conda environment, MS Visual Studios, etc below and only focus on installation of the Tennis_Windows_x86_64.zip file, as most of this section only needs to be done once.

Although details of setting up this environment can be found in the MoreProjectDetails.md (Dependencies section), briefly it involves:

1) downloading the Tennis_Windows_x86_64.zip file containing the self-contained unity environment 
2) put the resulting directory in the p3_collab-compet folder; we further placed the Tennis.exe file in the p3_collab-compet top folder
3) we also followed the Udacity README.md file concerning the setup of the (CONDA) environment for the dqn alone (one-time only):
	a) conda create --name drlnd python=3.6 
	b) activate drlnd
    c) (possible step) create the dnrln kernel for jupyter notebook: python -m ipykernel install --user --name drlnd --display-name "drnld"
	c) use the drlnd kernel in the Jupyter notebook when running this project
    d) we used the file environment.yml (included) to create our environment; for our purposes it was named drlnd7 (6 failed attempts:(). This meant on the powershell command line doing 
        conda create env -f environment.yml
    e) download the torch 0.4.0 wheel (.whl) file for GPU; although GPU is not needed for this particular task, it apparently is tightly coupled with the provided Tennis.exe file and as far as I can tell is the main reason for my previous failed installs of drlnd's.
4) We installed MS Visual Studios 2017 & 2022 (one-time only). We did not find the "Build Tools for Visual Studio 2019" on the link provided (https://visualstudio.microsoft.com/downloads/) as indicated in the provided instructions, but rather mostly VS 2022 (VS_Community.exe) and some other things. We selected Python and Unity aspects of the download to hopefully cover our bases there and that seemed to work.
5) Clone the repository locally (if you haven't already; I had) and pip install the python requirements (last line, c)) (one-time only):
	a)git clone https://github.com/udacity/deep-reinforcement-learning.git
	b) cd deep-reinforcement-learning/python
	c) pip install .
6) pip install unityagents
	a) which may require downloading the unity app for personal individual use (one-time only): https://store.unity.com/front-page?check_logged_in=1#plans-individual

We have provided the Tennis.exe, Tennis_Windows_x86_64 and the python directory within the repository for convenience as well environment.yml for creation of the environment as well (I wasn't able to get this working on my local machine but was much closer than by 6 other attempts).

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

# My model description

I originally tried to base my model after the Udacity MADDPG (Multi-Agent Deep Deterministic Policy Gradient) lessons in the multi-agent actor-critic section for agent collaboration-competition, specifically the maddpg miniproject, however, I had issues with getting this to run with more than 3 non-zero rallies, so after looking on the web a several approaches, I decided to abandon the MADDPG framework (which was mainly for a custom environment setup anyways) and work with a list of 2 of the DDPG Agents from the P2 / bipedal examples (I found the P2 DDPG Agent to be most useful).

After many failed attempts, I reviewed the below solutions, which were quite simple but with large networks, I decided to simplify things (but leave the hooks in to show what I tried) and my approach is slightly different than either of these approaches but they were very helpful in focusing my next steps.

https://github.com/Nathan1123/P3_Collaboration_and_Competition

https://github.com/ainvyu/p3-collab-compet

https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
by Mike Richardson
with the note at the top of the python files:
""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""


## APPROACH
--> UPDATE: I started with the default (intial) values for the previous DDPG. 

The two included python files are ddpg_model.py, which contains the DDPG implementation (only seen by the agent) from DDPG, and ddpg_agent.py, which contains the DDPGAgent class. I separated out the supporting ReplayBuffer class to store the experiences in tuples for use by the DDPG and the OUNoise class into their own .py files.

I create two separate instances of the ddpg_agents and loop through them in the processing, looking for max of their scores

The NN is composed of 3 fully connected layers with two NN internal sizes (defaults: fc1_size=***40*** and fc2_size=***20***) using RELU activation functions along with an initial state_size and a final action_size to map into the Tennis input (state) and output (action) environment via a tanh function for the actor and the output of the final layer for the critic. 

The DDPG has an __init__ function to be invoked on class creation and the forward method using the NN's to convert the current state into an action.

Here we used the same structure we had for the DDPG for the pendulum problem, but based on the paper comments we reduced the size of the layers to 20 and 40, respectively, after some suggestions from corporate folks who had done it previously.

Based on suggestions from outside sources (corporate) we tried adding noise to both the weights in the network and in the states as well as to the action. We also tried batch normalization, dropouts, etc but none of these seemed effective in training at  the time that I looked at it.

The biggest breakthrough aspect of our training was increasing the BUFFERSIZE from 16 to 128 (we also tried 64 and 256 buffer sizes  but 128 seemed to work best with our searched for values). Although early on (with the MADDPG framework) increasing the buffersize didn't have much of any effect, with our simplier network implementation all the larger sizes seemed to keep the learning going much longer, albiet in the other two cases the 100 sample moving average was more about 0.1 in about 4000-8000 episodes instead of the 0.5 we required

We originally started with a higher gamma 0.9 and the previous values of the learning rates for the actor (1e-4) and critic (1e-3) but settled on the provided values in our final solution after our extensive earlier searches (but with the smaller buffersize). We searched on pretty much everything we could think of, including a number of approaches suggested by both mentors (increasing the hidden layer sizes, which was contrary to the corporate suggestion to keep them small (20 and 20 was big enough)), ranging from 400,300 to 20,20. Similarly we examined 6 separate values of lra from 1e-3 to 1e-5 and 7 values of lrc from 2e-3 to 1e-5, although the search wasn't exhaustive. Values of gamma were explored from 0.8 to 0.99 and 5 values of tau from 1e-1 to 1e-3

We also examined a number of suggestions from my corporate guide (use parameter noise not action noise -- didn't seem to work at the time), and I looked briefly at adding state noise and dropout layers as well but eventually removed them as again they didn't seem to work and the prior examples I looked at didn't do that to get solutions, so I abandoned the effort.

Because we were (potentially) adding noise to the actions, we needed to clip them before passing them out when that was the case.

Once I settled on the larger batch size (but with 20,20 hidden layer sizes), the results were promising but not sufficient so I started looking into increasing the hidden layer size and the 20,40 looked promising enough that I first set the episodes to exit after hitting the 0.5 mark, which it did shortly after 4000 iterations (8000 max) and then added a 10% buffer (0.55 before exit but print when it met the mark) and this took nearly 8000 iterations (2.5 hrs to run!).

Although normally I would like to do a little more hyperparameter searches to try to optimize, I ran out of time and with a solution (>0.5) I submitted it as is.


### the final agent solution used in the DDPG mini project decreased the LR_CRITIC by a factor of 10, among other things
Buffer size: 100,000
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.8             # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

However you can see the output of many of the more fixed hyperparameters (not passed via argument list) in the Tennis.ipynb file.

The Agent class itself is composed of an __init__ fuction for construction, which creates the two actor-critic networks, one that is local and one that is the target network, along with the optimizer and memory buffer classes (separate file (package)) from the ReplayBuffer class to store experiences and a noise source (Orstein-Uhlenbeck, from the original DDPG mini-project) class as well. 

The Agent step method adds the current experience into the memory buffer for each agent, and stores the experience into memory and exectutes the learn fucntion once there are enough experiences based on BUFFERSIZE.

The Agent act method returns actions for a given state given current policy. When it is not learning (no_grad()) it retrieves the actions for each actor from the local (actor) network and the current states. It does this by evaluating (eval) the local actor network, get new actions from the local network, train the local network if applicable, and finally select actions with noise.

The Agent learn method was the one for which we had to provide the appropriate solutions previously with the DDPG P2 project, but with two instances in a list and one memory location. Here we unpack the tuple experiences into states, actions, rewards, next_states, and dones. The next_states are used in the target (NOT local) qnetwork to get the next target actions. These are then detached from the resulting tensor to make a true copy, access Qtable for the next action, and hence the rewards of the target network. We then get the next action results from the local network and then determine the MSE loss between the target and local network fits. We then zero_grad the optimizer, propagate the loss backwards through the network, and perform a step in the optimizer. Finally a soft update is performed on the target network, using TAU times the local network parameters and (1-TAU) times the target network parameters to update the target network parameters.

As indicated the original DDPG, the agent has a helper class ReplayBuffer, with methods add, to add experiences to the buffer, and sample, to sample experiences from the buffer, and is used extensively in the step method for the Agent class.

Like our previous project, we would have liked to potentially explore some of the post-DDPG example approaches, such as MADDPG (Multi-Agent DDPG) and the N-step. However since these were mainly modifications of the internal workings of the agents and the like, we felt that it was best to first get the baseline DDPG running and then see if there are problems about possibly making these modifications. 

So we start with our original agent and model, which we've imported locally and import the (slightly modified) DDPG function for the unity setup, and this was found to be sufficient for this exercise.

# Running the model

To run the code, download my entire repository onto your local drive, which includes the Tennis .zip Unity file that you'll want to unpack locally. You will probably want to make sure you have a recent version of MS Visual Studios (2017 to 2022 seemed to be OK) and use your Anaconda powershell to create the drlnd anaconda environment, if you haven't already from the first project. In Anaconda,  click on the "Applications on " pull-down menu and select your newly created drlnd environment (drlnd) and once that loads then launch the Jupyter notebook from that particular environment. 

Once in the notebook, you'll want to go the the kernel and at the bottom change your kernel from python to your newly created drlnd. At this point you are ready to run the notebook.

At this point I usually select restart and run all to make sure all the cells will run without interruptions. 

As indicated above I wasn't able to run this locally so had to run it on the Udacity site. 

A few additional notes that caused some confusion and delay on my part
1) I found it safe / best to always restart and clear output EACH time you try to run. I often got weird errors that didn't make any sense that went away as soon as I cleared output
2) You have to click on the Unity window that opens up (or possibly minimize it); otherwise you'll get a frustrating timeout that doesn't make any sense.  
3) I also found that the kernel seemed to no longer be avaiable after a single run, so I switched between the GPU environment and the CPU environment on the Udacity site after each run.  


Techically speaking we made our requirement at 4520 with learning score of 0.5019, however in these cases I typically don't break out until I've exceeded the requirement by some small amount (in this case 10% = 0.55); this however resulted in taking 7910 episodes to reach a score of 0.5430


## FUTURE IMPROVEMENTS

As mentioned above we believe probably the greatest improvement will come from an N-step implementation:
1) N-step sampling 
Since this could provide information on longer-term solutions earlier

Another would be to do at least a much greater sensitivity exploration of the hyperparameters after our BATCHSIZE increase to 128, not only of the hyperparameters that were passed via argument list:

LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
TAU = 1e-2              # for soft update of target parameters
GAMMA = 0.8            # discount factor

But a number of the more "hardcoded" values:

DDPG Agent.init, UPDATE_EVERY: 10
DDPG Agent.init, UPDATES_PER_STEP: 10
DDPG Agent.init, LEARN_START: 0
DDPG Agent.init, NOISE_MU: 0.0
DDPG Agent.init, NOISE_THETA: 0.15
DDPG Agent.init, NOISE_SIGMA: 0.1

as well as a number of the various other techniques we tried before that didn't work initially:
1) adding noise to the weights in the network INSTEAD of the action
2) OR adding noise to the states INSTEAD of the action. 
3) batch normalization
4) dropouts.

neither of which we had time at the end to fully expore.


# REFERENCES
##### Basic

Github example of continous control with DDPG
https://github.com/MariannaJan/continous_control_DDPG

Article: "Continuous Deep Q-Learning with Model-based Acceleration", by Gu et al., 2016
https://arxiv.org/pdf/1603.00748.pdf

Article: "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING" by Lillencrap et al, 2019
https://arxiv.org/pdf/1509.02971.pdf

Article: "Benchmarking Deep Reinforcement Learning for Continuous Control" by Duan et al, 2016
https://arxiv.org/pdf/1604.06778.pdf


"Let’s make a DQN: Double Learning and Prioritized Experience Replay", 2016
https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/


Github RL Lab by by ShangtonZhang
https://github.com/rll/rllab

"DDPG (Actor-Critic) Reinforcement Learning using PyTorch and Unity ML-Agents", solution to 
"how to handle mutliple agents in ddpg pytorch" google search
https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents

Examples:
https://github.com/Nathan1123/P3_Collaboration_and_Competition
https://github.com/ainvyu/p3-collab-compet
 

##### D4PG
Article: "DISTRIBUTED DISTRIBUTIONAL DETERMINISTIC POLICY GRADIENTS",by Barth-Maron, 2018
https://openreview.net/pdf?id=SyZipzbCb

##### General knowledge of various topics in no particular order (as opened I think)
https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829

https://www.google.com/search?q=how+to+add+noise+to+hidden+weights+in+neural+network&sxsrf=APq-WBsz6Lkp4NEm56rbGqCPlcJUrUu3Bg%3A1649895248160&ei=UGdXYtW0CdDTkPIPvrylqAE&oq=how+to+add+noise+to+hidden+weights+&gs_lcp=Cgdnd3Mtd2l6EAEYAjIFCCEQoAEyBQghEKABMgUIIRCgAUoECEEYAEoECEYYAFAAWOVcYNJ9aABwAHgAgAGZAYgB4R6SAQUxMi4yNJgBAKABAcABAQ&sclient=gws-wiz#kpvalbx=_XGhXYraXAdmfkPIP79uQ0AQ3

https://stackoverflow.com/questions/59013993/how-can-i-add-bias-using-pytorch-to-a-neural-network

https://www.codetd.com/en/article/11834682

https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

https://discuss.pytorch.org/t/add-gaussian-noise-to-parameters-while-training/109260

https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/3

https://discuss.pytorch.org/t/how-to-install-pytorch-0-4-0-with-cuda-9-0/48914

https://coder.social/udacity/deep-reinforcement-learning

https://www.bing.com/search?q=pytorch+0.4.0+cuda&form=ANNTH1&refig=c3d77ee899304f5fabb8f3cedcf82bfe&ntref=1

https://discuss.pytorch.org/t/how-to-install-pytorch-0-4-0-with-cuda-9-0/48914

https://www.bing.com/search?q=KeyError%3A%20%22Couldn%27t%20find%20field%20FieldDescriptorProto.proto3_optional%22%20python%20unity&qs=n&form=QBRE&=%25eManage%20Your%20Search%20History%25E&sp=-1&pq=keyerror%3A%20%22couldn%27t%20find%20field%20google.protobuf.fielddescriptorproto.proto3_optional%22%20python%20unity&sc=0-97&sk=&cvid=E9ABE10449F64343A0C7058F615BDFE4&ntref=1

https://stackoverflow.com/questions/71626781/anaconda-python-expainerdashboard-keyerror-couldnt-find-field-google-pr

https://stackoverflow.com/questions/47257751/keyerror-couldnt-find-field-google-protobuf-descriptorproto-extensionrange-op

https://www.bing.com/search?q=how%20to%20install%20kernel%20in%20jupyter%20for%20specific%20anaconda%20environment&qs=n&form=QBRE&=%25eManage%20Your%20Search%20History%25E&sp=-1&pq=how%20to%20install%20kernel%20in%20jupyter%20for%20specific%20anaconda%20env&sc=0-58&sk=&cvid=8D1EBF16895D4FAC88936EF9826A2315&ntref=1

https://gdcoder.com/how-to-create-and-add-a-conda-environment-as-jupyter-kernel/

https://www.bing.com/search?q=how+to+get+a+directory+listing+in+python&form=ANNTH1&refig=7cad7cc5de9d4a9194816ef7db907e47&sp=3&qs=UT&pq=how+to+get+a+directory+listing+&sk=PRES1UT2&sc=8-31&cvid=7cad7cc5de9d4a9194816ef7db907e47

https://www.bing.com/search?q=gaierror+python+3&form=ANNTH1&refig=7fa870dbe59b43ac9c3f41d4079b6192&sp=1&qs=MT&pq=gaierror+&sk=PRES1&sc=8-9&cvid=7fa870dbe59b43ac9c3f41d4079b6192&ntref=1

https://www.tecmint.com/resolve-temporary-failure-in-name-resolution/

https://stackoverflow.com/questions/40238610/what-is-the-meaning-of-gaierror-errno-3-temporary-failure-in-name-resolutio

https://www.bing.com/search?q=python%20NewConnectionError%3A%20%3Curllib3.connection.HTTPConnection%20object%20at%200x0000022F86B32160%3E%3A%20Failed%20to%20establish%20a%20new%20connection%3A%20%5BErrno%2011001%5D%20getaddrinfo%20failed&qs=n&form=QBRE&=%25eManage%20Your%20Search%20History%25E&sp=-1&pq=newconnectionerror%3A%20%3Curllib3.connection.httpconnection%20object%20at%200x0000022f86b32160%3E%3A%20failed%20to%20establish%20a%20new%20connection%3A%20%5Berrno%2011001%5D%20getaddrinfo%20failed&sc=0-156&sk=&cvid=ACDA2B21B37E4AF3B44B2A8AA3F0365F&ntref=1

https://itecnote.com/tecnote/python-httpconnectionpool-failed-to-establish-a-new-connection-errno-11004-getaddrinfo-failed/

https://github.com/openai?msclkid=7f9ebcb6b20011ecb533e3c5de0cedd9

https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained?msclkid=7f9d4f9db20011eca2d8c4fc80eeec2f

https://github.com/tristandeleu/pytorch-maml-rl

https://github.com/openai/maddpg

https://github.com/rlworkgroup/garage

https://github.com/shariqiqbal2810/MAAC

https://github.com/openai/gym/wiki/BipedalWalker-v2

https://pytorch.org/docs/stable/generated/torch.cat.html

https://stackoverflow.com/questions/47331235/how-should-openai-environments-gyms-use-env-seed0

http://download.pytorch.org/whl/cpu/torch/

https://stackoverflow.com/questions/19283271/how-to-uninstall-requests-2-0-0

https://stackoverflow.com/questions/38411942/anaconda-conda-install-a-specific-package-version

https://pypi.org/project/mxnet/

https://discuss.pytorch.org/t/tensorboard-with-pytorch-summarywritter/84668

https://pytorch.org/docs/stable/generated/torch.norm.html

https://www.programcreek.com/python/example/107653/torch.nn.BatchNorm1d

https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html

https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-ML-Agents.md#training-using-concurrent-unity-instances

https://realpython.com/python-or-operator/

https://pytorch.org/get-started/previous-versions/

https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#download-cuda-software

https://developer.nvidia.com/cuda-toolkit

https://developer.nvidia.com/search?page=3&sort=relevance&term=using%20cuda%209.0%20python%20on%20windows%2011

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

https://www.google.com/search?q=conda+create+environment+from+yml&sxsrf=APq-WBtkxuAaqkH2fFvJeTBBEKl-6HLc1w%3A1649792235975&ei=69RVYumWO62dkPIP3_6RoA4&oq=conda+env+create+-f&gs_lcp=Cgdnd3Mtd2l6EAEYATIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwAzIHCAAQRxCwA0oECEEYAEoECEYYAFAAWABgnhpoAXABeACAAQCIAQCSAQCYAQDIAQjAAQE&sclient=gws-wiz

https://docs.conda.io/projects/conda-build/en/latest/user-guide/wheel-files.html

https://docs.python.org/3/library/copy.html#module-copy

https://www.geeksforgeeks.org/check-multiple-conditions-in-if-statement-python/

https://stackoverflow.com/questions/52288635/how-do-i-use-torch-stack

https://medium.com/adding-noise-to-network-weights-in-tensorflow/adding-noise-to-network-weights-in-tensorflow-fddc82e851cb

https://discuss.pytorch.org/t/how-to-change-the-weights-of-a-pytorch-model/41279

https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/

https://www.codetd.com/en/article/11834682

https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/

https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/9

https://arxiv.org/pdf/1805.08000.pdf

https://stackoverflow.com/questions/59013993/how-can-i-add-bias-using-pytorch-to-a-neural-network

https://discuss.pytorch.org/t/add-gaussian-noise-to-parameters-while-training/109260

https://github.com/christianversloot/machine-learning-articles/blob/main/using-dropout-with-pytorch.md

https://stackoverflow.com/questions/9413367/most-efficent-way-of-finding-the-minimum-float-in-a-python-list
