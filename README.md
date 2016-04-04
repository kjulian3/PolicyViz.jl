# PolicyViz

This Julia package was created to visualize policies created for ACAS Xu. There are two distinct types of policies that can be visualized: policies regressed from a known ACAS Xu policy, and policies created through deep reinforcement learning. For this reason, there are two separate functions to visualize each policy type. An example IJulia notebook is given in the examples folder.

Requirements:
* GridInterpolations
* Interact
* PGFPlots
* Colors
* ColorBrewer

To Install:
* Open Julia
* Execute "Pkg.clone("https://github.com/kjulian3/PolicyViz.jl.git")"
