# IT3105 2023
Course: https://www.idi.ntnu.no/emner/it3105/



## Usage
You can initiate training by executing: </br>`python3 ./rl-agent/main.py`

To run TOPP, you must provide an uuid: </br>`python3 ./rl-agent/main.py {uuid}`

The config will be saved to file, same with the model. See the config file to adjust model/system parameters. 

### Params
`--topp`: Execute topp, but requires that you provide a session uuid.

`--render`: Will render/print the games to console.


# Dependencies
Python: 3.11

PyTorch: 2.1.0.dev20230308+cu118