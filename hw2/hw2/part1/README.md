<!-- ## Part1 
Since cyvlfeat only support python version below
python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|3.4.*']
Create new conda env to support

### Conda env commands
#### List env
> conda info --envs

or
> conda env list
#### Create conda env
> conda create --name hw2_part1 python=3.6
#### Activate env
> conda activate hw2_part1
#### Deactivate env
> conda deactivate
#### Remove env
> conda env remove -n ENV_NAME -->

## Part1
### Run tiny feature
> python3 p1.py --feature tiny_image --classifier nearest_neighbor