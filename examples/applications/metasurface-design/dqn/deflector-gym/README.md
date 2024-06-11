# deflector-gym

* `ReticoloEnv*` requires MATLAB, but you can use deflector-gym with `meent`
* clone the repo and `pip install .`

## Example

```python
import deflector_gym
env = deflector_gym.make('MeentIndex-v0')
obs = env.reset()
for step in range(10):
  env.step(env.action_space.sample())
```

## Test
In project root directory, run
```shell
pytest
```
TODO: 
* maybe rename as optics-gym?
* gym's API changes too often, how to deal with that?
