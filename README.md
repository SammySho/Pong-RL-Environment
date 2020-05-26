# Pong-RL-Environment
An version of Pong created to be used with the OpenAI Gym API. There is single-agent setup and a multi-agent setup, with the option for cooperative or competitive reward schems.


There are simple and advanced mechanics in this project. The simple mechanics cause the ball to reflect off the paddle in the same way as refracting light, where the angle of incidence is = the angle of reflection. The advanced mechanics allow for the angle and speed of a return to be increased or reduced dependent on the location of contact of the ball on the paddle.

The multi-agent file requires actions to be passed to the step method as step(action_1, action_2). It returns observations in the form:

state_1, state_2, reward_1, reward_2, number_of_hits, done, {info}

Number of hits is an array containing agent_1_hits and agent_2_hits.

The single-agent file requires a single action be passed to the step method as step(action). It returns an observation in the following form:

state, reward, number_of_hits, done, {info}

The environment was created to be modular and as such the dimensions of the environment are very easily modifiable. Thses are mostly set in the constructor for the environment.
