def step_reward(state, ref_value, episode_ended:bool= False):
    Cl= state[-2]
    Cd= state[-1]

    current_value = Cl/Cd

    delta_value = current_value - ref_value

    if delta_value>0.01:
        reward = 0.5
    else:
        reward = -0.1
    return reward


def best_value_reward(state, ref_value, episode_ended:bool= False):
    Cl = state[-2]
    Cd = state[-1]

    current_value = Cl / Cd

    delta_value = current_value - ref_value

    reward = round(100*delta_value, 2)

    return reward


def final_start_value_reward(state, ref_value, episode_ended:bool= False):
    Cl = state[-2]
    Cd = state[-1]

    if Cd==0.:
        Cd=.00001
    current_value = Cl / Cd

    if episode_ended:
        delta_value = current_value - ref_value

        reward = round(delta_value, 2)
    else:
        reward = 0

    return reward

