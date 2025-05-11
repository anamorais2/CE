import numpy as np
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *
import os
import torch
import task3_3_GA_CMAES as task3_3
import neural_controller

# ---- SIMULATE BEST ROBOT ----
def simulate_best_robot(robot_structure, scenario=None, steps=500, controller = alternating_gait):
    
    connectivity = get_full_connectivity(robot_structure)
    #if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0
    
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size,t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        #if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0

        frames = []
        for t in range(200):
            actuation = controller(action_size,t)
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')
        


    
    

def simulate_best_pair(robot_structure,brain,scenario=None, steps=500, INPUT_SIZE=0, OUTPUT_SIZE=0):
    
    connectivity = get_full_connectivity(robot_structure)
    #if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    

    conn = get_full_connectivity(robot_structure)
    env = gym.make(scenario, max_episode_steps=steps,
                   body=robot_structure, connections=conn)


    state = env.reset()[0]
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    total_reward = 0.0
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    if INPUT_SIZE != input_size or OUTPUT_SIZE != output_size:
        print("Input or output size mismatch")
        return 0
        # return [-abs(INPUT_SIZE-input_size)-abs(OUTPUT_SIZE-output_size), -np.inf]
    
    for t in range(steps):  # Simulate for 200 timesteps
        with torch.no_grad():
            action = brain(torch.tensor(state, dtype=torch.float32)
                            .unsqueeze(0)).numpy().flatten()
        state, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return total_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance




def create_gif_pair(robot_structure, brain_weights, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, INPUT_SIZE=0, OUTPUT_SIZE=0):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        state = env.reset()[0]
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0

        input_size_atual = env.observation_space.shape[0]
        output_size_atual = env.action_space.shape[0]
        print(f"Input size: {input_size_atual}, Output size: {output_size_atual}")
        print(f"Expected Input size: {INPUT_SIZE}, Expected Output size: {OUTPUT_SIZE}")

        # Criar o modelo PyTorch e carregar os pesos
        brain = neural_controller.NeuralController(INPUT_SIZE, OUTPUT_SIZE)
        neural_controller.set_weights(brain, brain_weights)

        # Verificar se há descompasso nas dimensões
        if INPUT_SIZE != input_size_atual or OUTPUT_SIZE != output_size_atual:
            print("Input or output size mismatch. Adjusting weights...")
            # Adaptar os pesos do controlador
            old_weights = neural_controller.get_weights(brain)  # Obter os pesos atuais do controlador
            adapted_weights = task3_3.adapt_weights(
                old_weights=old_weights,
                old_input=INPUT_SIZE,
                old_output=OUTPUT_SIZE,
                new_input=input_size_atual,
                new_output=output_size_atual
            )
            # Atualizar os pesos do controlador
            neural_controller.set_weights(brain, adapted_weights)

        frames = []
        for t in range(steps):
            with torch.no_grad():
                action = brain(torch.tensor(state, dtype=torch.float32)
                               .unsqueeze(0)).numpy().flatten()
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')
        log_path = os.path.dirname(filename)
        filename = os.path.join(log_path, 'invalid.txt')
        with open(filename, 'w') as f:
            f.write('Invalid')
            
def create_gif_to_task3_2(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        state, _ = env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        frames = []
        total_reward = 0

        for t in range(steps):
            # Se o controller for uma rede neural
            if isinstance(controller, torch.nn.Module):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                actuation = controller(state_tensor).detach().numpy().flatten()
            else:
                # Se for um controller simples como "alternating_gait"
                actuation = controller(sim.get_dim_action_space('robot'), t)

            state, reward, terminated, truncated, _ = env.step(actuation)
            total_reward += reward
            frame = viewer.render('rgb_array')
            frames.append(frame)

            if terminated or truncated:
                break

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
        print(f"GIF salvo como {filename} | Recompensa total: {total_reward:.2f}")

    except Exception as e:
        print(f"Erro ao criar GIF: {e}")