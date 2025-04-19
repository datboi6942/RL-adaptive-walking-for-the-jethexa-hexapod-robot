#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rospy
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jethexa_gym_env import JetHexaEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained JetHexa PPO model.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.zip)')
    parser.add_argument('--vec-normalize', type=str, default=None,
                        help='Path to the VecNormalize statistics file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--difficulty', type=int, default=None,
                        help='Terrain difficulty level (0-4), cycles through all if not specified')
    parser.add_argument('--render', action='store_true',
                        help='Enable Gazebo GUI for visualization')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save evaluation results to CSV')
    parser.add_argument('--comparison', action='store_true',
                        help='Compare against baseline policies')
    return parser.parse_args()

def evaluate_policy(env, model, n_episodes=5, vec_normalize=None):
    """
    Evaluate the policy for n episodes.
    
    Args:
        env: JetHexa environment
        model: Trained PPO model
        n_episodes: Number of evaluation episodes
        vec_normalize: VecNormalize wrapper (if used during training)
    
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    distances = []
    falls = []
    energy_usage = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        rospy.loginfo(f"Starting evaluation episode {episode+1}/{n_episodes}")
        
        while not done:
            # Apply observation normalization if using VecNormalize
            if vec_normalize is not None:
                norm_obs = vec_normalize.normalize_obs(obs)
                action, _ = model.predict(norm_obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Logging for debug
            if episode_length % 100 == 0:
                rospy.loginfo(f"Episode {episode+1}, Step {episode_length}, " +
                             f"Distance: {info['distance_traveled']:.3f}, " +
                             f"Falls: {info['falls']}")
        
        # Collect episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        distances.append(info['distance_traveled'])
        falls.append(info['falls'])
        energy_usage.append(info['energy_used'])
        
        rospy.loginfo(f"Episode {episode+1} completed:")
        rospy.loginfo(f"  Reward: {episode_reward:.2f}")
        rospy.loginfo(f"  Distance: {info['distance_traveled']:.2f} meters")
        rospy.loginfo(f"  Falls: {info['falls']}")
        rospy.loginfo(f"  Energy used: {info['energy_used']:.2f}")
    
    # Calculate average metrics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'mean_falls': np.mean(falls),
        'mean_energy': np.mean(energy_usage),
        'all_distances': distances,
        'all_rewards': episode_rewards,
        'all_falls': falls,
        'all_energy': energy_usage
    }
    
    return results

def evaluate_random_policy(env, n_episodes=5):
    """Evaluate a random policy as baseline."""
    episode_rewards = []
    distances = []
    falls = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        rospy.loginfo(f"Evaluating random policy, episode {episode+1}/{n_episodes}")
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        distances.append(info['distance_traveled'])
        falls.append(info['falls'])
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'mean_distance': np.mean(distances),
        'mean_falls': np.mean(falls),
        'policy': 'random'
    }
    
    return results

def evaluate_scripted_gait(env, n_episodes=5):
    """Evaluate a scripted tripod gait as baseline."""
    # Predefined tripod gait parameters
    tripod_params = np.zeros(env.action_space.shape)
    tripod_params[0] = 0.5  # Medium frequency
    tripod_params[1] = 0.0  # Tripod gait type
    
    # Set fixed amplitudes for a typical tripod gait
    # Shoulder joints
    for i in range(6):
        idx = 2 + 6 + i*3
        tripod_params[idx] = 0.3
    
    # Hip joints
    for i in range(6):
        idx = 2 + 6 + i*3 + 1
        tripod_params[idx] = 0.5
    
    # Knee joints
    for i in range(6):
        idx = 2 + 6 + i*3 + 2
        tripod_params[idx] = 0.4
    
    episode_rewards = []
    distances = []
    falls = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        rospy.loginfo(f"Evaluating scripted tripod gait, episode {episode+1}/{n_episodes}")
        
        while not done:
            obs, reward, done, info = env.step(tripod_params)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        distances.append(info['distance_traveled'])
        falls.append(info['falls'])
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'mean_distance': np.mean(distances),
        'mean_falls': np.mean(falls),
        'policy': 'tripod'
    }
    
    return results

def plot_comparison(results, difficulty):
    """Plot comparison between policies."""
    policies = [r['policy'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    distances = [r['mean_distance'] for r in results]
    falls = [r['mean_falls'] for r in results]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot reward comparison
    ax[0].bar(policies, rewards)
    ax[0].set_title(f'Average Reward (Difficulty {difficulty})')
    ax[0].set_ylabel('Reward')
    
    # Plot distance comparison
    ax[1].bar(policies, distances)
    ax[1].set_title(f'Distance Traveled (Difficulty {difficulty})')
    ax[1].set_ylabel('Distance (m)')
    
    # Plot falls comparison
    ax[2].bar(policies, falls)
    ax[2].set_title(f'Number of Falls (Difficulty {difficulty})')
    ax[2].set_ylabel('Falls')
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"policy_comparison_diff{difficulty}.png"))
    plt.close()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize ROS node
    rospy.init_node('jethexa_model_evaluation')
    
    # Create JetHexa environment
    env = JetHexaEnv()
    
    # Configure environment
    if args.render:
        # Set Gazebo to render (not headless)
        # This would typically modify a ROS parameter or environment variable
        rospy.loginfo("Enabling Gazebo rendering for visualization")
    
    # Load the trained policy
    rospy.loginfo(f"Loading model from {args.model}")
    model = PPO.load(args.model)
    
    # Load VecNormalize statistics if provided
    vec_normalize = None
    if args.vec_normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        
        rospy.loginfo(f"Loading normalization stats from {args.vec_normalize}")
        vec_normalize = VecNormalize.load(args.vec_normalize, env)
        vec_normalize.training = False  # Don't update stats during evaluation
        vec_normalize.norm_reward = False  # Don't normalize rewards during evaluation
    
    # Decide which difficulty levels to evaluate
    if args.difficulty is not None:
        difficulty_levels = [args.difficulty]
    else:
        difficulty_levels = range(5)  # Evaluate all difficulty levels
    
    all_results = []
    
    for difficulty in difficulty_levels:
        rospy.loginfo(f"Evaluating at difficulty level {difficulty}")
        env.set_difficulty(difficulty)
        
        # Evaluate trained policy
        results = evaluate_policy(env, model, n_episodes=args.episodes, vec_normalize=vec_normalize)
        results['difficulty'] = difficulty
        results['policy'] = 'trained'
        all_results.append(results)
        
        # Evaluate baselines for comparison
        if args.comparison:
            # Random policy
            random_results = evaluate_random_policy(env, n_episodes=args.episodes)
            random_results['difficulty'] = difficulty
            all_results.append(random_results)
            
            # Scripted gait (tripod)
            tripod_results = evaluate_scripted_gait(env, n_episodes=args.episodes)
            tripod_results['difficulty'] = difficulty
            all_results.append(tripod_results)
            
            # Create comparison plot
            plot_comparison([
                results,
                random_results,
                tripod_results
            ], difficulty)
    
    # Save results to CSV if requested
    if args.save_csv:
        df_list = []
        for result in all_results:
            # Extract episode-level data if available
            if 'all_distances' in result:
                for i in range(len(result['all_distances'])):
                    df_list.append({
                        'policy': result['policy'],
                        'difficulty': result['difficulty'],
                        'episode': i,
                        'reward': result['all_rewards'][i],
                        'distance': result['all_distances'][i],
                        'falls': result['all_falls'][i],
                        'energy': result['all_energy'][i]
                    })
            else:
                # Use aggregated data
                df_list.append({
                    'policy': result['policy'],
                    'difficulty': result['difficulty'],
                    'episode': 'mean',
                    'reward': result['mean_reward'],
                    'distance': result['mean_distance'],
                    'falls': result['mean_falls']
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(df_list)
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        rospy.loginfo(f"Results saved to {csv_path}")
    
    # Print overall summary
    rospy.loginfo("Evaluation complete!")
    for difficulty in difficulty_levels:
        trained_results = next(r for r in all_results if r['policy'] == 'trained' and r['difficulty'] == difficulty)
        rospy.loginfo(f"Difficulty {difficulty}:")
        rospy.loginfo(f"  Avg Reward: {trained_results['mean_reward']:.2f}")
        rospy.loginfo(f"  Avg Distance: {trained_results['mean_distance']:.2f} meters")
        rospy.loginfo(f"  Avg Falls: {trained_results['mean_falls']:.2f}")
        
        if args.comparison:
            random_results = next(r for r in all_results if r['policy'] == 'random' and r['difficulty'] == difficulty)
            tripod_results = next(r for r in all_results if r['policy'] == 'tripod' and r['difficulty'] == difficulty)
            
            rospy.loginfo(f"  Comparison - Random policy distance: {random_results['mean_distance']:.2f} meters")
            rospy.loginfo(f"  Comparison - Tripod gait distance: {tripod_results['mean_distance']:.2f} meters")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 