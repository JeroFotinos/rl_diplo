import imageio

def concatenate_gifs(gif1_path: str, gif2_path: str, output_path: str):
    # Read in existing gifs into numpy arrays
    gif1 = imageio.mimread(gif1_path)
    gif2 = imageio.mimread(gif2_path)

    # Concatenate the two gifs
    concatenated_gif = gif1 + gif2
    
    # Write out the result to a new gif
    imageio.mimsave(output_path, concatenated_gif)

# Usage
concatenate_gifs("experiments/sarsa/animations/episode_0.gif", "experiments/sarsa/animations/episode_100.gif", "episode_0_100_sarsa_e-greedy.gif")
