__all__ = ["print_experim_info"]

def print_experim_info(curr_vals: dict):
    print("Run idx: {}\t Seed: {}\t Downsampler Ratio: {}\t Ratio Seed: {}".format(curr_vals["++run_idx"], curr_vals["++seed"], curr_vals["++downsampler_ratio"], curr_vals["++ratio_seed"]))

if __name__ == "__main__":
    pass 