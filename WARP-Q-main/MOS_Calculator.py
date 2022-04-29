import numpy as np
import pandas as pd
from absl import app

def main(_):
    warpq_data = pd.read_csv("Results_check.csv")
    calculate_mos(warpq_data)

def calculate_mos(warpq):
    warpq["mean"] = warpq.mean(axis = 1)
    warp_q_means = warpq["mean"]
    lower, upper = 1.0, 5.0
    warp_q_means_normalised = [lower + (upper - lower) * x for x in warp_q_means]
    
    # Display our pretty results
    for res in range(0, len(warp_q_means_normalised)):
        print("WARP-Q MOS:", round(warp_q_means_normalised[res], 2))
        warp_q_means_normalised[res] = round(warp_q_means_normalised[res], 2)

    # Final output
    mos = pd.DataFrame(warp_q_means_normalised)
    final_results = pd.DataFrame(warpq)
    final_results["MOS"] = mos

    # Move mean to the front for better visibility
    column_to_move = final_results.pop("mean")
    final_results.insert(0, "mean", column_to_move)
    # Move MOS just in front of mean
    column_to_move = final_results.pop("MOS")
    final_results.insert(0, "MOS", column_to_move)
    # Preview results
    print(final_results[:3])
    # Export results
    final_results.to_csv("WARPQ _MOS_Scores.csv")

if __name__ == '__main__':
  app.run(main)
