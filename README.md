# FYP-main

  This repository contains 2 programs, the WARP-Q software by itself, and the AQP Pipeline, which implements the WARP-Q audio metric.
  
  Both of these programs have been modified to include the work of introducing a deep lattice network to produce objective Mean Opinion Scores (MOS) by replicating subjectively input human results.
  
  The original programs without modifications can be found at:
  
    https://github.com/QxLabIreland/AQP
    
    https://github.com/QxLabIreland/WARP-Q
    
  Full credit to their respective authors and creators for everything not listed in the next paragraph.
  
  NOTE: To run them you will need to follow the steps in each respective README for each program. Additionally, you will need data. My findings were found using the Genspeech dataset, found at:
  
    https://github.com/QxLabIreland/datasets
  
# Modified files and files added:

  WARP-Q:
  
    deep_lattice_network.py
    
    WARPQ_main_code.py
  
  AQP:
  
    config/warpq_pesq_dataset.json
    
    nodes/dlnnode.py
    
    visualisations.ipynb

# Additional Libs Needed Beyond the WARP-Q and AQP Pipeline Libs
  tensorflow; keras
  
  tensorflow_lattice

  keras.layers; Dense

  keras.models; Sequential

  scipy.stats; pearsonr, spearmanr
  
  absl; app, flags
  
  # IMPORTANT
  
  To save time scrawling through the AQP Pipeline figuring out how to run it, follow these instructions.

   1. download the datasets from the link above.

   2. put them in a directory called "resources" in the AQP-main directory. 

   3. after initial setup, run the following 2 commands:

          python utils/prepend_files.py --prepend_with "resources/" --dataset "resources/genspeech.csv"

          pip install -r requirements.txt

   4. Then the command to run the actual pipeline with the dataset and the lattice network is this:

          python pipeline.py --root_node_id "Load DF" --graph_config_path "config/warpq_pesq_dataset.json" --plot_graph
