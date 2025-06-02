# Milestone 6: Final Development and Performance Testing – ASTRA

## Project Title:
**ASTRA: Adaptive Space Traffic Risk Assessor**

# Goal of the Month:
Reduce the error in the LSTM and GRU models' predictions, add the orbital paths of the benchmarked (real) satellites, based off their plotted points. Then polish and improve memory usage and user functionality by cleaning up the code and removing debugging functions and refining the CesiumJS visualization. 

## Current Progress Summary:
As of Milestone 6, ASTRA is functionally complete and integrates all core components of its AI-powered satellite trajectory prediction system. The Tkinter-based interface now allows users to choose between three models—LSTM, GRU, and XGBoost—each trained on historical satellite position data derived from real-world TLEs. Once a model is selected, predictions are computed and visualized in both a 3D Plotly environment and an interactive CesiumJS globe.

Significant UI improvements have been made this milestone. The prediction dropdown and labels are now organized more clearly, and all deprecated debug printouts have been removed from the notebook and GUI logic. The `start_prediction()` function now properly saves results as CSVs and Cesium-compatible `.json` files for visual validation. This streamlines the workflow and ensures all models follow a clean, consistent pipeline from training to evaluation.

## Challenges Faced:
The most significant challenge during this milestone was aligning the output data structures of the three different models so they could all be properly visualized using CesiumJS. While LSTM and GRU produce predictions from time series sequences, XGBoost requires a flat input format. Standardizing the `.json` output across models while maintaining correct coordinate formatting took extensive debugging.

Additionally, integrating satellite velocity data into the LSTM model required reshaping input tensors and updating the training/inference functions accordingly. This change introduced complexity in both scaling and inverse-transforming predictions, which needed custom handling in the post-processing pipeline. Thankfully, these issues were resolved by restructuring the training functions and saving model-specific scalers using `joblib`.

## Performance Testing Insights:
Using **Visual Studio Code** on a system with 32GB DDR5 RAM and an RTX 4070 GPU, we observed stable memory usage throughout model training and prediction. Before execution, the Python environment consumed about **1.3 GB** of RAM. Once the prediction process started—loading models, reading TLEs, generating sequences, and training—the RAM usage increased to approximately **3.3 GB**.

This jump indicates the system comfortably handles even the LSTM model’s training on 100+ samples, with no observed crashes, memory errors, or paging. Performance remained consistent during visualization steps, including generating 3D surface plots and writing JSON for Cesium. Given the hardware used and memory management within TensorFlow and XGBoost, ASTRA is well-optimized for graduate-level research workloads.

## Next Steps:
The final milestone steps include:
- Final GitHub documentation, including a refined README, setup instructions, and performance test summaries.
- Preparing all source code, test data, and visual assets for final evaluation.
- Coordinating with the advisor to finalize the evaluation rubric.

## Links:
- 🔗 [GitHub Repo](https://github.com/cczap7-Hub/DFSforSTMusingAI)
- Below are the screenshots of my RAM usage before and after running my project in VS Code. This shows my project is not too memory intensive for the vast majority of machines to run and compile it. 
  - ![Screenshot 2025-06-01 232319](https://github.com/user-attachments/assets/1fc0631a-4927-4fd5-b99d-58d074d9cb4a)
  - ![Screenshot 2025-06-01 232339](https://github.com/user-attachments/assets/06086f3e-2dae-4801-b2f3-e898817c5cd4)
- Below are the screenshots of my CesiumJS output of my satellites' benchmark and their orbit trajectories, as well as, the predicted outputs of my LSTM and GRU models' predictions.
  - ![Screenshot 2025-05-31 232930](https://github.com/user-attachments/assets/9fa220e6-a6ae-43d8-8d2a-0756e5e5784c)
  - ![Screenshot 2025-05-28 233402](https://github.com/user-attachments/assets/643106e0-d839-4351-ba63-9e4487acfcea)
  - ![Screenshot 2025-06-01 140106](https://github.com/user-attachments/assets/994dbff8-4306-4e42-9973-5a7081d1f2f9)
  - ![Screenshot 2025-06-01 140102](https://github.com/user-attachments/assets/bbd821bb-de8b-4b0f-b1e1-b4f97f439692)
  - ![Screenshot 2025-06-01 171331](https://github.com/user-attachments/assets/19335f2f-1b77-4cbc-be28-9eeb1c5204e4)

- Screencast and HLDD rubric were submitted for grading on my course's assignment pages.

_Authored by Corban Czap | June 2025_
