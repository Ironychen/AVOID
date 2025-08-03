[README.md](https://github.com/user-attachments/files/21563657/README.md)
# KDD_AVOID

KDD_AVOID is a simulation and detection framework for rumor propagation based on social networks and personalized agents. It leverages GCN-based propagation graph modeling and LLM-based multi-agent simulation to better understand and detect misinformation.

## Project Structure

```
KDD_AVOID/
├── Env_Rumor_politic_persona1/           # Stores agent environments including long- and short-term memory
├── gcn_data_politic_img_persona/         # Stores propagation graphs generated for each news article
├── persona/                              # Persona extraction and modeling code
├── dataset_processing.py                 # Dataset loading and preprocessing utilities
├── earlystopping.py                      # Early stopping mechanism for model training
├── gcn.py                                # GCN model definition
├── node_feature.py                       # Node-level feature extraction
├── LLM_prompt.py                         # LLM prompt construction and message simulation
├── Retriever.py                          # LLM-based retrieval for agent decision-making
├── self-map.py                           # Self-mapping logic for propagation
├── pro_mul.py                            # Multi-agent propagation simulation
├── main.py                               # Entry point for misinformation detection
```

## Module Descriptions

- **Environment Construction**:
  - `Env_Rumor_politic_persona1/`: Contains the full agent memory structure (short-term + long-term).
  - `persona/`: Extracts and assigns persona traits for realistic agent behavior.
  - `gcn_data_politic_img_persona/`: Contains propagation graphs (as adjacency matrices, features, etc.) per article.

- **dataprocessing**:
  - `dataset_processing.py`, `node_feature.py`: Used for data parsing, feature extraction, and graph preparation.
  - `gcn.py`: Implements the GCN for learning over propagation graphs.
  - `earlystopping.py`: Prevents overfitting during training by monitoring validation loss.

- **Agent-based Simulation**:
  - `LLM_prompt.py`, `Retriever.py`, `self-map.py`, `pro_mul.py`: Use LLMs and rules to simulate multi-agent rumor propagation, decisions, and interactions.

- **Detection Logic**:
  - `main.py`: Main detection pipeline that integrates all components and outputs results.




This project is developed for research purposes related to rumor propagation modeling and early misinformation detection in social environments.
