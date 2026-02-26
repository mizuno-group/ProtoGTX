# ProtoGTX: Prototypical Graph Transformer for Interpretable WSI Analysis

**ProtoGTX** is a novel Graph Transformer designed for interpretable computational pathology. By synergizing spatial topology and morphological semantics, it delivers high-performance diagnostic predictions with transparent, clinically-aligned explanations.

---

### 📂 Repository Structure

* **`protogtx/`**: Core architecture of the Prototypical Graph Transformer.
* **`graph_builder/`**: Modules for constructing spatial graphs from WSI patches.
* **`dataset_module/`**: Graph-based data loaders for efficient WSI processing.
* **`visualization/`**: Utilities for generating dual-modality (GraphCAM & Prototype) interpretability maps.
* **`example/`**: Step-by-step interactive tutorials and evaluation demos.

---

### 🚀 Getting Started

Follow our step-by-step tutorials in the `example/` directory.

### Interactive Tutorials (Jupyter Notebooks)
We provide a pipeline from graph construction to final evaluation:

1. **[1_build_grid_graphs.ipynb](example/1_build_grid_graphs.ipynb)**
   - Learn how to transform a Whole Slide Image (WSI) into a grid-based spatial graph representation.
2. **[2_define_prototype_features.ipynb](example/2_define_prototype_features.ipynb)**
   - Explore how latent features are mapped to an interpretable vocabulary of morphological prototypes.
3. **[3_prediction_evaluation.ipynb](example/3_prediction_evaluation.ipynb)**
   - Perform model inference on a representative LUAD (Lung Adenocarcinoma) case and visualize the dual-modality explanations.

---
