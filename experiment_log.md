# 🧠 LoRA Fine-Tuning Results and Analysis

This document summarizes the results, interpretations, and recommendations from running the `LoraFineTuner` pipeline on the **TinyLlama-1.1B** model with the **Guanaco (1K)** dataset on Google Colab.

---

## 🧩 Experiment Overview

**Command / Script:**
```python
tuner = LoraFineTuner(
    base_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    dataset_name="mlabonne/guanaco-llama2-1k",
    output_dir="./llama-1.1B-chat-guanaco",
    num_train_epochs=2,
    learning_rate=2e-4,
    batch_size=2,
    gradient_accumulation_steps=16,
)
tuner.run_full_pipeline()
🧠 Inference Before Fine-Tuning
