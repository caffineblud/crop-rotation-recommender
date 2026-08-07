# AI-Based Crop Rotation Recommendation System

University project — recommends the next crop to plant based on soil/climate data
and the previously grown crop, using ML + agronomic rotation rules.

## Project Structure
```
crop-rotation-recommender/
├── data/            # datasets (raw + cleaned)
├── notebooks/        # exploration, training, evaluation
├── app/              # Streamlit demo app
├── report/            # figures, charts, final report
├── requirements.txt
└── README.md
```

## Setup (100% free stack)

1. Create a virtual environment (or use Google Colab directly):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the dataset from Kaggle: **"Crop Recommendation Dataset"** (by Atharva Ingle).
   Place `Crop_recommendation.csv` in the `data/` folder.

3. Run the notebooks in order:
   - `01_data_exploration.ipynb` — EDA + rotation rule setup
   - (add) `02_model_training.ipynb` — train + save model as `crop_model.pkl`

4. Run the app locally:
   ```bash
   cd app
   streamlit run app.py
   ```

## Roadmap
- [ ] Week 1: Dataset + EDA + rotation rule mapping
- [ ] Week 2: Train & compare models (Random Forest, XGBoost, SVM)
- [ ] Week 3: Build Streamlit interface + integrate rotation logic
- [ ] Week 4: Report writing, evaluation, polish

## Notes
- Entire stack is free: Colab/local Python, scikit-learn, XGBoost, Streamlit (run locally, no deployment needed unless desired).
- Rotation logic combines ML crop suitability prediction with domain rules (crop family rotation, nitrogen fixing/depleting alternation).
