# Milestone 2 Reflection

For milestone 2, we implemented a usable prototype of the Vancouver Airbnb dashboard with three independent sections aligned to our research questions.

What is implemented:
- `RQ1` includes interactive filtering for budget, group size, room type, and minimum rating, with outputs shown as a map and ranked table.
- `RQ2` includes a neural-network based rating predictor with user controls for key listing attributes.
- `RQ3` includes a neural-network based price predictor with interactive controls.
- We added two Altair bar charts to communicate feature relevance at a high level for rating and price.
- The app now exposes `server = app.server` and includes a `Procfile` entry (`web: gunicorn src.app:server`) for deployment compatibility.
- A standalone NN training module (`src/ml_nn.py`) now supports a simple backend choice:
  - sklearn MLP baseline
  - PyTorch MLP (used when available, or requested explicitly)

What is not fully implemented yet:
- We are still using a simple feature set and baseline preprocessing strategy.
- Feature-importance charts are currently correlation-based summaries, not model-specific explainability.
- We have not implemented advanced validation workflows (e.g., full cross-validation reporting in the app).
- UI styling and layout remain at prototype quality and are not final.

Known limitations:
- Model quality is sensitive to missing values and feature coverage in the source dataset.
- Startup can take longer if models need to be retrained automatically.
- Deployment configuration is included, but hosted deployment verification depends on platform settings and environment variables.

Future improvements:
- Add richer model diagnostics and more robust evaluation reporting.
- Expand feature engineering (amenities, text-derived signals, and location enrichments).
- Improve app usability and visual consistency for milestone 3/4.
- Add issue-driven refinements based on TA and peer feedback.

