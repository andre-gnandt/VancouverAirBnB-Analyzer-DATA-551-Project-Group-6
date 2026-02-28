# Milestone 2 Reflection

For milestone 2, we implemented a usable prototype of the Vancouver Airbnb dashboard with three independent tab sections aligned to our research questions.

What is implemented:
- `RQ1` includes interactive filtering for budget, group size, room type, and minimum rating, with outputs shown as a map and ranked table.
- `RQ2` includes a random forest based rating predictor with user controls for key listing attributes.
- `RQ3` includes a random forest based price predictor with user controls for key listing attributes.
- We added two Altair bar charts to communicate feature relevance at a high level for rating and price.

What is not fully implemented yet:
- We are still using a simple feature set and baseline preprocessing strategy.
- Include Amenities and other important features in models
- More features and accuracy of tourist matches scoring system
- A comprehensively wrangled and cleaned dataset (cleaned data)
- UI styling and layout remain at prototype quality and are not final.

Known limitations:
- Model quality is sensitive to missing values and feature coverage in the source dataset.
- Startup can take longer if models need to be retrained automatically (already takes long from training random forests).
- Deployment configuration is included, but hosted deployment verification depends on platform settings and environment variables.

Future improvements:
- We aim to improve the 2 ML models accuracy (in RQ2 and RQ3) with more appropriate feature selection and maybe include additionaly important features in further milestones, particularly for the price model, the rating model is already quite accurate.
- We aim to complete a comprehensive data wrangling and cleaning process to produce a cleaned version of the datset to further optimize accuracy in our ML models. This is why we decided to skip data cleaning for milestone 2, because we want to do it thoroughly and carefully to help maximize ML accuracy.
- From the previous point, we aim to appropriately wrangle the "amenities" column of the datset into individual binary columns for only the significant amenities, to also help further maximize prediction accuracy
- Add richer model diagnostics and more robust evaluation reporting.
- Expand feature engineering (amenities, text-derived signals, and location enrichments).
- Improve app usability and visual consistency for milestone 3/4.
- Add issue-driven refinements based on TA and peer feedback.

