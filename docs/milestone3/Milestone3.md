# Milestone 3
## Group 6 - Andre Gnandt, Ilakiya Paulraj, Yusen Rong and Bingzheng Jin  

### Part 1 - Questions Asked to Users:  
To evaluate the usability and effectiveness of our Vancouver Airbnb dashboard, we asked peer users to explore the dashboard and answer several questions about their experience. The questions were designed to assess key usability aspects such as navigation, visualization clarity, interactivity, performance, and overall usefulness.  
  
1. How easy was it to navigate through the dashboard and locate the information you needed?  
  
2. Were the visualizations helpful in understanding Airbnb listings and pricing trends across Vancouver neighbourhoods?  
  
3. Were the interactive elements (such as filters for neighbourhood, room type, or price range) intuitive and easy to use?  
  
4. Did the dashboard help you quickly identify patterns or differences in Airbnb listings across different areas of Vancouver?  
  
5. How clear and visually appealing did you find the layout, colors, and design of the dashboard?  
  
6. Did the dashboard respond quickly when you interacted with filters or visualizations?  
  
7. Were there any features or information you expected to see but were missing?  
  
8. Do you have any suggestions that could improve the usability or usefulness of the dashboard?  
  
These questions helped us evaluate the overall user experience, usability, and effectiveness of the visualizations in supporting users’ exploration of Vancouver Airbnb data.  
  
### Part 2 - Feedback from Users:  
Feedback was collected from peer users who interacted with the Vancouver Airbnb dashboard and answered the usability questions. The following summarizes the feedback received.  
  
#### User 1  
- The dashboard was easy to navigate, and the layout was intuitive.
- The map visualization helped in understanding how Airbnb prices vary across different Vancouver neighbourhoods.
- Filters for neighbourhood and room type were useful for exploring the data.
- Suggested adding clearer labels to some charts to improve interpretation.  
  
#### User 2
- The visualizations were informative and helped identify price differences between listings.
- The dashboard design looked clean and visually appealing.
- Suggested adding more detailed tooltips to display additional information when hovering over data points.
- Recommended including a price distribution chart to better understand overall pricing patterns.  
  
#### User 3
- The dashboard was easy to navigate, and the filters were intuitive and easy to use.
- The bar charts in the Price Estimator and Rating Estimator clearly showed which features influence price and ratings, but hover interactions would improve usability.
- Some labels in the “Tourist Listings” tab used variable names (e.g., tourist score) and should be replaced with clearer titles such as Tourist Score.
- The red color used for the Price Estimator and Rating Estimator text appeared too strong; a softer color such as blue or green was suggested.
- The dashboard loaded quickly when run locally, though deployment performance might need to be tested.
- A suggested feature was the ability to automatically find the cheapest or best-rated Airbnb based on selected criteria such as property type, host communication, ratings, and amenities.
- The user did not observe clear patterns between neighborhoods using the current visualizations.
Additional Feedback
- The “Calculate Price” and “Calculate Rating” controls did not clearly look like buttons.
- Hover interactions should be added to chart bars to display additional details.  
  
### Part 3 - Reflect on how you will improve your app based on user feedback.  
#### User 1 
**1. Suggested adding clearer labels to some charts for better interpretation.**  
&emsp;We recognize the lack of clarity in some of the filter titles in both the Price and Rating Estimator tabs. For example, we will find and replace titles of the estimator filters like 'Bathrooms' with something along the lines of 'Bathroom Quantity' to more accurately communicate its use to the user. Similar changes will be made to the matching Listing Feature labels on the y-axis of both the "Top Influences on Price" and "Top Influences on Rating" bar charts. Identical improvements will take place on the 'Tourist Listings' tab. These include replacing the 'Top N' slider label with a more accurate title such as 'Number of Matches Displayed (ranked best to worst)' to let the user know that only the number of top matches as identified by the slider will be displayed. We may also improve the hover displays of the map bubbles to include more relevant info on the listing data and exclude any irrelevant info.  
  
#### User 2
**1. Suggested adding more detailed tooltips to provide additional information when hovering over data points.**  
&emsp;As mentioned in the previous point, we will modify the hover data displayed on the map of the 'Tourist Listings' tab to include more detailed and relevant info related to the listing and exclude any irrelevant info.  
  
**2. Recommended including a price distribution chart to better understand overall pricing patterns.**  
&emsp;Time permitted, we may create a KDE (or histogram) for the price and rating estimator tabs. These charts will display the distribution of the price (or rating) for each feature listed in the filters. The y-axis would be the price (or rating) and the x-axis would be a single feature from the filters, with the option to select between these filter features with a dropdown.  
  
#### User 3  
**1. Include intra-plot interaction (hover actions on categories) for the price and rating influences bar charts.**  
&emsp;We will include hover actions for the bar chart's categories, as long as these actions support a suitable display format that does not inflate page load and refresh times, the common html.Iframe format was huge and resulted in enormously long page refresh times. The hovers will display the category name and exact percentage of influence.  
  
**2. Change the variable name "tourist_score" to "Tourist Score".**  
&emsp;We recognize that some of the titles and labels on the 'Tourist Scores' tab are actually the exact variable names from the Python code. We recognize this mistake, and it will be fixed.  
  
**3. I didn't notice any patterns between neighborhoods.**  
&emsp;We recognized this issue when testing our app and realized it could possibly be a problem with the ML model accuracy. We will investigate this further and improve the models' performances and accuracy if necessary. If so, the models will be improved to more accurately reflect the influence that neighborhoods have on Airbnb prices (or ratings) in Vancouver. Please note, it is quite possible that neighborhoods don't actually have a considerable influence on Airbnb prices in Vancouver.  
  
**4. I found the red text for Price Estimator and Rating Estimator to be a little strong. I think an easier color would be better suited, like green or blue.**  
&emsp;The color of this text will be changed to a more suitable color, such as green or blue.  
  
**5. It loaded quickly, but I was running it locally. When deployed, there might be a greater delay.**  
&emsp;This is actually not a true issue and possibly not fixable; the ML models need to be trained on each deployment/startup of the app, resulting in a great delay. This occurs only once at startup/deployment.  
  
**6. Having a feature to find the cheapest/best rated Airbnb for all neighborhoods would be a cool feature if possible. So I can select the property type, host communication, host response rate, location rating, overall rating, accommodates, bedrooms, beds, and bathrooms, and it finds the cheapest Airbnb or the top-rated Airbnb. This would be useful for users that want to look at the best possible price or best rating without having to try all the neighborhoods.**  
&emsp;We may incorporate this feature into one of our tabs (time permitting), most likely as part of the Tourist Listings tab. This could be a long and complex task, especially to display it on the current dashboard effectively.  
  
**7. The user did not at first recognize that the "calculate price" and "calculate rating" buttons were in fact buttons on the price estimator and ratings estimator tabs.**  
&emsp;We will investigate UI methods and come up with a solution to more effectively style the buttons so that the user in fact recognizes them as buttons.  
  
#### Possible Extra Improvements (time permitted only)
1. Improve the overall performance and accuracy of the ML models. (including more accurate feature importance and feature selection)
2. Related to the previous point, wrangle the amenities data to include significant amenities as filters and features of the models and predictor pages.
2. Improve the accuracy of the scoring system of the 'Tourist Listings' tab.
3. Add extra filters (like walkability and transit access) to the 'Tourist Listings' tab.











