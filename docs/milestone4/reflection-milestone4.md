# Reflection
By the end of this milestone we have implemented all 3 RQs into our dashboard, each on separate tabs. This consists of both the price and rating estimators (RQ2 and RQ3), and the tourist listings finder (RQ1).
  
We have generally completed our dashboard, but are missing a few enhancements. We attempted to respond to and resolve most points from the TA and peer review feedback, but unfortunately were unable to implement all of them due to time constraints or other reasons. The points we did not fully implement were:  
1. TA Feedback 
    - "The app was not developed for a full-screen view" - we are not as familiar with these concepts and found it difficult to do this within a limited amount of time, we apologize.
    - "Use a familiar gradient for the rating such as red to green" - unfortunately we ran out of time.
    - "If the model is not yet accurate, please leave a corresponding note" - the models are not yet entirely accurate due to time constraints, but we have done some improvements to the accuracy of the price model using a boosting method for this milestone.  
  
2. Peer feedback
    - "Include a price distribution chart for the selected filters" - insufficient time 
    - "Feature to find the cheapest Airbnb for all neighborhoods" - insufficient time
    - "The red text of estimates is a little strong" - insufficient time
    - "hover actions for bar charts" - insufficient time and format is possibly not compatible for fast loading and performance of app
    - "did not recognize the calculate buttons were buttons" - individual preference, left buttons as is
  
Aside from the points above, we have made accomodations for all the other feedback in this milestone, some other points we would like to clarify:
1. TA points
    - "I suspect your filter options are not working correctly because I have the following questions/concerns (there are probably more): How is it possible that a place has one bedroom and still accommodates 4 people?" - The filters have been enhanced in this milestone to include cascading options. Certain combinations are still possible though, like for example, there can be an Airbnb with many bunk beds (20+ guests) in one bedroom which all share a single bathroom (hostel style).
    - "How is it possible that a property type does not affect the rating?" - property type does effect the rating, but the effect is very minimal, because property type is not a very strong predictor of Airbnb ratings in Vancouver. The ratings are overshadowed by host communication and/or cleanliness. However, our model also lacks some accuracy and likely does miss some of the (more minor) effects of property type.
    - "Is tourist_score the same as the rating?" - Tourist score is a separate metric from rating, used for match compatability based on filters.




