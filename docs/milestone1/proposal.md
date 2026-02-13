# Proposal - Vancouver Airbnb Analyzer 

## Motivation and Purpose

### Motivation

Vancouver is one of the most popular travel destinations in Canada,
attracting both domestic and international visitors throughout the year.
At the same time, the city is known for its high cost of living, which
is reflected in the short-term rental market. Airbnb has become a major
accommodation option for travelers, offering a wide range of listings
that vary substantially in price, location, amenities, and guest
experience.

However, the abundance of options can make decision-making difficult for
travelers. Prices differ significantly across neighborhoods, and
listings with similar prices may offer very different levels of
convenience, accessibility, or overall quality. Moreover, the factors
that drive price variation are not always transparent, and the overall
structure of the Vancouver Airbnb market remains unclear. Similar points
can also be said about Airbnb ratings.

From a data science perspective, the Vancouver Airbnb dataset provides
an opportunity to explore how location, listing characteristics, and
guest preferences interact within a real-world marketplace. By analyzing
this dataset, we can move beyond simple descriptive statistics and
generate actionable insights for both travelers and market observers.

------------------------------------------------------------------------

### Purpose

The purpose of this project is threefold:

1.  **Support Traveler Decision-Making**\
    Identify which Vancouver neighborhoods offer the best value given
    different budgets, group sizes, and travel preferences such as
    walkability, transit access, safety, and proximity to attractions.

2.  **Understand Determinants of Airbnb Ratings**\
    Examine how listing characteristics and neighborhood attributes
    contribute to rating variation.

3.  **Understand Determinants of Airbnb Pricing**\
    Analyze how listing characteristics and neighborhood attributes
    contribute to price variation.

Together, these objectives allow us to analyze the Vancouver Airbnb
market from complementary perspectives: recommendation, explanation, and
segmentation.

------------------------------------------------------------------------

# Dataset Description

This project utilizes a comprehensive dataset of Airbnb listings in Vancouver, British Columbia. The data captures structural, pricing, availability, host, and review characteristics for active short-term rental properties, along with supporting calendar, review, and geographic boundary data.  
  
The primary detailed "listings" table in the dataset contains property-level observations, where each row represents a unique Airbnb listing.


## 1. Property Characteristics

-   `room_type`
-   `property_type`
-   `accommodates`, `bedrooms`, `beds`, `bathrooms`
-   `amenities`  
  
These features allow us to compare property configurations and evaluate how property size and type influence pricing and revenue potential.

## 2. Location Attributes

-   `neighbourhood`, `neighbourhood_group`
-   `latitude`, `longitude`  
  
These are supplemented by a GeoJSON boundary file for Vancouver neighborhood’s, enabling spatial visualization and neighbourhood-level aggregation of pricing, availability, and revenue metrics.

## 3. Pricing and Stay Constraints

-   `price`
-   `minimum_nights`, `maximum_nights`
-   `adjusted_price`  
  
These fields allow us to examine pricing distributions across neighbourhoods and assess how booking restrictions impact occupancy and revenue potential.

## 4. Host Activity

-   `host_id`, `host_name`
-   `host_listings_count`
-   `host_verifications`  
  
These variables help distinguish between single-property hosts and multi-property operators, which is particularly relevant in a regulated short-term rental market like Vancouver.

## 5. Availability Indicators

-   `availability_30`, `availability_60`, `availability_90`,
    `availability_365`
-   `estimated_occupancy_365d`
-   `estimated_revenue_365d`  
  
These variables enable estimation of revenue potential and occupancy performance at both listing and neighbourhood levels.

## 6. Review Performance

-   `number_of_reviews`, `reviews_per_month`
-   `review_scores_rating`
-   Sub-scores such as `review_scores_cleanliness`,
    `review_scores_location`, etc.  
  
Additionally, a separate reviews dataset contains timestamped textual reviews, allowing for qualitative exploration of guest satisfaction and neighbourhood perception.

------------------------------------------------------------------------

# Research Questions

## RQ1: Best Airbnb Options for Tourists

Our first research question will be **exploring the best Airbnb options for tourists to Vancouver, based on various factors including: price, location, group size, length of stay and more**. We will be performing analysis on the Vancouver Airbnb dataset to achieve these insights, particularly on the detailed “listings” table, which contains fields that correspond to many of the factors. A tourist hotspot of Vancouver will be identified (for example, downtown) and the listings will be compared in relation to their distance from the hotspot, price, amenities, max or min nights, group size, ratings and more. Our goal is to provide filtering on these tourist attributes (like group size, max distance, or price range) to allow tourists to find the best options for their needs. We may also incorporate some form of a scoring model or scoring system to accurately assess the overall “best fits” of the Airbnb given all the factors.  
  
 “A group of tourists plan to visit Vancouver under specific conditions. The conditions being that they have a group of 6, need to stay for a full week, and a limited price range for their stay. One of their group members has a disability which makes travelling far by foot difficult. It can be difficult to find the appropriate Airbnb for their stay under these circumstances. However, using our dashboard app they can select filters which apply to each of these circumstances. They are then provided with all their best options ranked from best possible match to least. This saves the group time because even with filters on a modern-day apps like google maps it can be quite difficult to isolate the options based on their circumstances, and they may have to dig to find the correct matches.”


## RQ2: Predictors of Higher Ratings

Our second research question **explores which listing and host characteristics most strongly predict higher guest ratings for Airbnb’s across Vancouver**. Using the detailed “listings” table, we will fit a predictive model with overall guest rating as the response variable and predictors such as price, location/neighborhood, room type, amenities, and host attributes. We will evaluate predictor importance to identify which features are most associated with higher ratings. Our dashboard will visualize the most influential predictors and allow users to compare predicted ratings across different listing profiles.  
  
 "A host notices their rating is lower than similar nearby listings but doesn’t know what to improve. Using our dashboard, they filter to comparable listings (same neighborhood, similar price, same room type) and see which features most strongly relate to higher ratings, helping them prioritize the changes most likely to improve guest satisfaction.”


## RQ3: Drivers of Airbnb Prices

Our third research question is to **explore which listing characteristics and neighborhood attributes most strongly influence Airbnb prices across Vancouver**. Our goal is to evaluate which of the predictors (including location, number of rooms, neighborhood, amenities, cleanliness and more) most strongly predicts a higher price. We will be fitting a predictive model to the data provided in the detailed “listings” table. The predictors of our model will include the various characteristic and attributes that influence Airbnb prices in Vancouver, and the response variable is the price itself. Predictor importance will be evaluated to identify the strongest influences. Our dashboard will include a simulated prediction values of price based on given inputs for the predictors. The predictor values can be modified on the board to see the resulting price, with the predictors being ranked by importance.  
  
 “An Airbnb host is planning is trying to decide what their listing price should be. The listing includes several extra amenities that the host in not sure whether should increase the price. The host was searching online but finding it hard to come to a finalized answer. Instead of searching online, they use our dashboard which allows them to easily predict what price is suitable for their listing by adjusting the amenity predictor and matching all predictors to their listing.”

