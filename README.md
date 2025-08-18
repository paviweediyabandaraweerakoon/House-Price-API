# House Price Prediction API

## Problem Description
This API predicts house prices based on key features: square footage, number of bedrooms/bathrooms, and age of the house. Uses a Random Forest Regression model trained on synthetic housing data.

## Model Choice Justification
- **Random Forest Regressor**: Chosen for its robustness, ability to handle non-linear relationships, and resistance to overfitting
- **Features**: Selected the most impactful features for house pricing (sqft, bedrooms, bathrooms, age)
- **Performance**: Achieves R² score of ~0.87 (87% accuracy)

## API Endpoints

### `GET /`
Health check endpoint - confirms API is running

### `POST /predict`
Main prediction endpoint
- **Input**: House features (sqft, bedrooms, bathrooms, age)
- **Output**: Predicted price with confidence level

### `GET /model-info`
Returns model metadata and feature information

## API Usage Examples

### Example 1: Average House
**Request:**
```json
{
  "sqft": 2000,
  "bedrooms": 3,
  "bathrooms": 2.0,
  "age": 10
}