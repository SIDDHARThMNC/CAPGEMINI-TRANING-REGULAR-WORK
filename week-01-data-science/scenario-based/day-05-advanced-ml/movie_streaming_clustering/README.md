# 🎬 Movie Streaming Platform - User Segmentation

## Scenario
A movie streaming company has collected data on 1,000 users to understand viewing patterns and improve personalization.

## Dataset Features
- **Average Watch Time per Week**: Hours spent watching content
- **Number of Devices**: TV, phone, tablet usage
- **Subscription Pauses**: Frequency of cancellations/pauses
- **Genre Preferences**: Action, Comedy, Drama percentages

## Objectives
1. **User Segmentation**: Group users into meaningful clusters
2. **Personalization**: Recommend tailored movie lists
3. **Loyalty Programs**: Design rewards for engaged users
4. **Churn Prevention**: Identify at-risk subscribers

## Methodology

### 1. K-Means Clustering
Applied unsupervised learning to discover natural user segments based on viewing behavior and preferences.

### 2. Elbow Method
Analyzed the within-cluster sum of squares (inertia) to find the point where adding more clusters provides diminishing returns.

### 3. Silhouette Score Validation
Measured cluster quality and separation to ensure segments are well-defined and meaningful.

## Expected Cluster Segments
- **Weekend Binge-Watchers**: High watch time, engaged users
- **Casual Family Viewers**: Low watch time, occasional viewers
- **Genre Loyalists**: Strong preference for specific genres
- **At-Risk Cancelers**: High subscription pauses
- **Multi-Device Power Users**: Active across multiple platforms

## Business Applications
- **Personalized Recommendations**: Tailored content for each segment
- **Loyalty Rewards**: Special perks for binge-watchers
- **Re-engagement Campaigns**: Win back at-risk users
- **Content Strategy**: Invest in genres preferred by largest segments

## Files
- `movie_streaming_analysis.py`: Complete clustering analysis
- `user_streaming_data.csv`: Generated user dataset
- `clustered_users.csv`: Users with cluster assignments
- `streaming_analysis.png`: Comprehensive visualization

## How to Run
```bash
python movie_streaming_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
