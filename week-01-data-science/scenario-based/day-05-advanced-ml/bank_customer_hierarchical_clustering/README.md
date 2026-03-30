# 🏦 Retail Bank Customer Segmentation - Hierarchical Clustering

## Scenario
A retail bank wants to understand its customers better by analyzing their Age and Annual Income. The goal is to group customers into meaningful segments for targeted financial products and marketing campaigns.

## Dataset
Sample of 8 customers with:
- **Age**: Customer age in years
- **Annual Income**: Yearly income in rupees

## Business Objectives
1. **Targeted Loan Offers**: Design appropriate loan products for each segment
2. **Personalized Investment Plans**: Create investment strategies based on income levels
3. **Marketing Campaigns**: Develop segment-specific promotional campaigns
4. **Customer Retention**: Identify high-value customers for premium services

## Methodology

### Hierarchical Clustering
Unlike K-Means, hierarchical clustering:
- Doesn't require pre-specifying number of clusters
- Creates a tree-like structure (dendrogram) showing relationships
- Allows visual inspection to determine optimal clusters
- Works well with small datasets

### Linkage Methods Compared
1. **Ward**: Minimizes variance within clusters (best for customer segmentation)
2. **Complete**: Maximum distance between clusters
3. **Average**: Average distance between all pairs
4. **Single**: Minimum distance between clusters

### Evaluation
- **Silhouette Score**: Measures how well-separated clusters are (range: -1 to 1)
- **Dendrogram Analysis**: Visual inspection of cluster hierarchy

## Expected Customer Segments
- **Young Starters**: Low income, younger age (entry-level products)
- **Growing Professionals**: Moderate income, mid-age (career growth products)
- **Established Earners**: Good income, mature age (wealth building)
- **High-Income Segment**: High income, senior age (premium services)

## Business Applications
- **Product Design**: Tailor financial products to segment needs
- **Risk Assessment**: Different credit policies per segment
- **Cross-Selling**: Recommend relevant products based on segment
- **Customer Service**: Prioritize high-value segments

## Files
- `hierarchical_clustering_bank.py`: Complete analysis with multiple linkage methods
- `customer_data.csv`: Original customer dataset
- `clustered_customers.csv`: Customers with cluster assignments
- `hierarchical_clustering_analysis.png`: Comprehensive visualizations

## How to Run
```bash
python hierarchical_clustering_bank.py
```

## Key Insights
- Dendrogram shows natural groupings in customer base
- Ward linkage typically provides best business-relevant clusters
- Small dataset allows detailed per-customer analysis
- Clear income-based segmentation enables targeted strategies

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
