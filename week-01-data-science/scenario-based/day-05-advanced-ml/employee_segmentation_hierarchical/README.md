# 💻 Tech Company Employee Segmentation - Hierarchical Clustering

## Business Problem
A tech company wants to understand its employees better to design effective training programs and fair salary structures. The HR department collected data on employee Age and Annual Salary to identify natural groupings.

## Dataset
Sample of 8 employees with:
- **Age**: Employee age in years
- **Annual Salary**: Yearly compensation in dollars

## Business Objectives
1. **Training Programs**: Design targeted development programs for each segment
2. **Salary Structures**: Create fair and competitive compensation bands
3. **Career Progression**: Identify clear career paths from entry to senior levels
4. **Retention Strategy**: Develop segment-specific retention plans
5. **Recruitment**: Establish benchmark salaries for hiring

## Expected Employee Segments
- **Entry-Level Employees**: Young, lower salary, need foundational training
- **Junior Professionals**: Growing experience, moderate salary
- **Mid-Career Professionals**: Established skills, competitive salary
- **Senior High-Earners**: Experienced leaders, premium compensation

## Methodology

### Hierarchical Clustering
- Creates tree-like structure (dendrogram) showing employee relationships
- Doesn't require pre-specifying number of clusters
- Allows visual inspection of natural groupings
- Multiple linkage methods compared (Ward, Complete, Average, Single)

### Evaluation Metrics
- **Silhouette Score**: Measures cluster quality and separation
- **Dendrogram Analysis**: Visual inspection of hierarchical structure
- **Cluster Statistics**: Age and salary distributions per segment

## HR Applications

### Training & Development
- **Entry-Level**: Onboarding, technical skills, mentorship programs
- **Junior**: Advanced courses, leadership basics, project management
- **Mid-Career**: Leadership development, strategic thinking, team management
- **Senior**: Executive coaching, industry conferences, innovation workshops

### Compensation Strategy
- **Entry-Level**: Competitive entry packages, learning stipends
- **Junior**: Mid-range salary, performance bonuses, health benefits
- **Mid-Career**: Above-market salary, stock options, comprehensive benefits
- **Senior**: Premium packages, equity, executive benefits, retention bonuses

### Career Growth
- **Entry-Level**: Clear career path, skill certifications, promotion criteria
- **Junior**: Team lead opportunities, cross-functional projects
- **Mid-Career**: Senior roles, department head positions
- **Senior**: C-level track, board positions, succession planning

## Key Insights
1. **Clear Salary Bands**: Distinct compensation tiers for fair pay structure
2. **Natural Progression**: Logical career path from entry to senior levels
3. **Targeted Development**: Different training needs per segment
4. **Retention Focus**: Premium retention strategies for high-earners
5. **Hiring Benchmarks**: Data-driven salary offers for new hires

## Files
- `employee_clustering.py`: Complete hierarchical clustering analysis
- `employee_data.csv`: Original employee dataset
- `clustered_employees.csv`: Employees with cluster assignments
- `employee_clustering_analysis.png`: Comprehensive visualizations

## How to Run
```bash
python employee_clustering.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

## Business Impact
- **Fair Compensation**: Data-driven salary structures
- **Employee Development**: Targeted training programs
- **Talent Retention**: Segment-specific retention strategies
- **Strategic Planning**: Clear workforce segmentation for HR decisions
