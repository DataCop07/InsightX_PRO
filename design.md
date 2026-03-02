# InsightX - Executive Financial Intelligence Platform

## Project Overview
InsightX is a comprehensive UPI transaction analytics platform built with Streamlit that transforms financial transaction data into strategic executive intelligence dashboards. The platform provides real-time monitoring, risk analysis, fraud detection, and behavioral intelligence for UPI transactions.

## Architecture

### Technology Stack
- **Framework**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Image Processing**: PIL (Python Imaging Library)
- **Analytics**: Custom CSV-based event logging

### Application Structure
```
project/
├── app.py                          # Main landing page and navigation
├── analytics.py                    # Event logging module
├── assets/
│   └── landing_image.jpg          # Landing page banner
├── data/
│   └── upi_transactions_2024.csv  # Transaction dataset
└── pages/
    ├── dashboard.py               # Transaction health dashboard
    ├── analysis_mode.py           # Advanced filtering and analysis
    ├── query_intelligence.py      # Query-based insights
    ├── risk_control.py            # Risk monitoring and fraud detection
    ├── innovation_lab.py          # AI-powered insights (rule-based)
    ├── export_section.py          # Data export functionality
    └── global_filter.py           # Interactive filtering interface
```

## Core Features

### 1. Landing Page (app.py)
- Animated gradient background with glass morphism design
- Executive KPI cards (transactions, revenue, success rate, risk alerts)
- CSV file upload functionality
- Session state management for global dataset
- Simple authentication system
- Navigation sidebar (Home, About, Contact, Login)

### 2. Dashboard (dashboard.py)
**Purpose**: Real-time financial analytics and transaction health monitoring

**Key Metrics**:
- Basic structure overview (states, banks, transaction types)
- Transaction health (success/failure rates, flagged transactions)
- Volume summary (total volume, monetary value, daily averages)
- Ecosystem overview (device types, merchant categories)
- Time & behavior intelligence (peak hours, weekend vs weekday patterns)

**Visualizations**:
- Pie charts for transaction type, device type, merchant category distribution
- Bar charts for hourly and daily transaction trends
- Gradient metric cards with color-coded insights

### 3. Analysis Mode (analysis_mode.py)
**Purpose**: Deep-dive transaction analysis with advanced filtering

**Features**:
- Multi-dimensional filtering (transaction type, merchant, status, state, bank, device, fraud flag)
- Amount range and risk score sliders
- Transaction ID search
- Smart duplicate detection (based on sender/receiver bank, amount, hour)
- Risk score calculation algorithm
- Executive summary generation

**Risk Score Formula**:
```
risk_score = (fraud_flag × 50) + 
             (amount/max_amount × 30) + 
             (late_night_hour × 10) + 
             (unusual_device × 10) + 
             (unusual_network × 10)
```

**Visualizations**:
- Transaction type and device distribution
- Amount distribution by risk level
- Risk score heatmap (day vs transaction type)
- Export filtered and high-risk data

### 4. Query Intelligence (query_intelligence.py)
**Purpose**: Pre-built analytical queries organized by category

**Query Categories**:

1. **Descriptive**: Basic transaction overview
   - Average amount by transaction type
   - Success/failure/flagged rates
   - State-wise volume
   - Total transaction value

2. **Comparative**: Performance comparison
   - Failure rate by device (Android vs iOS)
   - Network type success rates
   - Bank-wise failure rates
   - Weekend vs weekday performance
   - Age group transaction patterns

3. **Temporal**: Time-based analysis
   - Peak transaction hours
   - Highest volume days
   - Success variation by hour
   - Flagged transactions by hour
   - P2M hourly distribution

4. **Segmentation**: User & ecosystem segmentation
   - Age group with most P2P transactions
   - States with highest flagged ratio
   - Device types in high-value transactions
   - Merchant categories driving highest value
   - Bank-wise distribution

5. **Correlation**: Pattern analysis
   - Network vs failure rate correlation
   - High-value flagged transactions
   - States with unusual failure rates
   - Device type vs success correlation
   - Transaction amount by age group

6. **Risk Analysis**: Anomaly detection
   - High-value flagged percentage
   - Transaction type with highest flagged ratio
   - Duplicate amount detection
   - Bank with highest flagged concentration
   - Flagged transactions by state/hour

### 5. Risk Control (risk_control.py)
**Purpose**: Comprehensive risk monitoring and fraud detection

**Features**:
- Duplicate transaction flagging
- Advanced risk score computation
- Multi-filter sidebar (transaction type, merchant, banks, age groups, device, network, status, day)
- High-value threshold slider
- Transaction ID search

**Risk Metrics**:
- Total flagged transactions
- Duplicate transactions
- High-risk transactions (score ≥ 70)
- Average risk score
- Percentage of risky transactions (score ≥ 50)

**Risk Intelligence**:
- Top flagged states
- Merchant categories with highest risk
- Device type risk analysis
- Transaction type risk analysis
- Daily risk trend visualization
- High-risk transaction table export

### 6. Innovation Lab (innovation_lab.py)
**Purpose**: AI-powered insights using rule-based intelligence

**Features**:
1. **AI Chatbot**: Offline smart engine for natural language queries
   - Fraud rate queries
   - Highest risk transaction identification
   - Average amount calculations
   - Duplicate detection

2. **Direct UPI Entry**: Transaction simulation
   - Real-time risk prediction
   - Gateway simulation
   - Risk level classification

3. **AI Risk Predictor**: Rule-based risk assessment
   - Amount and hour-based prediction
   - Risk level categorization (Low/Moderate/High)

4. **Smart Suggestion Engine**: Automated recommendations
   - Multi-factor authentication suggestions
   - Transaction velocity monitoring alerts
   - Duplicate blocking recommendations

5. **External File Import**: CSV upload for new data analysis

6. **Behavioral Fingerprint Detector**: Pattern analysis
   - Bank-wise timing inconsistency detection

7. **Fraud Heat Intelligence Map**: Visual fraud density analysis
   - Day and hour-based heatmap

8. **System Performance Metric**: Safety score calculation

### 7. Export Section (export_section.py)
**Purpose**: Data export and reporting

**Export Options**:
- Full dataset CSV
- High-value transactions (top 10%)
- Flagged transactions
- Duplicate transactions (smart check)

### 8. Global Filter (global_filter.py)
**Purpose**: Interactive filtering with dynamic visualization

**Features**:
- Transaction ID search
- Multi-select filters (type, merchant, status, state, device, day)
- Amount range slider
- Real-time data table (600px height)
- Summary metrics dashboard
- Distribution pie charts
- Filter-based trend analysis (daily and hourly)
- Executive insights generation

## Data Model

### Transaction Dataset Schema
```
- transaction_id: Unique identifier
- amount_(inr): Transaction amount
- transaction_status: success/failed/failure
- transaction_type: P2P/P2M/etc.
- sender_state: Geographic location
- sender_bank: Originating bank
- receiver_bank: Destination bank
- sender_age_group: Demographic segment
- receiver_age_group: Demographic segment
- device_type: Android/iOS/etc.
- network_type: WiFi/4G/5G/etc.
- merchant_category: Business category
- fraud_flag: 0/1 indicator
- hour_of_day: 0-23
- day_of_week: Monday-Sunday
- is_weekend: Boolean indicator
```

## Design Patterns

### 1. Session State Management
- Global dataset stored in `st.session_state.df`
- Persistent across page navigation

### 2. Data Normalization
- Column name standardization (lowercase, underscore-separated)
- Text column cleaning (strip whitespace, handle nulls)
- Numeric conversion with error handling

### 3. Risk Scoring Algorithm
Multi-factor risk assessment combining:
- Fraud flag (50 points)
- Transaction amount (30 points)
- Time of day (10 points)
- Device type (10 points)
- Network type (10 points)
- Duplicate flag (10 points)

### 4. Visual Design System
- Gradient backgrounds with animation
- Glass morphism effects
- Color-coded metric cards
- Consistent color schemes:
  - Success: Green gradients
  - Warning: Orange/Yellow gradients
  - Error: Red gradients
  - Info: Blue gradients

### 5. Analytics Logging
- CSV-based event tracking
- Timestamp, event name, and details capture
- Append-only log file

## User Workflows

### Executive Dashboard Flow
1. Upload CSV dataset
2. View landing page KPIs
3. Navigate to Dashboard for health metrics
4. Explore Analysis Mode for deep-dive
5. Export filtered data

### Risk Management Flow
1. Access Risk Control page
2. Apply filters (bank, device, amount threshold)
3. Review high-risk transactions
4. Analyze risk trends
5. Export high-risk report

### Query Intelligence Flow
1. Navigate to Query Intelligence
2. Select query category
3. Click specific query button
4. View results and visualizations
5. Iterate with different queries

### Innovation Lab Flow
1. Access Innovation Lab
2. Ask chatbot questions
3. Simulate transactions
4. Review AI suggestions
5. Analyze behavioral patterns

## Security Considerations
- Hardcoded file paths (needs environment configuration)
- Basic authentication (username: admin, password: admin123)
- No encryption for sensitive data
- CSV-based logging (not production-ready)

## Performance Optimizations
- Data caching with `@st.cache_data`
- Efficient pandas operations
- Plotly for interactive visualizations
- Session state for data persistence

## Future Enhancements
1. Database integration (PostgreSQL/MongoDB)
2. Real-time data streaming
3. Machine learning models for fraud detection
4. Role-based access control
5. API integration for external systems
6. Advanced authentication (OAuth, JWT)
7. Automated alerting system
8. Mobile responsive design
9. Multi-language support
10. Advanced AI/ML integration
