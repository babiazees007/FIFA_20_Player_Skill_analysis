# PRCP-1004: FIFA 20 Player Skill Analysis, Unsupervised Clustering & Tactical Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering%20%26%20PCA-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Dashboard](https://img.shields.io/badge/Interactive-Web%20Dashboard%20(HTML5%2FCSS3)-00f5a0.svg)](index.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end sports analytics and machine learning system evaluating **18,278 professional football players** from EA Sports' FIFA 20 dataset. This project addresses real-world sports scouting challenges, eliminates target leakage in positional clustering, answers high-impact recruitment and economic compensation questions, and benchmarks four unsupervised machine learning algorithms for production deployment.

---

## 📌 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [Interactive Web Dashboard (`index.html`)](#-interactive-web-dashboard-indexhtml)
3. [Repository Structure](#-repository-structure)
4. [Task 3: Resolution of Core Business Questions](#-task-3-resolution-of-core-business-questions)
   - [Q1: Top 10 Player Producing Nations](#q1-top-10-player-producing-nations)
   - [Q2: Player Aging Curve & The Ceiling Horizon](#q2-player-aging-curve--the-ceiling-horizon)
   - [Q3: Offensive Wage Economics (Mean vs. Median Paradox)](#q3-offensive-wage-economics-mean-vs-median-paradox)
5. [Task 2: Feature Engineering & Machine Learning Clustering](#-task-2-feature-engineering--machine-learning-clustering)
   - [Data Cleaning & Positional String Parsing](#data-cleaning--positional-string-parsing)
   - [Target Leakage Prevention](#target-leakage-prevention)
   - [Unsupervised Model Benchmark](#unsupervised-model-benchmark)
   - [Discovered Tactical Archetypes](#discovered-tactical-archetypes)
6. [Task 1: Exploratory Data Analysis & Domain Deep-Dives](#-task-1-exploratory-data-analysis--domain-deep-dives)
7. [7 Engineering Challenges & Technical Solutions](#-7-engineering-challenges--technical-solutions)
8. [Production Deployment & Strategic Recommendations](#-production-deployment--strategic-recommendations)
9. [Getting Started & Installation](#-getting-started--installation)

---

## 🎯 Executive Summary & Problem Statement

Modern football recruitment is often plagued by subjective scouting biases, cognitive heuristics, and inflated transfer valuations tied to media hype. 

### Core Objectives:
1. **Mathematical Player Profiling**: Group players purely by their **in-game athletic and technical style**, rather than superficial composite ratings (`overall`) or commercial popularity (`international_reputation`).
2. **Economic & Lifecycle Intelligence**: Empirically determine when players stop developing and demystify the financial disparity across offensive positions.
3. **Low-Latency Production Clustering**: Deliver an algorithm capable of sub-second inference latency ($O(k \cdot d)$) for real-time scouting databases.

---

## 🌐 Interactive Web Dashboard (`index.html`)

This repository includes a modern, zero-dependency interactive HTML5/CSS3/JavaScript dashboard: [`index.html`](index.html).

### Key Features:
- **Interactive Visualizations**: Powered by Chart.js (National Player Rankings, Age vs. Growth Curve, Offensive Compensation Comparison, and 6-Axis Tactical Radar).
- **Tactical Archetype Explorer**: Attribute breakdowns and centroid signatures for all discovered clusters.
- **Accordion Architecture**: Interactive drill-down into the 7 engineering challenges and mathematical solutions.
- **Glassmorphic UEFA Dark-Mode Theme**: Styled with modern typography (Outfit, JetBrains Mono) and dynamic responsive layouts.

> **How to view**: Double-click [`index.html`](index.html) or open it directly in any web browser!

---

## 📂 Repository Structure

```text
├── Data/
│   └── players_20.csv                            # Raw FIFA 20 player dataset (18,278 players, 104 attributes)
├── FIFA20_Player_Analysis_and_Clustering.ipynb   # Master end-to-end data science Jupyter Notebook
├── main.ipynb                                    # Analytical notebook workspace
├── main.py                                       # Python application entrypoint
├── index.html                                    # Interactive Executive Web Showcase & Visual Analytics Dashboard
├── pyproject.toml                                # Project build & dependency configuration
├── .gitignore                                    # Git ignore configuration
└── README.md                                     # Comprehensive technical documentation
```

---

## 💡 Task 3: Resolution of Core Business Questions

### Q1: Top 10 Player Producing Nations
*Which countries are producing the most footballers that play at this level?*

| Rank | Country | Player Count | % of Database | Mean Overall | Elite Representative |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **England** | 1,667 | 9.12% | 63.25 | H. Kane (89) |
| **2** | **Germany** | 1,216 | 6.65% | 65.94 | M. ter Stegen (90) |
| **3** | **Spain** | 1,035 | 5.66% | 70.14 | Sergio Ramos (89) |
| **4** | **France** | 984 | 5.38% | 67.82 | K. Mbappé (89) |
| **5** | **Argentina** | 886 | 4.85% | 67.70 | L. Messi (94) |
| **6** | **Brazil** | 824 | 4.51% | 71.16 | Neymar Jr (92) |
| **7** | **Italy** | 732 | 4.00% | 67.66 | G. Chiellini (89) |
| **8** | **Colombia** | 591 | 3.23% | 65.68 | J. Rodríguez (85) |
| **9** | **Japan** | 453 | 2.48% | 63.69 | M. Hasebe (79) |
| **10** | **Netherlands** | 438 | 2.40% | 68.32 | V. van Dijk (89) |

#### Structural Drivers:
- **European League Pyramids**: England, Germany, Spain, and France lead in sheer headcount due to deep tiered domestic leagues (Premier League through League Two; 1. & 2. Bundesliga; La Liga & Segunda) fully licensed in the database.
- **South American Export Powerhouses**: Argentina and Brazil represent the world's most productive exporter pipelines, maintaining high player counts and world-class elite talent density (Brazil leads with an average overall of 71.16).
- **Youth Training Academies**: Spain and France exhibit top-tier quality density driven by world-renowned youth centers (*La Masia*, *Clairefontaine*).

---

### Q2: Player Aging Curve & The Ceiling Horizon
*Plot the distribution of overall rating vs. age of players. What is the age after which a player stops improving?*

#### Empirical Findings:
- **Rapid Developmental Phase (Ages 16–22)**: Substantial headroom (+8.4 to +19.4 growth rating points).
- **Maturation Phase (Ages 23–26)**: Progressive consolidation (+2.3 to +6.9 growth rating points).
- **Peak Performance Plateau (Ages 27–29)**: Mean overall rating hits its statistical ceiling at **69.2 – 69.4**.
- **The Improvement Ceiling**:
  - At **Age 27**, average remaining growth drops below 1.0 (0.94 points).
  - At **Age 28**, remaining growth drops to **0.39 points**.
  - By **Age 29–30**, growth is negligible (< 0.15 points).
  - At **Age 31+**, growth potential is strictly **0.00 points**.

> 🎯 **Definitive Conclusion**: Empirically, professional footballers **STOP IMPROVING after age 28 to 29**. Beyond age 29, biological degradation in physical metrics (sprint speed, acceleration, agility) outpaces marginal improvements in tactical composure and positional awareness.

---

### Q3: Offensive Wage Economics (Mean vs. Median Paradox)
*Which type of offensive players tends to get paid the most: the striker, the right-winger, or the left-winger?*

#### Registered Position Wage Matrix:

| Position | Player Count | Mean Wage (€) | Median Wage (€) | 75th Pct (€) | Max Wage (€) | Highest Earner |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Right Winger (RW)** | 369 | **€15,848** | €3,000 | €12,000 | €565,000 | **L. Messi** (€565k/wk) |
| **Left Winger (LW)** | 378 | **€14,037** | €3,000 | €11,000 | €470,000 | **E. Hazard** (€470k/wk) |
| **Striker (ST/RS/LS)**| 2,582 | €10,153 | **€4,000 (+33%)**| €10,000 | €405,000 | **C. Ronaldo** (€405k/wk) |

#### Resolution of the Paradox:
1. **By Arithmetic Mean**: **Right-Wingers earn the most** (€15,848/wk), followed by Left-Wingers (€14,037/wk), with Strikers lowest (€10,153/wk).
2. **By Median (Typical Professional)**: **Strikers earn significantly more** (€4,000/wk vs. €3,000/wk).
3. **The Root Economic Causes**:
   - **Superstar Skewness Effect**: The world's top earners operate on the wings (Lionel Messi at RW earning €565k/wk; Eden Hazard at LW earning €470k/wk). These extreme statistical outliers pull the arithmetic mean up drastically.
   - **Lower-League Roster Volume Effect**: Strikers are universally required across every professional league tier (2,582 strikers vs. ~370 wingers). The heavy volume of lower-tier strikers dilutes the mean, while high demand elevates the baseline median wage.

---

## 🤖 Task 2: Feature Engineering & Machine Learning Clustering

### Data Cleaning & Positional String Parsing
- **Positional Modifier Strings**: Positional capability attributes (`st`, `rw`, `cm`, etc.) contained algebraic compound strings (e.g. `'89+2'`, `'74-1'`). A custom regex function `parse_positional_rating()` parsed operators and computed the true net numerical capability.
- **Work Rate Bilateral Decomposition**: Compound strings (`'High/Medium'`) were split into orthogonal numerical features: `AttackWorkRate` and `DefenseWorkRate` mapped ordinally ($0.0 = \text{Low}, 0.5 = \text{Medium}, 1.0 = \text{High}$).
- **Outfield Segmentation**: Goalkeepers (2,036) were partitioned away from outfield players (16,242) to prevent zero-variance athletic distortion.
- **Dimensionality Reduction**: Normalized via `StandardScaler`. Scree plot analysis of Principal Component Analysis (PCA) confirmed that **9 orthogonal principal components retain >80% of cumulative variance**.

### Target Leakage Prevention
Features like `overall` and `international_reputation` were **strictly excluded** from the clustering matrix. Including them would collapse Euclidean distance into trivial "good vs. bad" sorting rather than discovering genuine tactical playing styles.

---

### Unsupervised Model Benchmark

| Model Algorithm | Clusters ($k$) | Silhouette Score ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Fit Latency | Inference Complexity | Production Readiness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Mini-Batch K-Means** | **4** | **0.1622** | **4,366.0** | **1.7213** | **262.7 ms** | **$O(k \cdot d)$ - Constant** | **10 / 10 (Selected)** |
| **Standard K-Means** | 4 | 0.1638 | 4,384.0 | 1.7178 | 273.3 ms | $O(k \cdot d)$ - Constant | 9.5 / 10 |
| **Gaussian Mixture Model (GMM)** | 4 | 0.1534 | 4,299.0 | 1.7476 | 160.2 ms | $O(k \cdot d^2)$ - Moderate | 7.5 / 10 |
| **Agglomerative Hierarchical** | 4 | 0.1347 | 827.0 | 1.8743 | 537.2 ms | $O(N^2)$ - Prohibitive | 4.0 / 10 |

---

### Discovered Tactical Archetypes

The optimal 4-cluster model cleanly partitions outfield footballers into actionable tactical roles:

```text
       Cluster 3: Pure Clinical Strikers & Elite Creators (4,205 Players)
       [Finishing: 68.4 | Dribbling: 73.7 | Pace: 72.9 | Passing: 67.9]
       Key Examples: L. Messi (94), C. Ronaldo (93), Neymar Jr (92), K. De Bruyne (91)
                                      ▲
                                      │
  Cluster 0: Engine Midfielders       │       Cluster 2: Ball-Playing Anchors
  & Wingbacks (4,145 Players)         ┼────── & Box-to-Box (4,675 Players)
  [Pace: 70.9 | Dribbling: 64.5]      │       [Physical: 66.7 | Defending: 63.8]
                                      │
                                      ▼
       Cluster 1: Dominant Defensive Stoppers (3,217 Players)
       [Defending: 60.8 | Physical: 64.8 | Pace: 58.3 | Finishing: 33.7]
       Key Examples: K. Koulibaly (89), G. Chiellini (89), D. Godín (88), K. Manolas (85)
```

---

## 📊 Task 1: Exploratory Data Analysis & Domain Deep-Dives

1. **Messi vs. Ronaldo Skill Attribution**:
   - Lionel Messi dominates in **dribbling (96), vision (94), balance (95), and passing (92)**.
   - Cristiano Ronaldo dominates in **jumping (95), shot power (95), sprint speed (91), and strength (78)**.
2. **Ideal Squad Budget & Diminishing Returns Knee Point**:
   - Lineup optimization demonstrates that performance scaling flattens dramatically past an aggregate 11-man squad value of **€180M – €220M**. Beyond this threshold, additional capital yields minimal rating gains.
3. **Top 5% Elite Footballer Profiling**:
   - The top 5% of players exhibit statistically massive effect sizes in **ball control ($d = 2.45$), composure ($d = 2.38$), and reactions ($d = 2.31$)** compared to standard professionals.

---

## ⚙️ 7 Engineering Challenges & Technical Solutions

| # | Challenge Encountered | Technical Root Cause | Engineered Solution |
| :---: | :--- | :--- | :--- |
| **1** | **Goalkeeper Skill Disparity** | Missing/zero values in technical outfield skills distort distances. | **Structural Dataset Partitioning**: Isolated 2,036 GKs; clustered 16,242 outfielders. |
| **2** | **Algebraic Positional Strings** | Strings like `'89+2'` represent base + live in-form modifiers. | **Vectorized Regex Evaluation**: Custom arithmetic reduction to net float ratings. |
| **3** | **Target Leakage / Dominance** | `overall` rating artificially forces clusters into simple quality tiers. | **Feature Space Isolation**: Excluded overall from clustering matrix $X$. |
| **4** | **Compound Work Rates** | Formats like `'High/Medium'` bundle attack and defense behaviors. | **Bilateral Ordinal Mapping**: Decomposed into separate attack/defense scales (0, 0.5, 1.0). |
| **5** | **High Multicollinearity** | 37 features exhibit high collinearity (tackles, passes $r > 0.85$). | **StandardScaler + PCA**: 9 orthogonal principal components explain >80% variance. |
| **6** | **Unsupervised Ground Truth Absence**| No labels exist to calculate supervised metrics (accuracy/ROC). | **Multi-Metric Triangulation**: Evaluated Silhouette, CH, DB, and domain centroids. |
| **7** | **Heavy-Tailed Wage Skewness** | Power-law distribution with superstar outliers (€565k vs €4k median). | **Dual Parametric Reporting**: Reported both Mean and Median to explain wage divergence. |

---

## 🚀 Production Deployment & Strategic Recommendations

### Why Mini-Batch K-Means Wins in Production:
1. **$O(k \cdot d)$ Inference Latency**: Assigning a player to their archetype requires calculating distance against only 4 centroid vectors of length 37 (sub-millisecond execution).
2. **Minimal Memory Footprint**: Stores only a $4 \times 37$ floating-point matrix, making it suitable for edge deployment or mobile scouting apps.
3. **Streaming Online Learning**: Compatible with partial-fit mini-batches as new scouting data is streamed weekly.

### Recruitment Strategy Playbook:
- **Optimal Acquisition Age Window**: Acquire players between **ages 18 and 22** where remaining growth headroom (+8 to +19) maximizes capital appreciation.
- **Contract Renewal Warning**: Avoid offering long-term max contracts extending beyond **age 29**, where physical decline outpaces technical gains.
- **Winger Compensation Discipline**: Do not use arithmetic mean wages to set winger compensation benchmarks. Wingers are skewed by elite outliers; median striker value is higher across competitive leagues.

---

## 🛠️ Getting Started & Installation

### Prerequisites
- Python 3.9+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### 1. Clone the Repository
```bash
git clone https://github.com/babiazees007/FIFA_20_Player_Skill_analysis.git
cd FIFA_20_Player_Skill_analysis
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook FIFA20_Player_Analysis_and_Clustering.ipynb
```

### 4. Launch the Interactive Dashboard
Simply open `index.html` in any browser:
```powershell
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments
- **Project Lead**: Babi Azees ([@babiazees007](https://github.com/babiazees007))
- **Data Source**: Electronic Arts FIFA 20 Database / Kaggle
- **Project Code**: `PRCP-1004`
