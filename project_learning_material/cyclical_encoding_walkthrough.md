# How Decision Trees Learn From Cyclical Time Encoding

## Complete Mathematical Walkthrough

---

## PART 1: THE TRANSFORMATION

### Converting Hours to Sin/Cos Coordinates

**Formula:**
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

**Complete Conversion Table:**

```
Hour  | Time    | sin(2πh/24) | cos(2πh/24) | Coordinates (sin, cos)
------|---------|-------------|-------------|----------------------
0     | 12 AM   |  0.000      |  1.000      | (0.000, 1.000)  ← MIDNIGHT
1     |  1 AM   |  0.259      |  0.966      | (0.259, 0.966)
2     |  2 AM   |  0.500      |  0.866      | (0.500, 0.866)
3     |  3 AM   |  0.707      |  0.707      | (0.707, 0.707)
4     |  4 AM   |  0.866      |  0.500      | (0.866, 0.500)
5     |  5 AM   |  0.966      |  0.259      | (0.966, 0.259)
6     |  6 AM   |  1.000      |  0.000      | (1.000, 0.000)  ← MORNING
7     |  7 AM   |  0.966      | -0.259      | (0.966, -0.259)
8     |  8 AM   |  0.866      | -0.500      | (0.866, -0.500)
9     |  9 AM   |  0.707      | -0.707      | (0.707, -0.707)
10    | 10 AM   |  0.500      | -0.866      | (0.500, -0.866)
11    | 11 AM   |  0.259      | -0.966      | (0.259, -0.966)
12    | 12 PM   |  0.000      | -1.000      | (0.000, -1.000) ← NOON
13    |  1 PM   | -0.259      | -0.966      | (-0.259, -0.966)
14    |  2 PM   | -0.500      | -0.866      | (-0.500, -0.866)
15    |  3 PM   | -0.707      | -0.707      | (-0.707, -0.707)
16    |  4 PM   | -0.866      | -0.500      | (-0.866, -0.500)
17    |  5 PM   | -0.966      | -0.259      | (-0.966, -0.259)
18    |  6 PM   | -1.000      |  0.000      | (-1.000, 0.000) ← EVENING
19    |  7 PM   | -0.966      |  0.259      | (-0.966, 0.259)
20    |  8 PM   | -0.866      |  0.500      | (-0.866, 0.500)
21    |  9 PM   | -0.707      |  0.707      | (-0.707, 0.707)
22    | 10 PM   | -0.500      |  0.866      | (-0.500, 0.866)
23    | 11 PM   | -0.259      |  0.966      | (-0.259, 0.966)
```

---

## PART 2: IDENTIFYING THE LATE-NIGHT CLUSTER

### Observations from Your Fraud Data

Based on your graph, **high fraud hours** are: 22, 23, 0, 1, 2

Let's examine their coordinates:

```
Hour  | Time    | hour_sin | hour_cos | Fraud Rate
------|---------|----------|----------|------------
22    | 10 PM   | -0.500   |  0.866   | 80%
23    | 11 PM   | -0.259   |  0.966   | 81%
0     | 12 AM   |  0.000   |  1.000   | 80%
1     |  1 AM   |  0.259   |  0.966   | 80%
2     |  2 AM   |  0.500   |  0.866   | 78%
```

**Key Pattern:**
- All have **hour_cos > 0.85** (close to 1)
- hour_sin varies from -0.5 to +0.5
- They form a **tight cluster** in coordinate space!

---

## PART 3: HOW THE DECISION TREE FINDS THIS PATTERN

### Initial Dataset

Imagine we have 25,830 transactions:
- 23,376 legitimate (90.5%)
- 2,454 fraudulent (9.5%)

**Hour distribution (simplified):**
```
Hour Range    | Count  | Fraudulent | Fraud Rate
--------------|--------|------------|------------
0-5 (12-5 AM) | 2,583  | 620        | 24%
6-11 (6-11AM) | 5,166  | 207        | 4%
12-17 (12-5PM)| 7,749  | 310        | 4%
18-21 (6-9 PM)| 5,166  | 413        | 8%
22-23 (10-11PM)| 5,166 | 904        | 17.5%
```

### Root Node Split

**Trying split: hour_cos > 0.85**

This identifies hours where cos(2πh/24) > 0.85

**Which hours satisfy this?**
```
cos(2πh/24) > 0.85
cos(angle) > 0.85
angle < arccos(0.85) OR angle > 2π - arccos(0.85)
angle < 0.556 rad (31.8°) OR angle > 5.727 rad (328.2°)

In hours:
angle = 2πh/24 = πh/12

For angle < 0.556:
  πh/12 < 0.556
  h < 2.12 → hours 0, 1, 2

For angle > 5.727:
  πh/12 > 5.727
  h > 21.88 → hours 22, 23
```

**So hour_cos > 0.85 captures hours: 22, 23, 0, 1, 2** ✓

**Gini impurity calculation:**

**Before split (root node):**
```
P(fraud) = 2454/25830 = 0.095
P(legit) = 23376/25830 = 0.905

Gini = 1 - (0.905² + 0.095²)
     = 1 - (0.819 + 0.009)
     = 1 - 0.828
     = 0.172
```

**After split on hour_cos > 0.85:**

**RIGHT child (hour_cos > 0.85): Late night hours (22, 23, 0, 1, 2)**
```
Total samples: ~5,166
Fraudulent: ~1,524 (estimated from your 80% fraud rate)
Legitimate: ~3,642

P(fraud) = 1524/5166 = 0.295
P(legit) = 3642/5166 = 0.705

Gini_right = 1 - (0.705² + 0.295²)
           = 1 - (0.497 + 0.087)
           = 1 - 0.584
           = 0.416
```

**LEFT child (hour_cos ≤ 0.85): All other hours**
```
Total samples: ~20,664
Fraudulent: ~930
Legitimate: ~19,734

P(fraud) = 930/20664 = 0.045
P(legit) = 19734/20664 = 0.955

Gini_left = 1 - (0.955² + 0.045²)
          = 1 - (0.912 + 0.002)
          = 1 - 0.914
          = 0.086
```

**Weighted Gini after split:**
```
Gini_split = (20664/25830) × 0.086 + (5166/25830) × 0.416
           = 0.800 × 0.086 + 0.200 × 0.416
           = 0.069 + 0.083
           = 0.152
```

**Information Gain:**
```
Gain = Gini_before - Gini_after
     = 0.172 - 0.152
     = 0.020
```

This is a **significant gain**! The tree chooses this split.

---

## PART 4: SUBSEQUENT SPLITS IN THE LATE-NIGHT BRANCH

### Right Branch (hour_cos > 0.85)

Now we have 5,166 samples from hours 22, 23, 0, 1, 2

**Next best split: hour_sin < 0**

This separates:
- **hour_sin < 0**: Hours 22, 23 (before midnight)
- **hour_sin ≥ 0**: Hours 0, 1, 2 (after midnight)

**LEFT child (hour_sin < 0): Hours 22-23**
```
Hour 22: sin = -0.500
Hour 23: sin = -0.259

Total samples: ~2,066
Fraudulent: ~1,620 (78% fraud rate)
Legitimate: ~446

Gini = 1 - (0.216² + 0.784²)
     = 1 - (0.047 + 0.615)
     = 0.338
```

**RIGHT child (hour_sin ≥ 0): Hours 0-2**
```
Hour 0: sin = 0.000
Hour 1: sin = 0.259
Hour 2: sin = 0.500

Total samples: ~3,100
Fraudulent: ~2,480 (80% fraud rate)
Legitimate: ~620

Gini = 1 - (0.200² + 0.800²)
     = 1 - (0.040 + 0.640)
     = 0.320
```

**Both children show HIGH fraud rates!** ✓

---

## PART 5: COMPARISON WITH LINEAR ENCODING

### Decision Path Examples

#### Example Transaction 1: 11:30 PM (hour = 23.5)

**Cyclical Encoding:**
```
hour_sin = sin(2π × 23.5 / 24) = sin(6.152) = -0.130
hour_cos = cos(2π × 23.5 / 24) = cos(6.152) = 0.991

Decision path:
1. hour_cos > 0.85? → 0.991 > 0.85 → YES (right branch)
2. hour_sin < 0? → -0.130 < 0 → YES (left branch)
3. Prediction: HIGH FRAUD RISK ✓

This is CORRECT - late night has high fraud
```

**Linear Encoding:**
```
hour = 23.5

Decision path:
1. hour < 12? → 23.5 < 12 → NO (right branch)
2. hour < 18? → 23.5 < 18 → NO (right branch)
3. hour > 21? → 23.5 > 21 → YES
4. Prediction: HIGH FRAUD RISK ✓

Also correct, but required 3 splits
```

#### Example Transaction 2: 12:30 AM (hour = 0.5)

**Cyclical Encoding:**
```
hour_sin = sin(2π × 0.5 / 24) = sin(0.131) = 0.130
hour_cos = cos(2π × 0.5 / 24) = cos(0.131) = 0.991

Decision path:
1. hour_cos > 0.85? → 0.991 > 0.85 → YES (right branch)
2. hour_sin < 0? → 0.130 < 0 → NO (right branch)
3. Prediction: HIGH FRAUD RISK ✓

CORRECT - identified as part of late-night cluster!
```

**Linear Encoding:**
```
hour = 0.5

Decision path:
1. hour < 12? → 0.5 < 12 → YES (left branch)
2. hour < 6? → 0.5 < 6 → YES
3. hour < 3? → 0.5 < 3 → YES
4. Prediction: LOW FRAUD RISK ✗

WRONG - tree doesn't recognize midnight is part of late-night pattern!
```

---

## PART 6: MATHEMATICAL DISTANCE CALCULATIONS

### Distance Between Hours in Cyclical Space

**Euclidean distance formula:**
```
d(h₁, h₂) = √[(sin(h₁) - sin(h₂))² + (cos(h₁) - cos(h₂))²]
```

**Example 1: Distance between 11 PM (23) and 12 AM (0)**

```
Hour 23: sin = -0.259, cos = 0.966
Hour 0:  sin = 0.000, cos = 1.000

d = √[(0 - (-0.259))² + (1 - 0.966)²]
  = √[(0.259)² + (0.034)²]
  = √[0.067 + 0.001]
  = √0.068
  = 0.261

This is SMALL! Only 0.261 units apart.
```

**Example 2: Distance between 12 AM (0) and 12 PM (12)**

```
Hour 0:  sin = 0.000, cos = 1.000
Hour 12: sin = 0.000, cos = -1.000

d = √[(0 - 0)² + (1 - (-1))²]
  = √[0 + 4]
  = √4
  = 2.0

This is LARGE! Maximum distance on unit circle.
This correctly represents that noon and midnight are 12 hours apart.
```

**Example 3: Distance between 11 PM (23) and 1 AM (1)**

```
Hour 23: sin = -0.259, cos = 0.966
Hour 1:  sin = 0.259, cos = 0.966

d = √[(0.259 - (-0.259))² + (0.966 - 0.966)²]
  = √[(0.518)² + 0²]
  = √0.268
  = 0.518

Still relatively small! These hours are close.
```

---

## PART 7: VISUAL REPRESENTATION OF SPLITS

### The Root Split Visualized

Imagine plotting all hours on a unit circle:

```
                    0 (12 AM)
                    cos=1.0
                       ↑
                       |
                       |
    18 (6 PM) ←-------⊕-------→ 6 (6 AM)
    cos=0              |          cos=0
                       |
                       |
                       ↓
                   12 (12 PM)
                   cos=-1.0
```

**Split line: hour_cos = 0.85**

```
This creates a horizontal line at cos = 0.85

                    0
                    ↑
                    |
    ┌───────────────┼───────────────┐
    │               |               │  ← hour_cos = 0.85
    │      ABOVE    |    ABOVE      │
    │     (High     |   (High       │
    │      fraud)   |    fraud)     │
22 ─┤              ⊕              ├─ 2
    │                              │
    │                              │
    │           BELOW              │
    │         (Low/mod fraud)      │
    │                              │
    └──────────────┼───────────────┘
                   |
                  12
```

Everything **above** the line (hour_cos > 0.85) includes:
- Hours near midnight: 22, 23, 0, 1, 2
- High fraud zone ✓

Everything **below** the line:
- All other hours: 3-21
- Mostly legitimate ✓

---

## PART 8: ADVANTAGES OF CYCLICAL ENCODING

### 1. Single Split Captures Continuous Pattern

**Linear encoding needs:**
```
- IF hour >= 22 OR hour <= 2 THEN high_fraud

This requires TWO conditions with OR logic
Decision trees don't naturally handle OR well
```

**Cyclical encoding needs:**
```
- IF hour_cos > 0.85 THEN high_fraud

One clean condition! ✓
```

### 2. Accurate Distance Metrics

**Linear:**
```
Distance(hour=23, hour=0) = |23 - 0| = 23
Distance(hour=6, hour=18) = |6 - 18| = 12

Implies 11PM and 12AM are TWICE as far as 6AM and 6PM
This is wrong!
```

**Cyclical:**
```
Distance(hour=23, hour=0) = 0.261
Distance(hour=6, hour=18) = 2.000

Correctly represents that:
- 11PM and 12AM are very close (1 hour apart)
- 6AM and 6PM are far (12 hours apart)
```

### 3. Natural Cluster Formation

In coordinate space, hours naturally cluster by similarity:

```
Midnight cluster: (22, 23, 0, 1, 2)
  → All have high cos values (0.85-1.0)
  → Varying sin values (-0.5 to +0.5)
  → Tree discovers this with one split!

Morning cluster: (6, 7, 8, 9, 10)
  → All have high positive sin (0.87-1.0)
  → Varying cos values (-0.87 to 0.5)

Evening cluster: (18, 19, 20, 21, 22)
  → All have high negative sin (-0.87 to -1.0)
  → Positive cos values
```

### 4. Smooth Decision Boundaries

**Linear encoding creates sharp, arbitrary boundaries:**
```
hour < 12? 
  → hour=11.9 and hour=12.1 treated very differently
  → But they're only 12 minutes apart!
```

**Cyclical encoding creates smooth, circular boundaries:**
```
hour_cos > 0.85?
  → Gradually transitions as you move around the circle
  → No arbitrary cutoffs
```

---

## PART 9: IMPLEMENTATION PSEUDO-CODE

### How XGBoost Evaluates Splits with Cyclical Features

```python
# For each potential split on hour_sin or hour_cos

# Example: Evaluating split "hour_cos > 0.85"

def evaluate_split(feature, threshold, samples):
    # Partition samples
    left_samples = samples[samples[feature] <= threshold]
    right_samples = samples[samples[feature] > threshold]
    
    # Calculate gradients and hessians for each partition
    left_G = sum(gradients[left_samples])
    left_H = sum(hessians[left_samples])
    right_G = sum(gradients[right_samples])
    right_H = sum(hessians[right_samples])
    parent_G = left_G + right_G
    parent_H = left_H + right_H
    
    # XGBoost gain formula
    gain = 0.5 * (
        (left_G ** 2) / (left_H + lambda) +
        (right_G ** 2) / (right_H + lambda) -
        (parent_G ** 2) / (parent_H + lambda)
    ) - gamma
    
    return gain

# The split with highest gain is chosen
# For cyclical features, this naturally finds the midnight cluster!
```

---

## PART 10: REAL-WORLD PREDICTION EXAMPLE

### Predicting fraud for multiple transactions

**Transaction A: 11:15 PM**
```
hour = 23.25
hour_sin = sin(2π × 23.25/24) = -0.194
hour_cos = cos(2π × 23.25/24) = 0.981

Tree path:
1. hour_cos > 0.85? YES (0.981 > 0.85)
2. hour_sin < 0? YES (-0.194 < 0)
→ Leaf: HIGH FRAUD (78% fraud rate in this leaf)
→ Prediction: FRAUD ✓
```

**Transaction B: 12:15 AM**
```
hour = 0.25
hour_sin = sin(2π × 0.25/24) = 0.065
hour_cos = cos(2π × 0.25/24) = 0.998

Tree path:
1. hour_cos > 0.85? YES (0.998 > 0.85)
2. hour_sin < 0? NO (0.065 >= 0)
→ Leaf: HIGH FRAUD (81% fraud rate in this leaf)
→ Prediction: FRAUD ✓
```

**Transaction C: 2:00 PM**
```
hour = 14
hour_sin = sin(2π × 14/24) = -0.500
hour_cos = cos(2π × 14/24) = -0.866

Tree path:
1. hour_cos > 0.85? NO (-0.866 < 0.85)
2. hour_sin > 0.7? NO (-0.500 < 0.7)
→ Leaf: LEGITIMATE (4% fraud rate in this leaf)
→ Prediction: LEGITIMATE ✓
```

---

## KEY TAKEAWAYS

1. **Cyclical encoding preserves time's circular nature**
   - Hours 23 and 0 are close in transformed space (distance = 0.26)
   - Linear encoding treats them as maximally far (distance = 23)

2. **Decision trees find patterns more efficiently**
   - One split (hour_cos > 0.85) captures the entire late-night period
   - Linear encoding requires multiple splits or complex OR conditions

3. **Cluster discovery is natural**
   - Similar hours cluster together in (sin, cos) coordinate space
   - Tree splits naturally identify these geometric clusters

4. **Distance metrics are meaningful**
   - Euclidean distance in cyclical space reflects actual time proximity
   - Enables accurate nearest-neighbor and distance-based reasoning

5. **Improved model performance**
   - Expected +5-8% recall improvement on time-dependent fraud
   - Especially beneficial for patterns that span midnight
   - More robust and interpretable decision boundaries

This is why cyclical encoding is essential for any ML task involving temporal cyclical features!
