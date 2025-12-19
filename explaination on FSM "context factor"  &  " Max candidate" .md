Here is a detailed explanation of how the **Context Factor** and **Max Candidates** parameters control the Full Subsequence Matching (FSM) algorithm in your application.

### 1. Context Factor (`m_factor`)

This parameter determines **how much history** the algorithm looks at before a gap to decide what "pattern" to search for.

* **The Logic:**
The size of the search window () is calculated as:


* **Example:**
Imagine you have a **missing gap of 6 hours**.
* **If Factor = 1.0 (Default):** The algorithm looks at the **6 hours** immediately before the gap. It says, *"Find me another time in history where the water level looked exactly like these last 6 hours."*
* **If Factor = 2.0:** The algorithm looks at the **12 hours** before the gap. It effectively tries to match a longer trend.
* **If Factor = 0.5:** The algorithm looks at only the **3 hours** before the gap.



#### **How it effects the calculation:**

| Setting | Effect on Imputation | Best Used For |
| --- | --- | --- |
| **Higher (e.g., 2.0)** | **More Specific.** It enforces a stricter match. It ensures the *trend* (e.g., slowly rising water) matches perfectly. However, if the pattern is too unique, it might not find *any* good match. | Long gaps or complex flood events where trend is critical. |
| **Lower (e.g., 0.5)** | **More Flexible.** It matches the immediate level but ignores the longer trend. It will find many matches, but they might lead to different outcomes. | Short gaps or erratic/noisy data. |
| **1.0 (Balanced)** | **Standard.** The window context is the same size as the missing data. | General usage. |

---

### 2. Max Candidates

This parameter controls **speed vs. accuracy** by limiting how many historical windows the algorithm checks.

* **The Logic:**
A file with 5 years of hourly data has about **43,000** potential starting points (candidates) for a pattern match. Comparing your gap against *every single one* of them involves heavy math (Euclidean distance calculations) and can freeze the app for 10-20 seconds per gap.
* **What it does:**
If you set **Max Candidates = 5,000**, the algorithm randomly picks 5,000 historical windows to check and ignores the rest. It picks the best match from that random sample.

#### **How it effects the calculation:**

| Setting | Effect on Imputation | Trade-off |
| --- | --- | --- |
| **High (e.g., 50,000)** | **Maximum Accuracy.** It scans almost the entire file. It guarantees finding the absolute best mathematical match available in your history. | **Slower.** The app might lag if you have many gaps. |
| **Low (e.g., 1,000)** | **High Speed.** The calculation is nearly instant. | **Lower Accuracy.** It might miss the "perfect" match because it didn't look at that specific part of the file. |

### Summary Recommendation

* **For most river analysis:** Keep **Context Factor = 1.0**. This is the standard in hydrological literature (often referred to as ).
* **For huge files (10+ years):** Reduce **Max Candidates** to **5,000** to keep the app fast.
* **For short files (1 year):** Increase **Max Candidates** to **10,000+** to ensure you scan the whole file for the best possible accuracy.
