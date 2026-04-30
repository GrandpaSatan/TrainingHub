Testing a custom model on mathematical reasoning requires a mix of benchmarks that target different levels of complexity, from basic arithmetic to competitive problem-solving. Running these locally is best handled using evaluation frameworks like **LM Evaluation Harness** or **Inspect**, which support many of these datasets out of the box.

### 1. Fundamental & Scholastic Benchmarks
These datasets test the model's ability to handle word problems and multi-step reasoning typically found in grade-school and high-school curricula.

* **GSM8K (Grade School Math 8K):** The gold standard for high-quality grade-school math word problems. It requires multi-step reasoning (Chain of Thought) to reach a final answer.
* **MATH:** A significantly more difficult dataset consisting of 12,500 problems from high school math competitions. It covers Algebra, Calculus, and Number Theory.
* **ASDiv (Academic Scholastic Dictionary):** Focuses on diverse language patterns in math problems to ensure the model isn't just "memorizing" specific phrasing.

### 2. Specialized & Logic-Heavy Benchmarks
If you want to push the model toward formal logic or university-level STEM, these are essential:

* **GSM-Symbolic:** A variant of GSM8K that uses templates to generate new problems with different numbers and names. This prevents "data contamination" (the model memorizing the original GSM8K answers from its training data).
* **SAT Math:** Problems sourced from SAT practice materials to gauge standardized testing performance.
* **TheoremQA:** A benchmark that tests the model’s ability to apply specific mathematical theorems and formulas to solve problems in science and engineering.

---

### 3. Technical Implementation (Local Evaluation)
To run these locally without writing custom scrapers for every dataset, use one of the following industry-standard tools:

| Tool | Best For | Description |
| :--- | :--- | :--- |
| **LM Evaluation Harness** | Breadth | The most popular tool. Supports GSM8K and MATH natively with simple CLI commands. |
| **OLMo-Eval / Inspect** | Precision | Great for tracking "Chain of Thought" accuracy and more granular logging of where the model failed. |
| **LightEval** | Hugging Face Users | A lightweight framework by Hugging Face specifically designed for model suites like LLM-foundry. |

---

### 4. Key Metrics to Track
When evaluating math, "Accuracy" isn't the only metric. Consider measuring:

* **Pass@1:** The percentage of problems the model gets right on the first try.
* **Maj@k (Majority Vote):** Generating $k$ answers (e.g., 5 or 10) and taking the most common result. This often boosts scores for Reasoning (CoT) models.
* **Strict vs. Flexible Matching:** Use regex to extract the numerical answer from the model's text response to ensure you aren't penalizing it for extra "chatter" in the output.

### 5. Recommendation for Local Testing
If you are just starting, begin with **GSM8K**. It is computationally light and there is a massive amount of baseline data for other models to compare your results against. If your model clears **80-90%** on GSM8K, move immediately to the **MATH** benchmark to find the actual ceiling of its capabilities.

