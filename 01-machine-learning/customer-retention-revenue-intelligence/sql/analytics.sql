-- QUERY: monthly_revenue
SELECT date_trunc('month', invoice_date)::DATE AS month,
       SUM(line_revenue) AS revenue_gbp,
       COUNT(DISTINCT invoice_no) AS invoices,
       COUNT(DISTINCT customer_id) AS identified_customers
FROM transactions WHERE is_valid_purchase
GROUP BY 1 ORDER BY 1;

-- QUERY: country_revenue
SELECT country, SUM(line_revenue) AS revenue_gbp,
       COUNT(DISTINCT invoice_no) AS invoices,
       COUNT(DISTINCT customer_id) AS customers
FROM transactions WHERE is_valid_purchase
GROUP BY country ORDER BY revenue_gbp DESC;

-- QUERY: cancellations
SELECT date_trunc('month', invoice_date)::DATE AS month,
       COUNT(*) AS cancellation_lines,
       SUM(ABS(line_revenue)) AS cancellation_value_gbp
FROM transactions WHERE is_cancellation
GROUP BY 1 ORDER BY 1;

-- QUERY: rfm_segments
WITH customer_rfm AS (
    SELECT customer_id,
           date_diff('day', MAX(invoice_date), (SELECT MAX(invoice_date) FROM valid_sales)) AS recency_days,
           COUNT(DISTINCT invoice_no) AS frequency,
           SUM(line_revenue) AS monetary_gbp
    FROM valid_sales WHERE customer_id IS NOT NULL GROUP BY customer_id
),
scored AS (
    SELECT *,
           NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
           NTILE(5) OVER (ORDER BY frequency) AS f_score,
           NTILE(5) OVER (ORDER BY monetary_gbp) AS m_score
    FROM customer_rfm
)
SELECT CASE
    WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
    WHEN f_score >= 4 AND m_score >= 3 THEN 'Loyal high value'
    WHEN r_score >= 4 AND f_score <= 2 THEN 'Recent customers'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At risk'
    WHEN r_score <= 2 AND f_score <= 2 THEN 'Hibernating'
    ELSE 'Potential loyalists' END AS segment,
    COUNT(*) AS customers, AVG(monetary_gbp) AS avg_customer_revenue_gbp
FROM scored GROUP BY 1 ORDER BY customers DESC;

-- QUERY: cohort_retention
WITH first_orders AS (
    SELECT customer_id, date_trunc('month', MIN(invoice_date))::DATE AS cohort_month
    FROM valid_sales WHERE customer_id IS NOT NULL GROUP BY customer_id
),
customer_months AS (
    SELECT DISTINCT customer_id, date_trunc('month', invoice_date)::DATE AS order_month
    FROM valid_sales WHERE customer_id IS NOT NULL
),
cohorts AS (
    SELECT f.cohort_month,
           date_diff('month', f.cohort_month, m.order_month) AS month_number,
           COUNT(DISTINCT m.customer_id) AS active_customers
    FROM first_orders f JOIN customer_months m USING (customer_id)
    WHERE m.order_month >= f.cohort_month
    GROUP BY 1, 2
)
SELECT cohort_month, month_number, active_customers,
       FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY month_number) AS cohort_size,
       active_customers::DOUBLE / NULLIF(FIRST_VALUE(active_customers) OVER (
           PARTITION BY cohort_month ORDER BY month_number), 0) AS retention_rate
FROM cohorts ORDER BY cohort_month, month_number;

-- QUERY: revenue_concentration
WITH customer_revenue AS (
    SELECT customer_id, SUM(line_revenue) AS revenue_gbp
    FROM valid_sales WHERE customer_id IS NOT NULL GROUP BY customer_id
),
ranked AS (
    SELECT customer_id, revenue_gbp,
           ROW_NUMBER() OVER (ORDER BY revenue_gbp DESC) AS customer_rank,
           SUM(revenue_gbp) OVER () AS total_revenue,
           SUM(revenue_gbp) OVER (ORDER BY revenue_gbp DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
    FROM customer_revenue
)
SELECT customer_id, revenue_gbp, customer_rank,
       cumulative_revenue / NULLIF(total_revenue, 0) AS cumulative_revenue_share
FROM ranked ORDER BY customer_rank;
