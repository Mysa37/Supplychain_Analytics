import pandas as pd
import numpy as np

np.random.seed(42)

# Parameters
start_date = "2022-01-01"
end_date = "2023-12-31"
dates = pd.date_range(start=start_date, end=end_date, freq='D')

num_products = 100
warehouses = ['WH_A', 'WH_B', 'WH_C']
categories = ['Beverages', 'Snacks', 'Personal Care', 'Household']
regions = ['North', 'South', 'East', 'West']

data = []

for product_id in range(1, num_products + 1):
    base_demand = np.random.randint(20, 200)
    trend = np.random.uniform(-0.05, 0.05)
    category = np.random.choice(categories)
    unit_cost = np.random.uniform(5, 50)
    margin = np.random.uniform(0.1, 0.4)
    selling_price = unit_cost * (1 + margin)

    for warehouse in warehouses:
        inventory_level = np.random.randint(500, 2000)

        for date in dates:
            day_of_year = date.dayofyear
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)

            weekly_pattern = 1.1 if date.weekday() in [5, 6] else 0.9

            noise = np.random.normal(0, 10)

            demand = base_demand * seasonal_factor * weekly_pattern
            demand = demand + (trend * day_of_year) + noise
            demand = max(0, int(demand))

            # Random demand shock
            if np.random.rand() < 0.01:
                demand *= np.random.randint(2, 4)

            lead_time = np.random.randint(3, 10)
            transportation_cost = np.random.uniform(1, 5)

            order_quantity = max(0, demand - inventory_level)

            inventory_level = max(0, inventory_level - demand)

            data.append([
                date,
                f"P{product_id}",
                category,
                warehouse,
                np.random.choice(regions),
                demand,
                inventory_level,
                lead_time,
                unit_cost,
                selling_price,
                order_quantity,
                transportation_cost
            ])

columns = [
    "Date",
    "Product_ID",
    "Category",
    "Warehouse",
    "Region",
    "Demand",
    "Inventory_Level",
    "Lead_Time",
    "Unit_Cost",
    "Selling_Price",
    "Order_Quantity",
    "Transportation_Cost"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("supply_chain_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())