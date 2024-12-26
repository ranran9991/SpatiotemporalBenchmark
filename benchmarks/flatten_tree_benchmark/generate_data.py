import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import pyarrow.parquet as pq

# Parameters
n_types = 7
node_counts = [1, 5, 25, 50, 150, 450, 1000]  # Number of nodes for each type
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)
num_updates = 500000

# Generate nodes for each type
nodes = []
for t, count in enumerate(node_counts, start=1):
    for _ in range(count):
        nodes.append((str(uuid.uuid4()), f"t{t}"))

# Build the parent-child relationships
parent_map = {}
current_index = 0
for t, count in enumerate(node_counts[:-1]):
    parents = [node[0] for node in nodes[current_index:current_index + count]]
    children = nodes[current_index + count:current_index + count + node_counts[t + 1]]
    for child_id, child_type in children:
        parent_map[child_id] = np.random.choice(parents)
    current_index += count


# Generate updates
def generate_updates():
    updates = [[
        str(uuid.uuid4()),
        nodes[0][0],
        nodes[0][1],
        None,
        None,
        0
    ]]
    current_time = start_date

    for _ in range(num_updates):
        child_id, child_type = nodes[np.random.randint(1, len(nodes))]  # Exclude the root node
        father_id = parent_map[child_id]
        father_type = next(node[1] for node in nodes if node[0] == father_id)
        update_id = str(uuid.uuid4())
        update_time = int(current_time.timestamp() * 1000)  # Milliseconds

        updates.append([
            update_id,
            child_id,
            child_type,
            father_id,
            father_type,
            update_time
        ])

        # Randomize next update time
        current_time += timedelta(seconds=np.random.randint(1, 60))

    return updates


updates = generate_updates()
df = pd.DataFrame(updates, columns=["update_id", "child_id", "child_type", "father_id", "father_type", "update_time"])

# Save to parquet file
df.to_parquet("data/tree_updates.parquet", engine="pyarrow")

print(f"Generated {len(df)} updates and saved to tree_updates.parquet.")
