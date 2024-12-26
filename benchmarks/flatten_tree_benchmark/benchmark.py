import pandas as pd
import duckdb
from timeit import default_timer as timer
import numpy as np
from typing import Dict, List, Tuple


def process_tree_updates(input_file, output_file):
    df = pd.read_parquet(input_file)
    # Sort by update_time once at the start
    df = df.sort_values('update_time').reset_index(drop=True)
    # Pre-allocate result arrays for better performance
    n_rows = len(df)
    hierarchy_ids = np.empty(n_rows, dtype=object)
    hierarchy_types = np.empty(n_rows, dtype=object)

    # Dictionaries to cache current state
    # Using dict instead of defaultdict for better performance
    current_paths: Dict[str, List[str]] = {}  # node -> path of IDs
    current_types: Dict[str, List[str]] = {}  # node -> path of types
    latest_parent: Dict[str, str] = {}  # node -> immediate parent
    node_type_cache: Dict[str, str] = {}  # node -> its current type

    def update_node_type(node_id: str, node_type: str, as_father: bool = False) -> None:
        """Update the cached type for a node"""
        if as_father:
            node_type_cache[node_id] = node_type
        elif node_id not in node_type_cache:
            node_type_cache[node_id] = node_type

    def get_path(node_id: str, update_idx: int) -> Tuple[List[str], List[str]]:
        """Get or build path for a node, utilizing cached paths where possible"""
        if node_id not in latest_parent:
            # New node, create new path
            path_ids = [node_id]
            path_types = [node_type_cache.get(node_id, 'unknown')]
            return path_ids, path_types

        parent_id = latest_parent[node_id]
        if parent_id is None:
            # Root node
            path_ids = [node_id]
            path_types = [node_type_cache.get(node_id, 'unknown')]
            return path_ids, path_types

        # Get parent's path from cache and extend it
        parent_path_ids = current_paths.get(parent_id, [parent_id])
        parent_path_types = current_types.get(parent_id, [node_type_cache.get(parent_id, 'unknown')])

        return parent_path_ids + [node_id], parent_path_types + [node_type_cache[node_id]]

    # Process rows in chronological order
    for idx, row in enumerate(df.itertuples()):
        child_id = row.child_id
        father_id = row.father_id

        # Update type caches
        update_node_type(child_id, row.child_type)
        if father_id is not None:
            update_node_type(father_id, row.father_type, as_father=True)

        # Update parent relationship
        latest_parent[child_id] = father_id

        # Build and cache new path
        path_ids, path_types = get_path(child_id, idx)
        current_paths[child_id] = path_ids
        current_types[child_id] = path_types

        # Store results
        hierarchy_ids[idx] = ','.join(path_ids)
        hierarchy_types[idx] = ','.join(path_types)

    # Add result columns efficiently
    result_df = df.copy()
    result_df['hierarchy_id'] = hierarchy_ids
    result_df['hierarchy_type'] = hierarchy_types

    result_df.to_parquet(output_file, engine="pyarrow")


# Function to add hierarchy columns using DuckDB
def process_tree_with_duckdb(input_file, output_file):
    # Load parquet file into DuckDB
    df = pd.read_parquet(input_file)
    df = df.sort_values('update_time').reset_index(drop=True)

    con = duckdb.connect()
    con.register('updates', df)

    query = query = """
    WITH RECURSIVE 
    -- First materialize the base data with row numbers for efficiency
    numbered_updates AS MATERIALIZED (
        SELECT 
            *,
            ROW_NUMBER() OVER () as rn
        FROM updates
    ),
    
    -- Build paths iteratively
    paths AS (
        -- Base case: nodes themselves
        SELECT 
            rn,
            child_id as node_id,
            father_id,
            child_type,
            father_type,
            update_time,
            [child_id] as path_ids,
            [child_type] as path_types,
            1 as level
        FROM numbered_updates
        
        UNION ALL
        
        -- Recursive case: add parents
        SELECT 
            p.rn,
            u.father_id as node_id,
            u.father_id,
            u.father_type as node_type,
            u.father_type,
            p.update_time,
            array_prepend(u.father_id, p.path_ids) as path_ids,
            array_prepend(u.father_type, p.path_types) as path_types,
            p.level + 1 as level
        FROM paths p
        JOIN numbered_updates u 
            ON p.father_id = u.child_id
            AND u.update_time <= p.update_time
            AND u.father_id IS NOT NULL
    ),
    
    -- Get the complete paths
    complete_paths AS MATERIALIZED (
        SELECT 
            rn,
            path_ids,
            path_types,
            ROW_NUMBER() OVER (
                PARTITION BY rn 
                ORDER BY level DESC
            ) as path_rank
        FROM paths
    )
    
    -- Final selection joining with original data
    SELECT 
        u.*,
        CASE 
            WHEN u.father_id IS NULL THEN u.child_id
            ELSE array_to_string(p.path_ids, ',')
        END as hierarchy_id,
        CASE 
            WHEN u.father_id IS NULL THEN u.child_type
            ELSE array_to_string(p.path_types, ',')
        END as hierarchy_type
    FROM numbered_updates u
    LEFT JOIN complete_paths p 
        ON u.rn = p.rn 
        AND p.path_rank = 1
    ORDER BY u.rn;
    """

    result = con.execute(query).df()
    result.to_parquet(output_file, engine='pyarrow')
    con.close()

    print(f"Processed file with DuckDB and saved to {output_file}.")


if __name__ == '__main__':
    start = timer()
    process_tree_with_duckdb('data/tree_updates.parquet', 'data/tree_updates_hierarchy_duckdb.parquet')
    end = timer()
    print(f'Dubkdb took : {end - start}')
    start = timer()
    process_tree_updates('data/tree_updates.parquet', 'data/tree_updates_hierarchy_pandas.parquet')
    end = timer()
    print(f'Pandas took : {end - start}')
