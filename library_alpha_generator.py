import os
import random
import pandas as pd

# Define the data fields for each region from the handwritten notebook
REGIONS = {
    "GLB": {
        "supporting": {
            "Analyst": ["analyst10", "analyst69", "analyst48"],
            "Earnings": ["ern3"],
            "Fundamental": ["fnd23", "fnd44"],
            "Imbalance": ["imbalance5"],
            "Model": ["md177", "md138", "md110", "md1238", "md1219", "md1243", "md1138"],
            "Other": ["oth335", "oth496", "oth296"],
            "Risk": ["rsk70", "rsk88"],
            "Sentiment": ["snt21"]
        },
        "booster": ["oth335", "rsk70", "md138", "md1238", "md110", "analyst69"]
    },
    "EURDI": {
        "supporting": {
            "Analyst": ["analyst10", "analyst69", "analyst39"],
            "Earnings": ["ern3"],
            "Fundamental": ["fnd17", "fnd23", "fnd6", "fnd90", "fnd28"],
            "Imbalance": ["imbalance5"],
            "Model": ["md126", "md10", "md138", "md1239", "md1219"],
            "Other": ["oth335"],
            "PriceVolume": ["pvi", "pv27", "pvr", "pv13", "pv97", "pv46"],
            "Risk": ["rsk70", "rsk88"],
            "Sentiment": ["snt21"]
        },
        "booster": ["md138", "oth335", "md126"]
    },
    "ASI": {
        "supporting": {
            "Analyst": ["analyst10", "analyst69", "analyst39", "analyst48"],
            "Earnings": ["ern3"],
            "Fundamental": ["fnd23", "fnd6", "fnd17", "fnd28", "fnd90"],
            "Imbalance": ["imbalance5"],
            "Model": ["md110", "md138", "md117", "md1250", "oth176", "md1219"],
            "Other": ["oth496", "oth335"],
            "PriceVolume": ["pv1", "pv37", "pv29"],
            "Risk": ["rsk70", "rsk88"],
            "Sentiment": ["snt21"]
        },
        "booster": ["oth335", "oth176", "rsk70", "md138", "md1250", "md110", "md1238"]
    },
    "IND": {
        "supporting": {
            "Analyst": ["analyst39"],
            "Earnings": ["ern3", "ern11", "ern6"],
            "Fundamental": ["fnd23", "fnd90", "fnd94", "fnd44", "fnd86"],
            "Imbalance": ["imbalance5"],
            "Model": ["md126", "md138", "md110", "md177", "md1219", "md1177", "md1243", "md1250", "oth176"],
            "Other": ["oth335", "oth128"],
            "PriceVolume": ["md126", "prr", "pv29", "pv30", "pv13"],
            "Risk": ["rsk70", "rsk88"],
            "Sentiment": ["snt21", "snt23"]
        },
        "booster": ["oth335", "md138", "imbalance5"]
    }
}

# Define operators
# Format: (operator_name, requires_lookback)
BOOSTER_OPERATORS = [
    ("rank", False),
    ("scale", False),
    ("ts_rank", True),
    ("ts_entropy", True),
    ("sigmoid", False)
]

SUPPORTING_OPERATORS = [
    ("ts_mean", True),
    ("delay", True),
    ("ts_arg_max", True),
    ("ts_arg_min", True),
    ("sign", False),
    ("ts_backfill", False)
]

LOOKBACKS = [63, 126, 252, 504]

def apply_operator(op_info, field, lookback):
    op_name, req_lookback = op_info
    if req_lookback:
        return f"{op_name}({field}, {lookback})"
    else:
        return f"{op_name}({field})"

def generate_formulas(regions=None, max_per_region=50, seed=42):
    """
    Generate formulas based on the template:
    Alpha = supporting_operator( booster_operator( supporting_field ) ) * booster_operator( booster_field )
    """
    random.seed(seed)
    if regions is None:
        regions = REGIONS.keys()
        
    generated_list = []
    
    for region in regions:
        if region not in REGIONS:
            print(f"Region {region} not found in configurations.")
            continue
            
        region_data = REGIONS[region]
        
        # Flatten all supporting fields for the region
        all_supporting_fields = []
        for cat, fields in region_data["supporting"].items():
            for f in fields:
                all_supporting_fields.append((f, cat))
                
        booster_fields = region_data["booster"]
        
        # Compute total possible combinations for this region
        # Total = len(supp_fields) * len(boost_fields) * len(supp_ops) * len(boost_ops_1) * len(boost_ops_2) * len(lookbacks)
        # To avoid combinatorial explosion, we will randomly sample up to max_per_region unique combinations
        combinations_seen = set()
        attempts = 0
        max_attempts = max_per_region * 50
        
        while len(combinations_seen) < max_per_region and attempts < max_attempts:
            attempts += 1
            
            # Select random components
            supp_field, supp_cat = random.choice(all_supporting_fields)
            boost_field = random.choice(booster_fields)
            
            supp_op = random.choice(SUPPORTING_OPERATORS)
            boost_op_supp = random.choice(BOOSTER_OPERATORS)
            boost_op_boost = random.choice(BOOSTER_OPERATORS)
            
            lookback = random.choice(LOOKBACKS)
            
            key = (supp_field, boost_field, supp_op[0], boost_op_supp[0], boost_op_boost[0], lookback)
            if key in combinations_seen:
                continue
                
            combinations_seen.add(key)
            
            # Build parts of the formula
            # booster_operator( supporting_field )
            inner_supp = apply_operator(boost_op_supp, supp_field, lookback)
            # supporting_operator( inner_supp )
            outer_supp = apply_operator(supp_op, inner_supp, lookback)
            # booster_operator( booster_field )
            outer_boost = apply_operator(boost_op_boost, boost_field, lookback)
            
            # Combine: outer_supp * outer_boost
            formula = f"{outer_supp} * {outer_boost}"
            
            generated_list.append({
                "region": region,
                "supporting_category": supp_cat,
                "supporting_field": supp_field,
                "booster_field": boost_field,
                "supporting_operator": supp_op[0],
                "booster_operator_supp": boost_op_supp[0],
                "booster_operator_boost": boost_op_boost[0],
                "lookback": lookback,
                "formula": formula
            })
            
    df = pd.DataFrame(generated_list)
    return df

if __name__ == "__main__":
    print("Generating alpha formulas offline based on handwritten notebook...")
    df_alphas = generate_formulas(max_per_region=50)
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "generated_alphas.csv")
    df_alphas.to_csv(output_path, index=False)
    print(f"Generated {len(df_alphas)} alpha formulas across GLB, EURDI, ASI, and IND.")
    print(f"Saved formulas to: {output_path}")
    
    # Show a few examples in terminal
    print("\nSample of generated formulas:")
    print(df_alphas[["region", "supporting_field", "booster_field", "lookback", "formula"]].head(10).to_string())
